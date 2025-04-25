import os
import numpy as np
import preprocessing 
from pathlib import Path
import scipy
from spectral import open_image
import imageio
import mbll_functions
import config

class DataLoader():

    # reference values in [1/cm]
    params_ref_gray_matter = np.array([0.0646, 0.0114, 0.0064, 0.0016, 0.73, 0.1, 40.8, 3.089])
    params_ref_gray_matter_fraction = mbll_functions.concentrations_to_blood_fraction(params_ref_gray_matter)
    params_ref_blood_vessel = np.array([1.836, 0.488, 0, 0, 0.55, 0.01, 22.0, 0.660])
    params_ref_blood_vessel_fraction = mbll_functions.concentrations_to_blood_fraction(params_ref_blood_vessel)

    gray_matter_parameters = np.array([0.0646, 0.0114, 0.0064, 0.0016, 0.73, 0.1, 40.8, 3.089])

    tissue_parameters = {
        "gray matter": [ # Digital instrument paper
            np.array([0.0646, 0.0114, 0.0064, 0.0016, 0.73, 0.1]),
            np.array([40.8, 3.089]),
            0.85,
            1.36
        ],
        "artery": [ # Digital instrument paper
            np.array([2.279, 0.0465, 0, 0, 0.55, 0.01]),
            np.array([22.0, 0.660]),
            0.935,
            1.4
        ],
        "vein": [ # Digital instrument paper
            np.array([1.3944, 0.9296, 0, 0, 0.55, 0.01]),
            np.array([22.0, 0.660]),
            0.935,
            1.4
        ],
        "blood vessel average": [ 
            np.array([1.836, 0.488, 0, 0, 0.55, 0.01]),
            np.array([22.0, 0.660]),
            0.935,
            1.4
        ],
        "tumor": [
            # white matter should have approx. half of the absorption of gray matter, and white matter has very similar absorption to gray matter
            np.array([0.0646, 0.0114, 0.0064, 0.0016, 0.73, 0.1]) / 2,
            np.array([33.6, 1.712]), # Jacques Overview
            0.96, # Madsen
            1.4 # typical refractive index of brain tissue, also used by Poulon et. al
        ]
    }


    
    def __init__(self, path, wavelength_left_cut, wavelength_right_cut):
        assert(wavelength_left_cut < wavelength_right_cut)
        self.wavelength_left_cut = wavelength_left_cut
        self.wavelength_right_cut = wavelength_right_cut
        # might be overwritten by subclass
        self.wavelengths = np.arange(wavelength_left_cut, wavelength_right_cut + 1)
        if type(path) == str:
            self.path = Path(path)
        else:
            self.path = path

        # instance method uses loader wavelengths
        self.absorption_coefs = lambda use_diff_oxycco, use_water_and_fat: DataLoader.absorption_coefs(self.wavelengths, use_diff_oxycco, use_water_and_fat)


    @staticmethod
    def absorption_coefs(wavelengths, use_diff_oxycco=False, use_water_and_fat=True):
        spectra_wavelengths = np.loadtxt(config.caredda_spectra / "lambda.txt")

        if not np.all(np.diff(spectra_wavelengths) > 0):
            raise ValueError("Error in spectra wavelenghts: Values not ascending.")
    
        if wavelengths[0] < spectra_wavelengths[0] or wavelengths[-1] > spectra_wavelengths[-1]:
            raise ValueError("Requested wavelength outside of spectra wavelengths.")
        
        mu_a_matrix = np.empty((len(wavelengths), 4 + use_water_and_fat * 2))
        
        for i, mol in enumerate(["HbO2", "Hb", "oxCCO", "redCCO"]):
            spectra_vals = np.loadtxt(config.caredda_spectra / f"eps_{mol}.txt") / 1e3 * np.log(10)
            mu_a_matrix[:, i] = np.interp(wavelengths, spectra_wavelengths, spectra_vals)
        
        if use_diff_oxycco:
            mu_a_matrix_diff = np.empty((len(wavelengths), mu_a_matrix.shape[1]-1))
            mu_a_matrix_diff[:, :3] = mu_a_matrix[:, :3]
            mu_a_matrix_diff[:, 2] -= mu_a_matrix[3]
            mu_a_matrix = mu_a_matrix_diff
        
        if not use_water_and_fat:
            return mu_a_matrix
        
        for i, mol in enumerate(["H2O", "Fat"]):
            spectra_vals = np.loadtxt(config.caredda_spectra / f"mua_{mol}.txt")
            mu_a_matrix[:, -2 + i] = np.interp(wavelengths, spectra_wavelengths, spectra_vals)
        
        return mu_a_matrix
        
    # absorption coefficients in 1/(mM*cm)
    def absorption_coefs_old(self, use_diff_oxycco=True, use_water_and_fat=False):
        molecules, x = preprocessing.read_molecules(self.wavelength_left_cut, self.wavelength_right_cut, self.wavelengths)
        y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water, y_fat = molecules
        if not (x == self.wavelengths).all():
            raise Exception("Wavelengths were changed.")
        # molecule_names = ["HbO2", "Hbb", "diff oxyCCO"]
        if not use_diff_oxycco:
            mu_a_matrix = np.transpose(np.vstack((
                np.asarray(y_hbo2_f),
                np.asarray(y_hb_f),
                np.asarray(y_coxa),
                np.asarray(y_creda)
            )))
        else:
            mu_a_matrix = np.transpose(np.vstack((
                np.asarray(y_hbo2_f),
                np.asarray(y_hb_f),
                np.asarray(y_coxa) - np.asarray(y_creda)
            )))
        
        if use_water_and_fat:
            mu_a_matrix = np.hstack((
                mu_a_matrix,
                np.asarray(y_water)[None, :].transpose(),
                np.asarray(y_fat)[None, :].transpose(),
        ))
        
        return mu_a_matrix

    # wavelengths in nm, result in cm^-1
    def absorption_coef_background(self):
        return 7.84e8*self.wavelengths**(-3.255)

    @staticmethod
    def mu_a_func_gray_matter(wl):
        mu_a_matrix = DataLoader.absorption_coefs(wl, use_diff_oxycco=False, use_water_and_fat=True)
        return mu_a_matrix @ DataLoader.params_ref_gray_matter[:6]
    
    @staticmethod
    def mu_a_func_blood_vessel(wl):
        mu_a_matrix = DataLoader.absorption_coefs(wl, use_diff_oxycco=False, use_water_and_fat=True)
        return mu_a_matrix @ DataLoader.params_ref_blood_vessel[:6]
    
    @staticmethod
    def mu_s_red_func_gray_matter(wl):
        return DataLoader.params_ref_gray_matter[-2] * np.power((wl/500), -DataLoader.params_ref_gray_matter[-1])
    
    @staticmethod
    def mu_s_red_func_blood_vessel(wl):
        return DataLoader.params_ref_blood_vessel[-2] * np.power((wl/500), -DataLoader.params_ref_blood_vessel[-1])
    

    @staticmethod
    def mu_a_func_tissue(wl, tissue):
        mu_a_matrix = DataLoader.absorption_coefs(wl, use_diff_oxycco=False, use_water_and_fat=True)
        return mu_a_matrix @ DataLoader.tissue_parameters[tissue][0]
    
    @staticmethod
    def mu_s_red_func_tissue(wl, tissue):
        a, b = DataLoader.tissue_parameters[tissue][1]
        return a * np.power(wl/500, -b)
       

class DataLoaderNIRS(DataLoader):

    def __init__(self, path, wavelength_left_cut=740, wavelength_right_cut=900, spectrum_cut=1000):
        super().__init__(path, wavelength_left_cut, wavelength_right_cut)
        self.spectrum_cut = spectrum_cut
        self.piglet_id = None
    
    def load_data(self, piglet_id):
        p = self.path / piglet_id
        img_files = list(map(lambda path: path.name, p.glob(piglet_id + "*.mat")))
        img_file = list(filter(lambda path: "DarkCount" not in path, img_files))[-1]
        img_darkcount_file = list(filter(lambda path: "DarkCount" in path, img_files))[-1]
        img_whitecount_file = "refSpectrum.mat"
        if piglet_id == "LWP498":
            img_file = "LWP498_Ws_24Apr2017_15.mat"
            img_darkcount_file = "LWP498 _DarkCount_24Apr2017.mat"
        print(f"Loading data from:\nImage file: {img_file}\nDarkcount file: {img_darkcount_file}\nWhitecount file: {img_whitecount_file}")
        img = scipy.io.loadmat(p / img_file)
        img_darkcount = scipy.io.loadmat(p / img_darkcount_file)
        img_whitecount = scipy.io.loadmat(p / img_whitecount_file)

        wavelengths = img['wavelengths'].astype(float)
        idx = (wavelengths >= self.wavelength_left_cut) & (wavelengths <= self.wavelength_right_cut)
        self.wavelengths = wavelengths[idx]
        self.white_full = img_whitecount['refSpectrum'].astype(float)[idx.squeeze()]
        self.dark_full = img_darkcount['DarkCount'].astype(float)[idx.squeeze()]
        self.spectra = img['spectralDataAll'].astype(float)[idx.squeeze()][:, :self.spectrum_cut]
        self.reflectance = (self.spectra - self.dark_full[:, 0][:, None]) / (self.white_full[:, 0] - self.dark_full[:, 0])[:, None]
        self.reflectance[self.reflectance <= 0] = 0.0001
        self.concentrations_paper = img["AllConcentration"].T # convert to (num_molecules, num_spectra)
        # switch first two columns from (Hbb, Hb02) -> (Hb02, Hbb)
        self.concentrations_paper[[0, 1], :] = self.concentrations_paper[[1, 0], :]

        self.piglet_id = piglet_id

    
    def get_attenuation(self, piglet_id):
        if not self.piglet_id:
            self.load_data(piglet_id)
        
        return -np.log(self.reflectance)
        


    def get_attenuation_change(self, piglet_id, reference_idx):
        if self.piglet_id != piglet_id:
            self.load_data(piglet_id)
        
        #self.reference_spectr = (self.spectra[:, reference_idx] - self.dark_full[:, 0]) / (self.white_full[:, 0] - self.dark_full[:, 0])
        #self.reference_spectr = self.reference_spectr[:, None]
        #self.reference_spectr[self.reference_spectr <= 0] = 0.0001

        #normed_spectr = (self.spectra - self.dark_full[:, 0][:, None]) / (self.white_full[:, 0] - self.dark_full[:, 0])[:, None]
        #normed_spectr[normed_spectr <= 0] = 0.0001

        self.reference_reflectance = self.reflectance[:, [reference_idx]]


        return -np.log(self.reflectance/self.reference_reflectance) # index by [wavelength, index]



class DataLoaderHELICOID(DataLoader):
    
    def __init__(self, path, wavelength_left_cut, wavelength_right_cut, num_wavelengths=None):
        super().__init__(path, wavelength_left_cut, wavelength_right_cut)
        self.patient_id = None
        self.num_wavelengths = num_wavelengths

        self.wavelengths = np.linspace(400, 1000, 826)
        wavelength_idxs = (self.wavelengths >= self.wavelength_left_cut) & (self.wavelengths <= self.wavelength_right_cut)
        if self.num_wavelengths is not None:
            first_idx, last_idx = np.nonzero(wavelength_idxs)[0][[0, -1]]
            keep_idxs = np.round(np.linspace(first_idx, last_idx - 1, self.num_wavelengths)).astype(int)
            new_idxs = np.array([False] * len(wavelength_idxs))
            new_idxs[keep_idxs] = True
            wavelength_idxs &= new_idxs
        self.wavelength_idxs = wavelength_idxs
        self.wavelengths = self.wavelengths[wavelength_idxs]
    
    def load_data(self, patient_id):
        p = self.path / patient_id
        raw_data = open_image(p / "raw.hdr").load()
        white_reference = open_image(p / "whiteReference.hdr").load()
        dark_reference = open_image(p / "darkReference.hdr").load()
        self.label_map = np.squeeze(open_image(p / "gtMap.hdr").load())
        self.reflectance = (raw_data - dark_reference) / (white_reference - dark_reference)
        self.reflectance = np.transpose(self.reflectance, (2, 0, 1)) # move wavelength axis to front
        self.reflectance = self.reflectance[self.wavelength_idxs, ...] # choose desired wavelengths
        self.reflectance[self.reflectance <= 0] = 1e-9 #prevent 0 value in log

        self.patient_id = patient_id

        self.img = np.asarray(imageio.imread(p / "image.jpg"))
        

    def get_attenuation(self, patient_id):
        if not self.patient_id == patient_id:
            self.load_data(patient_id)
        
        return - np.log(self.reflectance.copy())

    def get_attenuation_change(self, patient_id, reference_label="normal", use_average_reference=False):
        if not self.patient_id == patient_id:
            self.load_data(patient_id)
        
        if reference_label == "gray matter":
            reference_label = "normal"

        labels = ["unlabeled", "normal", "tumor", "blood", "background"]
        label_num = 0
        while labels[label_num] != reference_label:
            label_num += 1
        
        label_idxs = np.nonzero(self.label_map == label_num) # returns two lists
        pixel_idx = len(label_idxs[0]) // 3
        self.reference_pixel = np.array([label_idxs[0][pixel_idx], label_idxs[1][pixel_idx]])
        # reference pixel as index in all tissue pixels 
        self.reference_pixel_tissue_ctr = pixel_idx
        
        self.reference_reflectance = self.reflectance[:, self.reference_pixel[0], self.reference_pixel[1]]

        return -np.log(self.reflectance / self.reference_reflectance[:, np.newaxis, np.newaxis])

        






