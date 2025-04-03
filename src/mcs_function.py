import numpy as np
import config
from mbll_functions import blood_fraction_to_concentrations

# attenuation from Monte Carlo Simulation
class SimulationAttenuation():

    def __init__(self, path):
        self.path = path
        data = np.load(path)
        self.nphoton = data["arr_0"]
        self.g = data["arr_1"]
        self.mu_s_vals = data["arr_2"] # expected to be in cm^-1
        self.num_mu_s_vals = len(self.mu_s_vals)
        self.delta_mu_s = self.mu_s_vals[1] - self.mu_s_vals[0]
        self.first_mu_s = self.mu_s_vals[0]
        max_ndetected_photons = max([data[f"arr_{i+3}"].shape[1] for i in range(self.num_mu_s_vals)])
        self.photon_data = np.empty((self.num_mu_s_vals, 2, max_ndetected_photons))
        self.photon_data[:, 0, :] = np.inf # pathlengths expected to be in cm
        self.photon_data[:, 1, :] = 0
        for i in range(self.num_mu_s_vals):
            cur_ndetected_photons = data[f"arr_{i+3}"].shape[1]
            self.photon_data[i, :, :cur_ndetected_photons] = data[f"arr_{i+3}"]

        print(f"Loaded data with {self.nphoton} photons and {self.num_mu_s_vals} values for mu_s.")
    
    def compute_weights(self, mu_a, mu_s_idx):
        return np.exp(-mu_a[..., None] * self.photon_data[mu_s_idx, 0, :])
        
    def A(self, mu_a, mu_s):
        #mu_s_upper_idxs = np.clip(np.searchsorted(self.mu_s_vals, mu_s), 1, self.num_mu_s_vals - 1)
        mu_s_upper_idxs = np.clip(np.ceil((mu_s - self.first_mu_s) / self.delta_mu_s), 1, self.num_mu_s_vals - 1).astype(int)
        weights_upper = self.compute_weights(mu_a, mu_s_upper_idxs)
        weights_lower = self.compute_weights(mu_a, mu_s_upper_idxs - 1)
        A_upper = -np.log(np.sum(weights_upper, axis=-1) / self.nphoton)
        A_lower = -np.log(np.sum(weights_lower, axis=-1) / self.nphoton)
        A_total =  (self.mu_s_vals[mu_s_upper_idxs] - mu_s) * A_lower
        A_total += (mu_s - self.mu_s_vals[mu_s_upper_idxs - 1]) * A_upper
        A_total /= self.delta_mu_s
        return A_total
    
    def A_concentrations(self, wavelengths, mu_a_matrix, c, a, b):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        c = np.atleast_2d(c).reshape(mu_a_matrix.shape[1], -1)

        mu_a = mu_a_matrix @ c
        mu_s_red = a * (wavelengths/500)[:, None] ** (-b)

        return self.A(mu_a, mu_s_red / (1 - self.g))
    
    def A_blood_fraction(self, wavelengths, mu_a_matrix, c, a, b):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        c = np.atleast_2d(c).reshape(mu_a_matrix.shape[1], -1)

        return self.A_concentrations(
            wavelengths,
            mu_a_matrix,
            blood_fraction_to_concentrations(c),
            a,
            b
        )

    # returns dA(mu_a, mu_s)/dmu_a and dA/dmu_s
    # if shape_mu is the shape of mu_a and mu_s, then return value has shape shape_mu + (2,)
    def jacobian(self, mu_a, mu_s):
        #mu_s_upper_idxs = np.clip(np.searchsorted(self.mu_s_vals, mu_s), 1, self.num_mu_s_vals - 1)
        mu_s_upper_idxs = np.clip(np.ceil((mu_s - self.first_mu_s) / self.delta_mu_s), 1, self.num_mu_s_vals - 1).astype(int)
        weights_upper = self.compute_weights(mu_a, mu_s_upper_idxs)
        weights_lower = self.compute_weights(mu_a, mu_s_upper_idxs - 1)
        total_weights_upper = np.sum(weights_upper, axis=-1)
        total_weights_lower = np.sum(weights_lower, axis=-1)
        A_upper = -np.log(total_weights_upper / self.nphoton)
        A_lower = -np.log(total_weights_lower / self.nphoton)
        weighted_pl_upper = np.sum(weights_upper * np.nan_to_num(self.photon_data[mu_s_upper_idxs, 0, :]), axis=-1) / total_weights_upper
        weighted_pl_lower = np.sum(weights_lower * np.nan_to_num(self.photon_data[mu_s_upper_idxs - 1, 0, :]), axis=-1) / total_weights_lower

        jacobian = np.empty((mu_a.shape + (2,)))
        jacobian[..., 0] = (mu_s - self.mu_s_vals[mu_s_upper_idxs - 1]) * weighted_pl_upper
        jacobian[..., 0] += (self.mu_s_vals[mu_s_upper_idxs] - mu_s) * weighted_pl_lower
        jacobian[..., 0] /= self.delta_mu_s
        jacobian[..., 1] = (A_upper - A_lower) / self.delta_mu_s

        return jacobian

    # jacobian for all concentrations and a parameter
    def jacobian_concentrations(self, wavelengths, mu_a_matrix, c, a, b):
        num_molecules = mu_a_matrix.shape[1]
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        c = np.atleast_2d(c).reshape(num_molecules, -1)
        num_spectra = c.shape[1]

        mu_a = mu_a_matrix @ c
        mu_s_no_a = ((wavelengths/500)[:, None] ** (-b)) / (1 - self.g)
        mu_s = mu_s_no_a * a

        jacobian_c = np.empty((len(wavelengths), num_spectra, num_molecules + 2))
        jacobian_base = self.jacobian(mu_a, mu_s)
        jacobian_c[..., :num_molecules] = jacobian_base[..., [0]]
        jacobian_c[..., num_molecules:] = jacobian_base[..., [1]]
        jacobian_c[..., :num_molecules] *= mu_a_matrix[:, None, :]
        jacobian_c[..., -2] *= mu_s_no_a
        jacobian_c[..., -1] *= -mu_s * np.log(wavelengths/500)[:, None]
        
        # first fill with dA/dmu_A and dA/dmu_s
        #weighted_nscat_upper = np.sum(weights_upper * self.photon_data[mu_s_upper_idxs, 1, :], axis=-1) / total_weights_upper
        #weighted_nscat_lower = np.sum(weights_lower * self.photon_data[mu_s_upper_idxs - 1, 1, :], axis=-1) / total_weights_lower
        #jacobian = np.empty((len(wavelengths), num_molecules + 2, num_spectra))
        #jacobian[:, :num_molecules, :] = ((mu_s - self.mu_s_vals[mu_s_upper_idxs - 1]) * weighted_pl_upper + (self.mu_s_vals[mu_s_upper_idxs] - mu_s) * weighted_pl_lower)[:, None, :]
        #jacobian[:, :num_molecules, :] /= self.delta_mu_s
        #jacobian[:, :num_molecules, :] *= mu_a_matrix[:, :, None]
        #jacobian[:, -2:, :] = (A_upper - A_lower)[:, None, :]
        #jacobian[:, -2:, :] += ((mu_s - self.mu_s_vals[mu_s_upper_idxs - 1]) * (weighted_pl_upper + weighted_nscat_upper / mu_s))[:, None,:]
        #jacobian[:, -2:, :] += ((self.mu_s_vals[mu_s_upper_idxs] - mu_s) * (weighted_pl_lower + weighted_nscat_lower / mu_s))[:, None,:]
        #jacobian[:, -2:, :] /= self.delta_mu_s
        #jacobian[:, -2, :] *= mu_s_red_no_a
        #jacobian[:, -1, :] *= -a * b * ((wavelengths/500)[:, None] ** (-b-1))        

        return jacobian_c

    
    def jacobian_blood_fraction(self, wavelengths, mu_a_matrix, c, a, b):
        num_molecules = mu_a_matrix.shape[1]
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        c = np.atleast_2d(c).reshape(num_molecules, -1)
        num_spectra = c.shape[1]

        f_blood, StO2 = c[:2, :]
        c_HbO2_pure = config.c_pure_HbT * StO2
        c_Hbb_pure = config.c_pure_HbT * (1 - StO2)
        c_HbO2 = c_HbO2_pure * f_blood
        c_Hbb = c_Hbb_pure * f_blood

        mu_a = mu_a_matrix @ np.row_stack((c_HbO2, c_Hbb, c[2:]))
        mu_s_no_a = ((wavelengths/500)[:, None] ** (-b)) / (1-self.g)
        mu_s = mu_s_no_a * a

        jacobian_f = np.empty((len(wavelengths), num_spectra, num_molecules + 2))
        jacobian_base = self.jacobian(mu_a, mu_s)
        jacobian_f[..., :num_molecules] = jacobian_base[..., [0]]
        jacobian_f[..., num_molecules:] = jacobian_base[..., [1]]
        jacobian_f[..., 0] *= c_HbO2_pure[None, :] * mu_a_matrix[:, None, 0] + c_Hbb_pure[None, :] * mu_a_matrix[:, None, 1]
        jacobian_f[..., 1] *= f_blood[None, :] * config.c_pure_HbT * (mu_a_matrix[:, None, 0] - mu_a_matrix[:, None, 1])
        jacobian_f[..., 2:num_molecules] *= mu_a_matrix[:, None, 2:]
        jacobian_f[..., -2] *= mu_s_no_a
        jacobian_f[..., -1] *= -mu_s * np.log(wavelengths/500)[:, None]

        return jacobian_f

