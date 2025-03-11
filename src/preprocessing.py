import bisect
import os
import numpy as np

from config import spectra_dir


# load spectra from txt files into dictionary
def load_spectra():
    path_dict = {}
    path_dict["cytoa_oxy_520"] = str(spectra_dir) + "/moody cyt aa3 oxidised.txt"
    path_dict["cytoa_red_520"] = str(spectra_dir) + "/moody cyt aa3 reduced.txt"
    path_dict["cytoa_oxy_410"] = str(spectra_dir) + "/cco_oxidized_visible.txt"
    path_dict["cytoa_red_410"] = str(spectra_dir) + "/cco_reduced_visible.txt"
    path_dict["hbo2_800"] = str(spectra_dir) + "/hb02.txt"
    path_dict["hbo2_450"] = str(spectra_dir) + "/z_adult_hbo2_450_630.txt"
    path_dict["hbo2_600"] = str(spectra_dir) + "/z_adult_hbo2_600_800.txt"
    path_dict["hb_800"] = str(spectra_dir) + "/hb.txt"
    path_dict["hb_450"] = str(spectra_dir) + "/z_adult_hb_450_630.txt"
    path_dict["hb_600"] = str(spectra_dir) + "/z_adult_hb_600_800.txt"
    path_dict["water_599"] = str(spectra_dir) + "/matcher94_nir_water_37.txt" # extinction
    path_dict["water_380"] = str(spectra_dir) + "/pope97_water_380_727.txt" # absorption
    path_dict["fat"] = str(spectra_dir) + "/fat.txt"
    return path_dict


### reading cpectra from .txt
def read_spectra(file_name):
    with open(file_name, 'r') as data:
        x, y = [], []
        for line in data:
            p = line.split()
            if line != "\n" and not p[0] == '\x00':
                x.append(float(p[0]))
                y.append(float(p[1]))
    return np.array(x), np.array(y)



def cut_spectra(x, y, left_cut, right_cut):
    """
    cuts off spectrogram according to cut values
    """
    ix_left = np.where(x == left_cut)[0][0]
    ix_right = np.where(x == right_cut)[0][0]
    return y[ix_left:ix_right + 1]


def wave_interpolation(y, x, mol_list, x_waves):
    """
    interpolate spectrogram values according to x_waves
    """
    lower_bound, upper_bound = x[0], x[-1]
    new_x = np.asarray([i for i in x_waves if lower_bound <= i <= upper_bound])

    new_y = {}
    for i in mol_list:
        new_y[i] = np.interp(new_x, x, y[i])

    return new_y, new_x


def merge_with_average(x1, x2, y1, y2):
        # take two sets of arrays with increasing x-values
        # merge them, average y-values of x-values appearing twice
        assert(x1[0] <= x2[0])   

        overlap_idx_high, overlap_idx_low = 0, -1

        # idx of last value in lower arr which is smaller or equal to first val in upper arr
        while x1[overlap_idx_low] > x2[0]:
            overlap_idx_low -= 1

        # idx of first val in upper arr which is greater to last val in lower arr
        while x2[overlap_idx_high] <= x1[-1]:
            overlap_idx_high += 1
        
        x_overlap = []
        y_overlap = []
        idx_low, idx_high = overlap_idx_low, 0

        while idx_high < overlap_idx_high:
            xval_low = x1[idx_low]
            xval_high = x2[idx_high]

            if xval_low < xval_high:
                x_overlap.append(xval_low)
                y_overlap.append(y1[idx_low])
                idx_low += 1
            elif xval_low == xval_high:
                x_overlap.append(xval_low)
                y_overlap.append((y1[idx_low] + y2[idx_high])/2)
                idx_low += 1
                idx_high += 1
            else:
                x_overlap.append(xval_high)
                y_overlap.append(y2[idx_high])
                idx_high += 1

        # will also work if idx_low is greater than array size
        if idx_low != 0:
            x_overlap = np.concatenate([x_overlap, x1[idx_low:]])
            y_overlap = np.concatenate([y_overlap, y1[idx_low:]])
        
        x_concat = np.concatenate([x1[:overlap_idx_low], x_overlap, x2[overlap_idx_high:]])
        y_concat = np.concatenate([y1[:overlap_idx_low], y_overlap, y2[overlap_idx_high:]])

        return x_concat, y_concat


def read_molecules(left_cut, right_cut, x_waves=None):
    path_dict = load_spectra()

    # read spectra for: cytochrome oxydised/reduced, oxyhemoglobin, hemoglobin, water, fat
    mol_list = ["cytoa_oxy_410", "cytoa_red_410", "cytoa_oxy_520", "cytoa_red_520", "hbo2_800", "hbo2_450", "hbo2_600", "hb_800", "hb_450", "hb_600", "water_599", "water_380", "fat"]
    x, y = {}, {}
    for i in mol_list:
        x[i], y[i] = read_spectra(path_dict[i])

    # from extinction to absorption
    # TODO check if water spectra was in extinction 
    y_list = ['hb_450', 'hb_600', 'hbo2_450', 'hbo2_600', "cytoa_oxy_410", "cytoa_red_410", "cytoa_oxy_520", "cytoa_red_520", 'water_599', "water_380"]
    for i in y_list:
        y[i] *= 2.3025851

    # from mm and micromole to cm and millimole, get rid of mole
    y["hbo2_800"] *= 10 * 1000 #/ 10
    y["hb_800"] *= 10 * 1000 #/ 10

    #y["cytoa_oxy"] /= 1000
    #y["cytoa_red"] /= 1000

    # from m to cm
    y["fat"] /= 100

    x["hbo2"], y["hbo2"] = merge_with_average(x["hbo2_450"], x["hbo2_600"], y["hbo2_450"], y["hbo2_600"])
    x["hb"], y["hb"] = merge_with_average(x["hb_450"], x["hb_600"], y["hb_450"], y["hb_600"])
    x["cytoa_oxy"], y["cytoa_oxy"] = merge_with_average(x["cytoa_oxy_410"], x["cytoa_oxy_520"], y["cytoa_oxy_410"], y["cytoa_oxy_520"])
    x["cytoa_red"], y["cytoa_red"] = merge_with_average(x["cytoa_red_410"], x["cytoa_red_520"], y["cytoa_red_410"], y["cytoa_red_520"])

    for mol_str in ["hbo2", "hb"]:
        assert(np.all(np.diff(x[mol_str]) > 0))
        x_interp = np.arange(x[mol_str][0], x[mol_str][-1] + 1)
        y_interp = np.interp(x_interp, x[mol_str], y[mol_str])
        x[mol_str] = np.concatenate([x_interp, x[mol_str + "_800"][151:]])
        y[mol_str] = np.concatenate((y_interp, np.asarray(y[mol_str + "_800"][151:])), axis=None)

    # water spectrum from matcher94 only accurate starting from about 700nm
    idx_700 = bisect.bisect_left(x["water_599"], 700)
    x["water_700"] = x["water_599"][idx_700:]
    y["water_700"] = y["water_599"][idx_700:]

    x["water"], y["water"] = merge_with_average(x["water_380"], x["water_700"], y["water_380"], y["water_700"])
    # water spectra are given for a pure solution in units of (1/cm)
    # to convert to (1/(mM*cm)) we divide by Molar concentration of pure Water (55400mM)
    # y["water"] /= 55400
    ### instead model water as volume fraction in [0, 1]

    assert(np.all(np.diff(x["water"]) > 0))
    x_interp = np.arange(x["water"][0], x["water"][-1]+1)
    y["water"] = np.interp(x_interp, x["water"], y["water"])
    x["water"] = x_interp

    # cutting all spectra to the range [left_cut, right_cut] nm
    x_new = x["cytoa_oxy"][bisect.bisect_left(x["cytoa_oxy"], left_cut):bisect.bisect_right(x["cytoa_oxy"], right_cut)]
    mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red", "water", "fat"]
    for i in mol_list:
        y[i] = cut_spectra(x[i], y[i], left_cut, right_cut)

    if x_waves is not None:
        y, x_new = wave_interpolation(y, x_new, mol_list, x_waves)

    return [y[i] for i in mol_list], x_new


def read_wavelengths(path):
    f = open(path, "r")
    txt = f.read().split("{")[-1].strip("}")
    txt = txt.split(",")
    x = [float(i) for i in txt]
    return x
