from scipy.linalg import pinv
from scipy.optimize import least_squares
from tqdm import tqdm
import numpy as np
from mbll_functions import *

def concentr_diffs_no_scattering(delta_A, mu_a_matrix, mu_a_inverted=False):
    mu_a_matrix_pinv = pinv(mu_a_matrix)
    delta_c = mu_a_matrix_pinv @ delta_A
    return delta_c

# if you want to apply cut to spectra, i.e. not take all into account,
# pass smaller array for delta_A
def concentr_diffs_scattering_2(delta_A, mu_a_matrix, wavelengths, b_ref, max_b):
    def f(x, delta_A, mu_a_matrix, wavelengths, b_ref):
        _, num_molecules = mu_a_matrix.shape
        delta_c, b_t = np.split(x, [num_molecules])
        delta_A_no_scattering_val = delta_A_no_scattering(delta_c, mu_a_matrix)
        return delta_A_scattering_2(delta_A_no_scattering_val, wavelengths, b_ref, b_t) - delta_A
    
    num_wavelengths, num_spectra = delta_A.shape
    _, num_molecules = mu_a_matrix.shape
    left_bound = np.append(np.ones(num_molecules)*(-np.inf), [-1])
    right_bound = np.append(np.ones(num_molecules)*np.inf, [max_b])
    delta_c = np.empty((num_molecules, num_spectra))
    delta_c[:, 0] = np.zeros(num_molecules) # spectrum one is reference, no concentration change
    b_t = np.empty((num_spectra,))
    b_t[0] = b_ref
    cur_x = np.zeros(num_molecules + 1)
    cur_x[-1] = 1
    errors = np.zeros((num_wavelengths, num_spectra))
    for t in tqdm(range(1, num_spectra)):
        res = least_squares(
            f,
            cur_x,
            args=(delta_A[:, t], mu_a_matrix, wavelengths, b_ref),
            bounds=(left_bound, right_bound)
        )
        delta_c[:, t], b_t[t] = np.split(res.x, [num_molecules])
        errors[:, t] = res.cost
        cur_x = res.x

    
    return delta_c, b_t, errors


def concentr_diffs_scattering_1(delta_A, mu_a_matrix, wavelengths, a_ref, b_ref, max_a, max_b):
    def f(x, delta_A, mu_a_matrix, wavelengths, a_ref, b_ref):
        _, num_molecules = mu_a_matrix.shape
        delta_c, a_t, b_t = np.split(x, [num_molecules, num_molecules + 1])
        delta_A_no_scattering_val = delta_A_no_scattering(delta_c, mu_a_matrix)
        return delta_A_scattering_1(delta_A_no_scattering_val, wavelengths, a_ref, b_ref, a_t, b_t) - delta_A
    
    num_wavelengths, num_spectra = delta_A.shape
    _, num_molecules = mu_a_matrix.shape
    left_bound = np.append(np.ones(num_molecules)*(-np.inf), [0, 0])
    right_bound = np.append(np.ones(num_molecules)*np.inf, [max_a, max_b])
    delta_c = np.empty((num_molecules, num_spectra))
    delta_c[:, 0] = np.zeros(num_molecules) # spectrum one is reference, no concentration change
    a_t = np.empty((num_spectra,))
    b_t = np.empty((num_spectra,))
    a_t[0], b_t[0] = a_ref, b_ref
    cur_x = np.zeros(num_molecules + 2)
    cur_x[-1] = 1
    errors = np.zeros((num_wavelengths, num_spectra))
    for t in tqdm(range(1, num_spectra)):
        res = least_squares(
            f,
            cur_x,
            args=(delta_A[:, t], mu_a_matrix, wavelengths, a_ref, b_ref),
            bounds=(left_bound, right_bound)
        )
        delta_c[:, t], a_t[t], b_t[t] = np.split(res.x, [num_molecules, num_molecules + 1])
        errors[:, t] = res.cost
        cur_x = res.x

    
    return delta_c, a_t, b_t, errors


def concentr_diffs_scattering_3(delta_A, mu_a_matrix, wavelengths, b, compute_error=False):
    num_wavelengths, num_spectra = delta_A.shape
    scattering_part = (wavelengths/500)**(-b)
    mu_a_matrix_extended = np.hstack((mu_a_matrix, scattering_part[..., None]))
    mu_a_matrix_extended_inv = pinv(mu_a_matrix_extended)
    res =  mu_a_matrix_extended_inv @ delta_A
    delta_c = res[:-1]
    delta_a = res[-1]
    if compute_error:
        error = np.zeros((num_wavelengths, num_spectra))
        delta_A_no_scattering_vals = delta_A_no_scattering(delta_c, mu_a_matrix)
        delta_A_scattering_vals = delta_A_scattering_3(
            delta_A_no_scattering_vals,
            wavelengths,
            delta_a,
            b
        )
        error = delta_A - delta_A_scattering_vals
        return delta_c, delta_a, error
    return delta_c, delta_a

def concentr_diffs_scattering_4(delta_A, mu_a_matrix, wavelengths, max_a, max_b):
    return concentr_diffs_scattering_1(delta_A, mu_a_matrix, wavelengths, 0.0, 1.0, max_a, max_b)


# hyperparameter search + concentration calculation
def fit_scattering_2(delta_A, mu_a_matrix, wavelengths, max_b, resolution=10):
    num_wavelengths, num_molecules = mu_a_matrix.shape
    _, num_spectra = delta_A.shape
    b_ref_arr = np.linspace(0, max_b, resolution)
    errors_total = np.empty((resolution, num_wavelengths, num_spectra))
    b_t_total = np.empty((resolution, num_spectra))
    delta_c_total = np.empty((resolution, num_molecules, num_spectra))

    for i in tqdm(range(resolution)):
        delta_c, b_t, errors = concentr_diffs_scattering_2(delta_A, mu_a_matrix, wavelengths, b_ref_arr[i], max_b)
        delta_c_total[i] = delta_c
        b_t_total[i] = b_t
        errors_total[i] = errors
    
    errors_reduced = np.sum(errors_total**2, axis=(1, 2))
    min_idx = np.argmin(errors_reduced)
    return delta_c_total[min_idx], b_t_total[min_idx], b_ref_arr[min_idx], errors_reduced

def fit_scattering_3(delta_A, mu_a_matrix, wavelengths, max_b, resolution=10):
    num_wavelengths, num_molecules = mu_a_matrix.shape
    _, num_spectra = delta_A.shape
    b_arr = np.linspace(0, max_b, resolution)
    errors_total = np.empty((resolution, num_wavelengths, num_spectra))
    delta_c_total = np.empty((resolution, num_molecules, num_spectra))
    delta_a_total = np.empty((resolution, num_spectra))


    for i in tqdm(range(resolution)):
        delta_c, delta_a, error = concentr_diffs_scattering_3(delta_A, mu_a_matrix, wavelengths, b_arr[i], compute_error=True)
        delta_c_total[i] = delta_c
        delta_a_total[i] = delta_a
        errors_total[i] = error
    
    errors_reduced = np.sum(errors_total**2, axis=(1, 2))
    min_idx = np.argmin(errors_reduced)
    return delta_c_total[min_idx], delta_a_total[min_idx], b_arr[min_idx], errors_reduced


def fit_scattering_1(delta_A, mu_a_matrix, wavelengths, max_a, max_b, resolution=10):
    num_wavelengths, num_molecules = mu_a_matrix.shape
    _, num_spectra = delta_A.shape
    a_ref_arr = np.linspace(0, max_a, resolution)
    b_ref_arr = np.linspace(0, max_b, resolution)
    a_ref_arr, b_ref_arr = np.meshgrid(a_ref_arr, b_ref_arr)
    errors_total = np.empty((resolution, resolution, num_wavelengths, num_spectra))
    a_t_total = np.empty((resolution, resolution, num_spectra))
    b_t_total = np.empty((resolution, resolution, num_spectra))
    delta_c_total = np.empty((resolution, resolution, num_molecules, num_spectra))

    for i in tqdm(range(resolution)):
        for j in range(resolution):
            delta_c, a_t, b_t, errors = concentr_diffs_scattering_1(delta_A, mu_a_matrix, wavelengths, a_ref_arr[i, j], b_ref_arr[i, j], max_a, max_b)
            delta_c_total[i, j] = delta_c
            a_t_total[i, j] = a_t
            b_t_total[i, j] = b_t
            errors_total[i, j] = errors
        
    
    errors_reduced = np.sum(errors_total**2, axis=(2, 3))
    min_idxs = np.unravel_index(np.argmin(errors_reduced), errors_reduced.shape)
    return delta_c_total[min_idxs], a_t_total[min_idxs], b_t_total[min_idxs], a_ref_arr[min_idxs], b_ref_arr[min_idxs], errors_reduced

## no hyperparameters for scenario 4




