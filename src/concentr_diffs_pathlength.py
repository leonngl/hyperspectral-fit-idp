from mbll_functions import *
from tqdm.notebook import tqdm, tqdm_notebook
from scipy.optimize import least_squares, minimize, Bounds, LinearConstraint, SR1
from ray import tune, train
import ray
import numpy as np
from scipy.linalg import pinv
import math
import concurrent.futures
from functools import partial
import multiprocessing


def concentr_fit_mbll(A, wavelengths, mu_a_matrix, pathlengths):
    pathlenghts = np.atleast_2d(pathlengths).reshape(wavelengths.shape[0], -1)
    mu_a_matrix_pinv = pinv(mu_a_matrix)
    return mu_a_matrix_pinv  @ (A / pathlengths)

class A_mbll_fit():

    def __init__(
        self,
        wavelengths,
        mu_a_matrix,
        tissue_types,
        # these paramters have as first dimension tissue type
        pathlength,
        scatterlength,
        baseline_attenuation,
        baseline_concentrations,
        baseline_a,
        baseline_b,
        g
    ):
        
        self.tissue_types = tissue_types
        num_wavelengths, num_molecules = mu_a_matrix.shape
        self.pseudoinverse = dict()
        self.offset = dict()

        for i, tissue_type in enumerate(tissue_types):
            cur_pl = np.atleast_2d(pathlength[i]).reshape(num_wavelengths, 1)
            cur_sl = np.atleast_2d(scatterlength[i]).reshape(num_wavelengths, 1)
            cur_g = g[i]
            matrix = np.column_stack(
                (
                    cur_pl * mu_a_matrix,
                    cur_sl / (1-cur_g) * np.power(wavelengths / 500, -baseline_b[i])[..., None]
                )
            )
            self.pseudoinverse[tissue_type] = pinv(matrix)
            self.offset[tissue_type] = -matrix @ np.concatenate(
                (
                    baseline_concentrations[i].reshape(num_molecules),
                    np.array(baseline_a[i]).reshape(1)
                )
            )
            self.offset[tissue_type] += baseline_attenuation[i].reshape(num_wavelengths)
            self.offset[tissue_type] = self.offset[tissue_type].reshape(num_wavelengths, 1)

    def concentr_fit(self, A, tissue_type):
        old_shape = A.shape
        A = np.atleast_2d(A).reshape(old_shape[0], -1)
        res = self.pseudoinverse[tissue_type] @ (A - self.offset[tissue_type])
        return res.reshape(-1, *old_shape[1:])


class A_mbll_fit_delta(A_mbll_fit):

    def __init__(
        self,
        wavelengths,
        mu_a_matrix,
        tissue_types,
        pathlength,
        scatterlength,
        baseline_b,
        g
    ):
        num_tissues = len(tissue_types)
        num_wavelengths, num_concentrations = mu_a_matrix.shape
        super().__init__(
            wavelengths,
            mu_a_matrix,
            tissue_types,
            pathlength,
            scatterlength,
            np.zeros((num_tissues, num_wavelengths)),
            np.zeros((num_tissues, num_concentrations)),
            np.zeros(num_tissues),
            baseline_b,
            g
        )

        assert all(np.allclose(self.offset[tissue], 0) for tissue in tissue_types)
            
    def concentr_fit_delta(self, delta_A, tissue_type):
        return self.concentr_fit(delta_A, tissue_type)
        

# experimental, not used
def concentr_fit_nonlinear_ref_optimization(
    A,
    wavelengths,
    mu_a,
    func,
    ref_idx, # idx of reference spectrum
    ref_vals=None,
    variables_bool_arr=np.array([True]*6),
    left_bounds=None,
    right_bounds=None,
    init_vals=None,
    verbosity=0
):
    
    num_wavelengths, num_spectra = A.shape
    num_molecules = mu_a.shape[1]
    num_vals = len(variables_bool_arr)
    num_params = num_vals - num_molecules
    num_vars = np.count_nonzero(variables_bool_arr)
    num_molecule_vars = np.count_nonzero(variables_bool_arr[:num_molecules])

    if not all(variables_bool_arr) and len(variables_bool_arr) != (num_molecules + 2):
        raise NotImplementedError("Works only when optimizing for all parameters.")
    
    # A is now of size (wavelenghts, spectra)
    def func_wrapper(x, A, ref_idx):
        x = x.reshape(num_vars, num_spectra)
        c_full = np.empty((num_molecules, num_spectra))
        c_full[variables_bool_arr[:num_molecules], :] = x[:num_molecule_vars, :]
        c_full[~variables_bool_arr[:num_molecules], :] = ref_vals[~variables_bool_arr][:(num_molecules - num_molecule_vars)]
        params = np.empty((num_params, num_spectra))
        params[variables_bool_arr[num_molecules:]] = x[num_molecule_vars:]
        params[~variables_bool_arr[num_molecules:]] = ref_vals[~variables_bool_arr][-num_params:]
        A_func = func(wavelengths, mu_a, c_full, *params)

        # will be zero at reference-idx wavelengths
        res = A_func - A_func[:, [ref_idx]] - A - A[:, [ref_idx]]

        return res.reshape(-1)

    if left_bounds.shape != right_bounds.shape:
        raise RuntimeError("Arrays for left and right bounds should have the same shape.")
    if len(left_bounds) == num_vals and num_vars != num_vals:
        print("Note: Bounds for parameters " + ", ".join(map(str, np.arange(num_vals)[~variables_bool_arr])) + " will be ignored, as they are not optimized.")
        left_bounds = left_bounds[variables_bool_arr]
        right_bounds = right_bounds[variables_bool_arr]
    elif len(left_bounds) != num_vars:
        raise RuntimeError("Invalid size for bounds.")

    left_bounds = np.tile(left_bounds, num_spectra)
    right_bounds = np.tile(right_bounds, num_spectra)

    res = least_squares(
        init_vals,
        func_wrapper,
        bounds=(left_bounds, right_bounds),
        args=(A, ref_idx),
        jac_sparsity=None,  ## TODO: This would have to be set to a sparse matrix
        verbose=verbosity
    )

    return np.array(res.x).reshape(num_vars, num_spectra), np.array(res.cost).reshape(num_wavelengths, num_spectra)
        

def concentr_fit_nonlinear_multiple_tissues_concurrent(
    A,
    tissue_idxs,
    wavelengths,
    mu_a,
    func,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    ref_vals=None,
    const_vals=None,
    jacobian=None,
    constraint=None,
    init_vals=None,
    update_init=True,
    progress_bar=False,
    verbosity=0,
    num_processes=6
):
    concentr_fit_nonlinear_curried = partial(
        concentr_fit_nonlinear_multiple_tissues,
        wavelengths=wavelengths,
        mu_a=mu_a,
        func=func,
        ref_vals=ref_vals,
        variables_bool_arr=variables_bool_arr,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
        jacobian=jacobian,
        constraint=constraint,
        init_vals=init_vals,
        update_init=update_init,
        progress_bar=progress_bar,
        verbosity=verbosity
    )

    num_vals = len(variables_bool_arr)
    num_vars = np.count_nonzero(variables_bool_arr)
    _, num_spectra = A.shape
    x = np.empty((num_vars, num_spectra))
    errors = np.empty_like(A)
    num_spectra_per_process = math.ceil(num_spectra / num_processes)
    variables_bool_arr = np.array(variables_bool_arr)

    tissue_types, tissue_idxs_inverse = np.unique(tissue_idxs, return_inverse=True)
    # convert tissue_idxs to ascending ints starting from zero
    num_tissues = len(tissue_types)
    tissue_idxs = np.arange(num_tissues)[tissue_idxs_inverse].astype(int)
    
    if const_vals is None:
        if num_vars != num_vals:
            raise AttributeError("Constant values for non-optimized variables are missing.")
        else:
            const_vals = np.empty((num_tissues, num_vals))
    elif num_tissues > len(const_vals):
        raise AttributeError("Too few const_vals.")

    #remove potentially unused rows, and convert to np.array
    const_vals = np.array(const_vals[:num_tissues])

    if const_vals.shape[1] == (num_vals - num_vars):
        const_vals_tmp = np.empty((num_tissues, num_vals))
        const_vals_tmp[:, ~variables_bool_arr] = const_vals
        const_vals = const_vals_tmp
    elif const_vals.shape[1] != num_vals:
        raise AttributeError("Dimension 1 of const_vals to short.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for t in range(0, num_spectra, num_spectra_per_process):
            cur_tissue_idxs = np.unique(tissue_idxs[t:t+num_spectra_per_process])
            cur_const_vals = const_vals[cur_tissue_idxs]
            future = executor.submit(
                concentr_fit_nonlinear_curried,
                A=A[:, t:t+num_spectra_per_process],
                tissue_idxs=tissue_idxs[t:t+num_spectra_per_process],
                const_vals=cur_const_vals
            )
        
            futures.append(future)
        
        for process_idx, t in enumerate(range(0, num_spectra, num_spectra_per_process)):
            x[:, t:t+num_spectra_per_process], errors[:, t:t+num_spectra_per_process] = futures[process_idx].result()
    
    return x, errors


def concentr_fit_nonlinear_single_tissue_concurrent(
    A,
    wavelengths,
    mu_a,
    func,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    ref_vals=None,
    const_vals=None,
    jacobian=None,
    constraint=None,
    init_vals=None,
    update_init=True,
    progress_bar=False,
    verbosity=0,
    num_processes=6
):
    
    A = np.atleast_2d(A).reshape(A.shape[0], -1)
    tissue_idxs = np.zeros(A.shape[1])
    if const_vals is not None and len(np.array(const_vals).shape) == 1:
        const_vals = [np.array(const_vals)]

    return concentr_fit_nonlinear_multiple_tissues_concurrent(
        A=A,
        tissue_idxs=tissue_idxs,
        wavelengths=wavelengths,
        mu_a=mu_a,
        func=func,
        variables_bool_arr=variables_bool_arr,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
        ref_vals=ref_vals,
        const_vals=const_vals,
        jacobian=jacobian,
        constraint=constraint,
        init_vals=init_vals,
        update_init=update_init,
        progress_bar=progress_bar,
        verbosity=verbosity,
        num_processes=num_processes,
    )


concentr_fit_nonlinear_concurrent = concentr_fit_nonlinear_single_tissue_concurrent


def concentr_fit_nonlinear_multiple_tissues(
    A,
    tissue_idxs,
    wavelengths,
    mu_a,
    func,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    const_vals=None,
    jacobian=None,
    constraint=None,
    ref_vals=None,
    init_vals=None,
    update_init=True,
    ref_val_tissue_idx=0,
    progress_bar=False,
    verbosity=0
):

    A = np.atleast_2d(A).reshape(A.shape[0], -1)
    variables_bool_arr = np.array(variables_bool_arr, dtype=bool)
    num_wavelengths, num_spectra = A.shape
    num_molecules = mu_a.shape[1]
    num_vals = len(variables_bool_arr) # variables in chosen function
    num_params = num_vals - num_molecules
    num_vars = np.count_nonzero(variables_bool_arr) # variables to be optimized
    num_molecule_vars = np.count_nonzero(variables_bool_arr[:num_molecules])

    def func_wrapper(x, A_t, tissue_type, A_ref=None, sum_and_square=False):
        c_full = np.empty((num_molecules,))
        c_full[variables_bool_arr[:num_molecules]] = x[:num_molecule_vars]
        c_full[~variables_bool_arr[:num_molecules]] = const_vals[tissue_type][~variables_bool_arr][:(num_molecules - num_molecule_vars)]
        params = np.empty((num_params,))
        params[variables_bool_arr[num_molecules:]] = x[num_molecule_vars:]
        params[~variables_bool_arr[num_molecules:]] = const_vals[tissue_type][~variables_bool_arr][-num_params:]
        A_func = func(wavelengths, mu_a, c_full, *params)
        res = A_t - A_func[:, 0]
        if A_ref is not None:
            res += A_ref[:, 0]
        if sum_and_square:
            res = np.sum(res**2)
        return res

    # only works if all variables (molecules and params) are optimized
    if not all(variables_bool_arr) and jacobian is not None:
        raise NotImplementedError("To use jacobian, all variables have to be optimized.")

    def jacobian_wrapper(x, *args, **kwargs):
        c = x[:num_molecules]
        params = x[num_molecules:]
        # don't forget to add minus to jacobian!
        jacobian_val = -jacobian(wavelengths, mu_a, c, *params)  
        return jacobian_val[:, 0, :]

    tissue_types, tissue_idxs_inverse = np.unique(tissue_idxs, return_inverse=True)
    # convert tissue_idxs to ascending ints starting from zero
    num_tissues = len(tissue_types)
    tissue_idxs = np.arange(num_tissues)[tissue_idxs_inverse].astype(int)
    

    if const_vals is None:
        if num_vars != num_vals:
            raise AttributeError("Constant values for non-optimized variables are missing.")
        else:
            const_vals = np.empty((num_tissues, num_vals))
    elif num_tissues > len(const_vals):
        raise AttributeError("Too few const_vals.")

    #remove potentially unused rows, and convert to np.array
    const_vals = np.array(const_vals[:num_tissues])

    if num_vars != num_vals:
        if const_vals.shape[1] == (num_vals - num_vars):
            const_vals_tmp = np.empty((num_tissues, num_vals))
            const_vals_tmp[:, ~variables_bool_arr] = const_vals
            const_vals = const_vals_tmp
        elif const_vals.shape[1] != num_vals:
            raise AttributeError("Dimension 1 of const_vals to short.")
    

    A_ref = None
    if ref_vals is not None:
        if len(ref_vals) == num_vars and num_vars < num_vals:
            ref_vals_new = np.empty(num_vals)
            ref_vals_new[variables_bool_arr] = ref_vals
            ref_vals_new[~variables_bool_arr] = const_vals[ref_val_tissue_idx, ~variables_bool_arr]
            ref_vals = ref_vals_new
        c_ref, params_ref = np.split(ref_vals, [num_molecules])
        A_ref = func(wavelengths, mu_a, c_ref, *params_ref)

    x = np.empty((num_vars, num_spectra))
    errors = np.empty_like(A)

    if left_bounds.shape != right_bounds.shape:
        raise AttributeError("Arrays for left and right bounds should have the same shape.")
    if len(left_bounds) == num_vals and num_vars != num_vals:
        print("Note: Bounds for parameters " + ", ".join(map(str, np.arange(num_vals)[~variables_bool_arr])) + " will be ignored, as they are not optimized.")
        left_bounds = left_bounds[variables_bool_arr]
        right_bounds = right_bounds[variables_bool_arr]
    elif len(left_bounds) != num_vars:
        raise AttributeError("Invalid size for bounds.")

    if init_vals is None:
        if ref_vals is not None:
            init_vals = ref_vals[variables_bool_arr]
        else:
            raise AttributeError("Initialization Values are missing.")
    elif len(init_vals) == num_vals and num_vars != num_vals:
        print("Note: Initial values for parameters " + ", ".join(map(str, np.arange(num_vals)[~variables_bool_arr])) + " will be ignored, as they are not optimized.")
        init_vals = init_vals[variables_bool_arr]
    elif len(init_vals) != num_vars:
        raise AttributeError("Invalid size for initial values.")

    if constraint is None:
        for t in tqdm(range(num_spectra), disable=~progress_bar):
            res = least_squares(
                func_wrapper,
                x[:, t-1] if t > 0 and update_init else init_vals,
                bounds=(left_bounds, right_bounds),
                jac="2-point" if jacobian is None else jacobian_wrapper,
                args=(A[:, t], tissue_idxs[t], A_ref),
                x_scale="jac",
                verbose=verbosity
            )

            x[:, t] = res.x
            errors[:, t] = res.cost

            if progress_bar:
                print(t)
    else:
        bounds = Bounds(left_bounds.astype(float), right_bounds.astype(float), keep_feasible=True)
        constraint_matrix, lower_constraint_bounds, upper_constraint_bounds = constraint
        constant_params_one_hot_vec = (~variables_bool_arr).astype(float)
        lower_constraint_bound_vars = lower_constraint_bounds - constraint_matrix @ constant_params_one_hot_vec
        upper_constraint_bound_vars = upper_constraint_bounds - constraint_matrix @ constant_params_one_hot_vec
        constraint_vars = LinearConstraint(constraint_matrix[:, variables_bool_arr], lower_constraint_bound_vars, upper_constraint_bound_vars, keep_feasible=True)

        for t in tqdm(range(num_spectra), disable=~progress_bar):
            res = minimize(
                func_wrapper,
                x[:, t-1] if t > 0 else init_vals,
                method="trust-constr",
                constraints=[constraint_vars],
                bounds=bounds,
                args=(A[:, t], tissue_idxs[t], A_ref, True)
            )

            x[:, t] = res.x
            errors[:, t] = np.sqrt(res.fun / num_wavelengths)
        

    return x, errors

def concentr_fit_nonlinear_single_tissue(
    A,
    wavelengths,
    mu_a,
    func,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    const_vals=None,
    jacobian=None,
    constraint=None,
    ref_vals=None,
    init_vals=None,
    update_init=True,
    progress_bar=False,
    verbosity=0
):
    
    A = np.atleast_2d(A).reshape(A.shape[0], -1)
    tissue_idxs = np.zeros(A.shape[1])
    if const_vals is not None and len(np.array(const_vals).shape) == 1:
        const_vals = [np.array(const_vals)]
    elif const_vals is not None:
        raise AttributeError("Wrong dimension for const_vals. Should be 1D-array/list for this function.")

    return concentr_fit_nonlinear_multiple_tissues(
        A,
        tissue_idxs,
        wavelengths,
        mu_a,
        func,
        variables_bool_arr,
        left_bounds,
        right_bounds,
        ref_vals=ref_vals,
        const_vals=const_vals,
        jacobian=jacobian,
        constraint=constraint,
        update_init=update_init,
        init_vals=init_vals,
        progress_bar=progress_bar,
        verbosity=verbosity
    )


concentr_fit_nonlinear = concentr_fit_nonlinear_single_tissue


# some parameters are now indexed by tissue type in first dimension
# to allow optimizing parameters over multiple tissue types
# There is still only one reference pixel with one tissue type
def concentr_fit_nonlinear_reference_param_search_multiple_tissues(
    A,
    tissue_idxs,
    wavelengths,
    mu_a,
    func,
    param_space,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    const_vals=None, #multiple tissues
    jacobian=None,
    constraint=None,
    update_init=True,
    init_vals=None, 
    ref_val_tissue_idx=0,
    spectra_per_report=1,
    grace_spectra=1,
    num_samples=100,
    time_budget_s=3600,
    max_concurrent_trials=16
):

    num_vars = np.count_nonzero(variables_bool_arr)
    num_wavelengths, num_spectra = A.shape
    num_vals = len(variables_bool_arr)
    len_param_space = len(param_space)
    variables_bool_arr = np.array(variables_bool_arr)

    tissue_types, tissue_idxs_inverse = np.unique(tissue_idxs, return_inverse=True)
    # convert tissue_idxs to ascending ints starting from zero
    num_tissues = len(tissue_types)
    tissue_idxs = np.arange(num_tissues)[tissue_idxs_inverse].astype(int)
    ref_val_tissue_idx = list(tissue_types).index(ref_val_tissue_idx)
    
    assert len_param_space >= num_vars and len_param_space <= num_vals

    if const_vals is None:
        if num_vars != num_vals:
            raise AttributeError("Constant values for non-optimized variables are missing.")
        else:
            const_vals = np.empty((num_tissues, num_vals))
    elif num_tissues > len(const_vals):
        raise AttributeError("Too few const_vals.")

    #remove potentially unused rows, and convert to np.array
    const_vals = np.array(const_vals[:num_tissues])

    if const_vals.shape[1] == (num_vals - num_vars):
        const_vals_tmp = np.ones((num_tissues, num_vals)) * np.nan
        const_vals_tmp[:, ~variables_bool_arr] = const_vals
        const_vals = const_vals_tmp
    elif const_vals.shape[1] != num_vals:
        raise AttributeError("Dimension 1 of const_vals to short.")

    def trainable(config, A=None, func=func, jacobian=jacobian):
        ref_vals = np.array([config[str(i)] for i in range(len_param_space)])

        x = np.empty((num_vars, num_spectra))
        errors = np.empty_like(A)

        for t in range(0, num_spectra, spectra_per_report):
            cur_tissue_idxs = np.unique(tissue_idxs[t:t+spectra_per_report])
            cur_const_vals = np.array(const_vals)[cur_tissue_idxs]
            x[:, t:t+spectra_per_report], errors[:, t:t+spectra_per_report] = concentr_fit_nonlinear_multiple_tissues(
                A[:, t:t+spectra_per_report],
                tissue_idxs[t:t+spectra_per_report],
                wavelengths=wavelengths,
                mu_a=mu_a,
                func=func,
                ref_vals=ref_vals,
                const_vals=cur_const_vals,
                variables_bool_arr=variables_bool_arr,
                left_bounds=left_bounds,
                right_bounds=right_bounds,
                jacobian=jacobian,
                constraint=constraint,
                init_vals=x[:, t-1] if t > 0 else init_vals,
                update_init=update_init,
                verbosity=0
            )

            tune.report({"sq_avg_error": np.sum(errors[:, t:t+spectra_per_report]**2) / num_wavelengths / (min(t + spectra_per_report, num_spectra) - t)})
    

    grace_reports = math.ceil(grace_spectra / spectra_per_report)
    param_space_dict = {str(i) : param_space[i] for i in range(len_param_space)}
    scheduler = tune.schedulers.AsyncHyperBandScheduler(
        # metric and mode are passed down from TuneConfig
        grace_period=grace_reports,
        max_t=num_spectra
    )
    tuner = tune.Tuner(
        tune.with_parameters(trainable, A=A, func=func, jacobian=jacobian),
        param_space=param_space_dict,
        tune_config=tune.TuneConfig(
            metric="sq_avg_error",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            max_concurrent_trials=min(max_concurrent_trials, multiprocessing.cpu_count())
        )
    )

    ray.init(
        _system_config={
            # Allow spilling until the local disk is 98% utilized.
            # This only affects spilling to the local file system.
            "local_fs_capacity_threshold": 0.98,
        },
    )

    res = tuner.fit()
    return res


def concentr_fit_nonlinear_reference_param_search(
    A,
    wavelengths,
    mu_a,
    func,
    param_space,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    const_vals=None,
    jacobian=None,
    constraint=None,
    update_init=True,
    init_vals=None,
    spectra_per_report=1,
    grace_spectra=1,
    num_samples=100,
    time_budget_s=3600,
    max_concurrent_trials=16
):
    
    A = np.atleast_2d(A).reshape(A.shape[0], -1)
    tissue_idxs = np.zeros(A.shape[1])
    if const_vals is not None and len(np.array(const_vals).shape) == 1:
        const_vals = [np.array(const_vals)]
    
    return concentr_fit_nonlinear_reference_param_search_multiple_tissues(
        A,
        tissue_idxs,
        wavelengths,
        mu_a,
        func,
        param_space,
        variables_bool_arr,
        left_bounds,
        right_bounds,
        const_vals=const_vals,
        jacobian=jacobian,
        constraint=constraint,
        update_init=update_init,
        init_vals=init_vals,
        spectra_per_report=spectra_per_report,
        grace_spectra=grace_spectra,
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        max_concurrent_trials=max_concurrent_trials
    )
    

concentr_fit_nonlinear_hyperparam_search = concentr_fit_nonlinear_reference_param_search
