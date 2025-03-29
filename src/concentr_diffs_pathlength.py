from mbll_functions import *
from tqdm.notebook import tqdm, tqdm_notebook
from scipy.optimize import least_squares, minimize, Bounds, LinearConstraint, SR1
from ray import tune, train
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



### input
# mu_a
# wavelengths
# boolean array indicating if parameter is variable
# array of initial values (ref values if A is delta A, else initial values for NLLS, also const values for non-variables)
# function that computes A (takes mu_a and wavelengths as input)
# bounds



#def fit_concentration(
#    A,
#    mu_a_matrix,
#    wavelengths,
#    reference_values,
#    variable_idxs,
#    func,
#    upper_bounds,
#    lower_bounds,
#    is_delta_A=false
#):
#    
#    def func_wrapper(x, func, A, mu_matrix, wavelengths, params):
#        A = func(mu_matrix, wavelengths)

# functions take mu_a, wavelengths, c 


def concentr_fit_nonlinear_concurrent(
    A,
    wavelengths,
    mu_a,
    func,
    ref_vals,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    jacobian=None,
    constraint=None,
    is_delta_A=False,
    init_vals=None,
    update_init=True,
    progress_bar=False,
    num_processes=6
):
    concentr_fit_nonlinear_curried = partial(
        concentr_fit_nonlinear,
        wavelengths=wavelengths,
        mu_a=mu_a,
        func=func,
        ref_vals=ref_vals,
        variables_bool_arr=variables_bool_arr,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
        jacobian=jacobian,
        constraint=constraint,
        is_delta_A=is_delta_A,
        init_vals=init_vals,
        update_init=update_init,
        progress_bar=progress_bar
    )

    num_vars = np.count_nonzero(variables_bool_arr)
    _, num_spectra = A.shape
    x = np.empty((num_vars, num_spectra))
    errors = np.empty_like(A)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        start_idx = 0
        for i, (cur_x, cur_errors) in enumerate(executor.map(concentr_fit_nonlinear_curried, np.array_split(A, min(num_processes, multiprocessing.cpu_count()), axis=1))):
            cur_len = cur_x.shape[1]
            end_idx = start_idx+cur_len
            x[:, start_idx:end_idx] = cur_x
            errors[:, start_idx:end_idx] = cur_errors
            start_idx = end_idx
            print(f"Process {i}/{num_processes} finished.")
    
    return x, errors


def concentr_fit_nonlinear(
    A,
    wavelengths,
    mu_a,
    func,
    ref_vals,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    jacobian=None,
    constraint=None,
    is_delta_A=False,
    init_vals=None,
    update_init=True,
    progress_bar=False
):

    num_wavelengths, num_spectra = A.shape
    num_molecules = mu_a.shape[1]
    num_vals = len(variables_bool_arr)
    num_params = num_vals - num_molecules
    num_vars = np.count_nonzero(variables_bool_arr)
    num_molecule_vars = np.count_nonzero(variables_bool_arr[:num_molecules])

    def func_wrapper(x, A_t, A_ref=None, sum_and_square=False):
        c_full = np.empty((num_molecules,))
        c_full[variables_bool_arr[:num_molecules]] = x[:num_molecule_vars]
        c_full[~variables_bool_arr[:num_molecules]] = ref_vals[~variables_bool_arr][:(num_molecules - num_molecule_vars)]
        params = np.empty((num_params,))
        params[variables_bool_arr[num_molecules:]] = x[num_molecule_vars:]
        params[~variables_bool_arr[num_molecules:]] = ref_vals[~variables_bool_arr][-num_params:]
        A_func = func(wavelengths, mu_a, c_full, *params)
        res = A_t - A_func[:, 0]
        if A_ref is not None:
            res += A_ref[:, 0]
        if sum_and_square:
            res = np.sum(res**2)
        return res  

    # only works if all variables (molecules and params) are optimized!
    def jacobian_wrapper(x, *args):
        c = x[:num_molecules]
        params = x[num_molecules:]
        jacobian_val = jacobian(wavelengths, mu_a, c, *params)  
        return jacobian_val[:, 0, :]  
    
    A_ref = None
    if is_delta_A:
        c_ref, params_ref = np.split(ref_vals, [num_molecules])
        A_ref = func(wavelengths, mu_a, c_ref, *params_ref)

    x = np.empty((num_vars, num_spectra))
    errors = np.empty_like(A)

    if left_bounds.shape != right_bounds.shape:
        raise RuntimeError("Arrays for left and right bounds should have the same shape.")
    if len(left_bounds) == num_vals and num_vars != num_vals:
        print("Note: Bounds for parameters " + ", ".join(map(str, np.arange(num_vals)[~variables_bool_arr])) + " will be ignored, as they are not optimized.")
        left_bounds = left_bounds[variables_bool_arr]
        right_bounds = right_bounds[variables_bool_arr]
    elif len(left_bounds) != num_vars:
        raise RuntimeError("Invalid size for bounds.")

    if init_vals is None:
        init_vals = ref_vals[variables_bool_arr]
    elif len(init_vals) == num_vals and num_vars != num_vals:
        print("Note: Initial values for parameters " + ", ".join(map(str, np.arange(num_vals)[~variables_bool_arr])) + " will be ignored, as they are not optimized.")
        init_vals = init_vals[variables_bool_arr]
    elif len(init_vals) != num_vars:
        raise RuntimeError("Invalid size for initial values.")

    if constraint is None:
        for t in tqdm(range(num_spectra), disable=~progress_bar):
            res = least_squares(
                func_wrapper,
                x[:, t-1] if t > 0 and update_init else init_vals,
                bounds=(left_bounds, right_bounds),
                jac="2-point" if jacobian is None else jacobian_wrapper,
                args=(A[:, t], A_ref)
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
                args=(A[:, t], A_ref, True)
            )

            x[:, t] = res.x
            errors[:, t] = np.sqrt(res.fun / num_wavelengths)
        

    return x, errors


def concentr_fit_nonlinear_hyperparam_search(
    A,
    wavelengths,
    mu_a,
    func,
    param_space,
    variables_bool_arr,
    left_bounds,
    right_bounds,
    jacobian=None,
    constraint=None,
    is_delta_A=False,
    update_init=True,
    spectra_per_report=1,
    grace_spectra=1,
    num_samples=100,
    time_budget_s=3600,
    max_concurrent_trials=16
):

    num_vars = np.count_nonzero(variables_bool_arr)
    num_wavelengths, num_spectra = A.shape
    num_vals = len(variables_bool_arr)
    

    def trainable(config):
        
        ref_vals = np.array([config[str(i)] for i in range(num_vals)])
 
        x = np.empty((num_vars, num_spectra))
        errors = np.empty_like(A)

        for t in range(0, num_spectra, spectra_per_report):
            x[:, t:t+spectra_per_report], errors[:, t:t+spectra_per_report] = concentr_fit_nonlinear(
                A[:, t:t+spectra_per_report],
                wavelengths,
                mu_a,
                func,
                ref_vals,
                variables_bool_arr,
                left_bounds,
                right_bounds,
                jacobian,
                constraint,
                is_delta_A,
                init_vals=x[:, t-1] if t > 0 else None,
                update_init=update_init
            )

            tune.report({"sq_avg_error": np.sum(errors[:, t:t+spectra_per_report]**2) / num_wavelengths / (min(t + spectra_per_report, num_spectra) - t)})
    
    grace_reports = math.ceil(grace_spectra / spectra_per_report)
    param_space_dict = {str(i) : param_space[i] for i in range(num_vals)}
    scheduler = tune.schedulers.AsyncHyperBandScheduler(
        # metric and mode are passed down from TuneConfig
        grace_period=grace_reports,
        max_t=num_spectra
    )
    tuner = tune.Tuner(
        trainable,
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

    res = tuner.fit()
    return res



def concentr_jacques_from_attenuation_change(
    delta_A,
    mu_a_matrix,
    wavelengths,
    c_ref,
    c_totcco,
    f_ref_water,
    f_ref_fat,
    a_ref,
    b_ref,
    m1,
    m2,
    m3,
    max_a,
    max_b,
    init_x=None
):
    def f(x, delta_A, A_jacques_ref, mu_a_matrix, wavelengths, m1, m2, m3):
        c, a_t, b_t = np.split(x, [4, 5])
        A_jacques_t = A_jacques(mu_a_matrix, c[..., None], wavelengths, a_t, b_t, m1, m2, m3)
        return (A_jacques_t - A_jacques_ref)[:, 0] - delta_A
    
    def f_const_totcco(x, delta_A, A_jacques_ref, mu_a_matrix, wavelengths, m1, m2, m3, c_totcco):
        c, a_t, b_t = np.split(x, [3, 4])
        c_full = np.concatenate((c, np.array([c_totcco - c[2], f_water, f_fat])), axis=0)
        A_jacques_t = A_jacques(mu_a_matrix, c_full[..., None], wavelengths, a_t, b_t, m1, m2, m3)
        return (A_jacques_t - A_jacques_ref)[:, 0] - delta_A
    
    num_spectra = delta_A.shape[-1]
    num_wavelengths, num_molecules = mu_a_matrix.shape

    assert num_molecules == 6

    if c_totcco is not None:
        num_variables = 5
        assert c_ref[2] <= c_totcco
        c_ref_full = np.concatenate((c_ref, np.array([c_totcco - c_ref[2], f_water, f_fat])), axis=0)
        f_optim = f_const_totcco
        right_bound = np.array([np.inf, np.inf, c_totcco, max_a, max_b])
        mu_a_matrix = mu_a_matrix.copy()
        mu_a_matrix[:, 2] -= mu_a_matrix[:, 3]
    else:
        f_optim = f
        num_variables = 6
        right_bound = np.array([np.inf, np.inf, np.inf, np.inf, max_a, max_b])
        c_ref_full = np.concatenate(c_ref, np.array([f_water, f_fat]))

    c = np.empty((num_variables - 2, num_spectra))
    a_t = np.empty((num_spectra,))
    b_t = np.empty((num_spectra,))
    A_jacques_ref = A_jacques(mu_a_matrix, c_ref_full[:, None], wavelengths, np.array([a_ref]), np.array([b_ref]), m1, m2, m3)
    left_bound = np.zeros(num_variables)
    
    a_t[0], b_t[0], c[:, 0] = a_ref, b_ref, c_ref
    cur_x = np.empty((num_variables))
    if init_x is not None:
        cur_x = init_x
    else:
        cur_x[:-2] = c_ref
        cur_x[-2] = a_ref
        cur_x[-1] = b_ref
    errors = np.zeros((num_wavelengths, num_spectra))

    for t in range(num_spectra):
        args = (delta_A[:, t], A_jacques_ref, mu_a_matrix, wavelengths, m1, m2, m3)
        if c_totcco is not None:
            args = args + (c_totcco, )
        
        res = least_squares(
            f_optim,
            cur_x,
            args=args,
            bounds=(left_bound, right_bound)
        )

        c[:, t], a_t[t], b_t[t] = np.split(res.x, [-2, -1])
        errors[:, t] = res.cost
        cur_x = res.x
    
    return c, a_t, b_t, errors


def fit_jacques_from_attenuation_change(delta_A, mu_a_matrix, wavelengths, max_a, max_b, param_space, num_samples=10, its_per_report=1):
    
    num_wavelengths, num_spectra = delta_A.shape
    _, num_molecules = mu_a_matrix.shape

    assert num_molecules == 6

    def trainable(config):
        c_ref = [config[c_str] for c_str in ["c_ref_HbO2", "c_ref_Hbb", "c_ref_oxCCO"]]
        if "c_totCCO" in config:
            c_totCCO = config["c_totCCO"]
            num_variables = 5
        else:
            c_totCCO = None
            c_ref += [config["c_ref_redCCO"]]
            num_variables = 6

        a_ref, b_ref = config["a_ref"], config["b_ref"]
        m1, m2, m3 = config["m1"], config["m2"], config["m3"]
        f_water = config["f_water"]
        f_fat = config["f_fat"]

        cur_x = np.empty((num_variables,))
        errors = np.zeros((num_wavelengths, num_spectra))

        # call calculate_concentr_diffs with ref spectrum and one spectrum
        # this weird stitching is done because we want to call train.report
        # after each iteration to incrementally report score
        # and potentially skip unpromising hyperparameters by scheduler
        ######  Update: I think this is completeley unnecessary
        prev_t = 0
        for t in range(its_per_report, num_spectra, its_per_report):
            #delta_A_tmp = np.empty((num_wavelengths, 2))
            #delta_A_tmp[:, 0] = delta_A[:, 0] # reference
            #delta_A_tmp[:, 1] = delta_A[:, t]

            c_res, a_res, b_res, errors_res = concentr_jacques_from_attenuation_change(
                delta_A[:, prev_t:t],
                mu_a_matrix,
                wavelengths,
                c_ref,
                c_totCCO,
                f_water,
                f_fat,
                a_ref,
                b_ref,
                m1,
                m2,
                m3,
                max_a,
                max_b,
                init_x=cur_x if t != 0 else None
            )
            
            #errors[:, t] = errors_res[:, 1]

            #cur_x[:num_molecules] = c_res[:, 1]
            #cur_x[num_molecules] = a_res[1]
            #cur_x[-1] = b_res[1]

            errors[:, prev_t:t] = errors_res
            cur_x[:-2] = c_res[:, -1]
            cur_x[-2] = a_res[-1]
            cur_x[-1] = b_res[-1]
            
            train.report({"score": np.sum(errors[:,prev_t:t]**2.0)/its_per_report})

            prev_t = t


    scheduler = tune.schedulers.AsyncHyperBandScheduler(
        #grace_period=300, # make sure to include time idxs, where hypoxia starts
        max_t=num_spectra,
    )

    tuner = tune.Tuner(
        trainable,
        param_space = param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="score",
            mode="min",
            scheduler=scheduler,
        )
    )
    res = tuner.fit()
    return res



def concentr_jacques_from_attenuation_change_blood_back_model(
    delta_A,
    mu_a_blood_back_model,
    wavelengths,
    f_blood_ref,
    stO2_ref,
    a_ref,
    b_ref,
    m1,
    m2,
    m3,
    max_a,
    max_b,
    init_x=None
):
    assert(mu_a_blood_back_model.shape[1] == 3) # we expect hb02, hbb and "background" absorption coefs
    def f(x, delta_A, A_jacques_ref, mu_a_blood_back_model, wavelengths, m1, m2, m3):
        f_blood, stO2, a_t, b_t = np.split(x, [1, 2, 3])
        c = blood_back_model_concentrations(f_blood, stO2)
        A_jacques_t = A_jacques(mu_a_blood_back_model, c, wavelengths, a_t, b_t, m1, m2, m3)
        return (A_jacques_t - A_jacques_ref)[:, 0] - delta_A
    
    num_spectra = delta_A.shape[-1]
    c_ref = blood_back_model_concentrations(f_blood_ref, stO2_ref)
    A_jacques_ref = A_jacques(mu_a_blood_back_model,c_ref[..., None], wavelengths, np.array([a_ref]), np.array([b_ref]), m1, m2, m3)
    num_wavelengths = mu_a_blood_back_model.shape[0]
    # f_blood and stO2 are in [0 - 1]
    left_bound = np.zeros(4)
    right_bound = np.append(np.ones(2), [max_a, max_b])
    f_blood_t = np.empty((num_spectra,))
    stO2_t = np.empty((num_spectra,))
    f_blood_t[0] = f_blood_ref
    stO2_t[0] = stO2_ref
    a_t = np.empty((num_spectra,))
    b_t = np.empty((num_spectra,))
    a_t[0], b_t[0] = a_ref, b_ref
    cur_x = np.zeros(4)
    if init_x is not None:
        cur_x = init_x
    else:
        cur_x[0] = f_blood_ref
        cur_x[1] = stO2_ref
        cur_x[2] = a_ref
        cur_x[-1] = b_ref
    errors = np.zeros((num_wavelengths, num_spectra))
    for t in range(1, num_spectra):
        res = least_squares(
            f,
            cur_x,
            args=(delta_A[:, t], A_jacques_ref, mu_a_blood_back_model, wavelengths, m1, m2, m3),
            bounds=(left_bound, right_bound)
        )
        f_blood_t[t], stO2_t[t], a_t[t], b_t[t] = np.split(res.x, [1, 2, 3])
        errors[:, t] = res.cost
        cur_x = res.x
    
    return f_blood_t, stO2_t, a_t, b_t, errors


def fit_jacques_from_attenuation_change_blood_back_model(delta_A, mu_a_blood_back_model, wavelengths, max_a, max_b, param_space, num_samples=10):
    
    num_wavelengths, num_spectra = delta_A.shape

    def trainable(config):
        
        f_blood_ref = config["f_blood_ref"]
        stO2_ref = config["stO2_ref"]
        a_ref, b_ref = config["a_ref"], config["b_ref"]
        m1, m2, m3 = config["m1"], config["m2"], config["m3"]

        cur_x = np.empty((4,))
        errors = np.zeros((num_wavelengths, num_spectra))

        for t in range(num_spectra):
            delta_A_tmp = np.empty((num_wavelengths, 2))
            delta_A_tmp[:, 0] = delta_A[:, 0] # reference
            delta_A_tmp[:, 1] = delta_A[:, t]

            f_blood_res, stO2_res, a_res, b_res, errors_res = concentr_jacques_from_attenuation_change_blood_back_model(
                delta_A_tmp,
                mu_a_blood_back_model,
                wavelengths,
                f_blood_ref,
                stO2_ref,
                a_ref,
                b_ref,
                m1,
                m2,
                m3,
                max_a,
                max_b,
                init_x=cur_x if t != 0 else None
            )
            
            errors[:, t] = errors_res[:, 1]

            cur_x[0] = f_blood_res[1]
            cur_x[1] = stO2_res[1]
            cur_x[2] = a_res[1]
            cur_x[-1] = b_res[1]
            
            train.report({"score": np.sum(errors[:,t]**2.0)})

    scheduler = tune.schedulers.AsyncHyperBandScheduler(
        grace_period=300, # make sure to include time idxs, where hypoxia starts
        max_t=num_spectra,
    )

    tuner = tune.Tuner(
        trainable,
        param_space = param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="score",
            mode="min",
            scheduler=scheduler,
        )
    )
    res = tuner.fit()
    return res



def concentr_diffs_mbll_blood_back_model(
    delta_A,
    mu_a_hb02,
    mu_a_hb,
    mu_a_back,
    wavelengths,
    f_blood_ref,
    stO2_ref,
    a_ref,
    b_ref,
    max_a,
    max_b,
    init_x=None
):
    def f(x, delta_A, c_ref, a_ref, b_ref, mu_a_matrix, wavelengths):
        f_blood, stO2, a_t, b_t = np.split(x, [1, 2, 3])
        c = blood_back_model_concentrations(f_blood, stO2)
        delta_c = c - c_ref[:, None]
        delta_A_no_scatter = delta_A_no_scattering(delta_c, mu_a_matrix)
        delta_A_scatter = delta_A_scattering_1(delta_A_no_scatter, wavelengths, a_ref, b_ref, a_t, b_t)
        return delta_A_scatter[:, 0] - delta_A
    
    num_spectra = delta_A.shape[-1]
    c_ref = blood_back_model_concentrations(f_blood_ref, stO2_ref)
    mu_a_matrix = np.column_stack((mu_a_hb02, mu_a_hb, mu_a_back))
    num_wavelengths = mu_a_matrix.shape[0]
    # f_blood and stO2 in [0 - 1]
    left_bound = np.zeros(4)
    right_bound = np.append(np.ones(2), [max_a, max_b])
    f_blood_t = np.empty((num_spectra,))
    stO2_t = np.empty((num_spectra,))
    f_blood_t[0] = f_blood_ref
    stO2_t[0] = stO2_ref
    a_t = np.empty((num_spectra,))
    b_t = np.empty((num_spectra,))
    a_t[0], b_t[0] = a_ref, b_ref
    cur_x = np.zeros(4)
    if init_x is not None:
        cur_x = init_x
    else:
        cur_x[0] = stO2_ref
        cur_x[1] = f_blood_ref
        cur_x[2] = a_ref
        cur_x[-1] = b_ref
    errors = np.zeros((num_wavelengths, num_spectra))
    for t in range(1, num_spectra):
        res = least_squares(
            f,
            cur_x,
            args=(delta_A[:, t], c_ref, a_ref, b_ref, mu_a_matrix, wavelengths),
            bounds=(left_bound, right_bound)
        )
        f_blood_t[t], stO2_t[t], a_t[t], b_t[t] = np.split(res.x, [1, 2, 3])
        errors[:, t] = res.cost
        cur_x = res.x
    
    return f_blood_t, stO2_t, a_t, b_t, errors


def fit_concentr_diffs_mbll_blood_back_model(delta_A, mu_a_hb02, mu_a_hb, mu_a_back, wavelengths, max_a, max_b, num_samples=10):
    
    num_wavelengths, num_spectra = delta_A.shape

    def trainable(config):
        
        f_blood_ref = config["f_blood_ref"]
        stO2_ref = config["stO2_ref"]
        a_ref, b_ref = config["a_ref"], config["b_ref"]

        cur_x = np.empty((4,))
        errors = np.zeros((num_wavelengths, num_spectra))

        for t in range(num_spectra):
            delta_A_tmp = np.empty((num_wavelengths, 2))
            delta_A_tmp[:, 0] = delta_A[:, 0] # reference
            delta_A_tmp[:, 1] = delta_A[:, t]

            f_blood_res, stO2_res, a_res, b_res, errors_res = concentr_diffs_mbll_blood_back_model(
                delta_A_tmp,
                mu_a_hb02,
                mu_a_hb,
                mu_a_back,
                wavelengths,
                f_blood_ref,
                stO2_ref,
                a_ref,
                b_ref,
                max_a,
                max_b,
                init_x=cur_x if t != 0 else None
            )
            
            errors[:, t] = errors_res[:, 1]

            cur_x[0] = f_blood_res[1]
            cur_x[1] = stO2_res[1]
            cur_x[2] = a_res[1]
            cur_x[-1] = b_res[1]
            
            train.report({"score": np.sum(errors[:,t]**2.0)})

        total_score = np.sum(errors**2)
        train.report({"total_score": total_score})


    #scheduler = tune.schedulers.AsyncHyperBandScheduler(grace_period=5, max_t=100)

    tuner = tune.Tuner(
        trainable,
        param_space = { # mu_a = [cm^-1mM^-1]
            "f_blood_ref": tune.uniform(0.8, 1.0), # percentage in(0, 1), accord. to Comparitive Study: 0.002 - 0.07
            "stO2_ref": tune.uniform(0.3,), #percentage, accord. to Comparitive Study: 0-1. 
                                        # We would expect 1/(150/6400 * 1e3) = 0.43 to get c_Hb02 = 1mM
            "a_ref": tune.uniform(0, max_a), #tune.uniform(0, max_a),
            "b_ref": tune.uniform(0, max_b),#tune.uniform(0, max_b),
        },
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="total_score",
            mode="min",
            #scheduler=scheduler,
        )
    )
    res = tuner.fit()
    return res



def concentr_jacques_from_attenuation_change_const_scatter(
    delta_A,
    mu_a_matrix,
    wavelengths,
    c_ref, # hbo2, hbb, oxcco, potentially redcco -> these will be fitted
    c_totcco,# either None or value
    f_water, # the following values will be assumed constant
    f_fat,
    m1,
    m2,
    m3,
    a,
    b,
    init_x=None
):
    def f(x, delta_A, A_jacques_ref, mu_a_matrix, wavelengths, m1, m2, m3, f_water, f_fat, a, b, c_totcco):
        c_full = np.concatenate((x, np.array([f_water, f_fat])), axis=0)
        A_jacques_t = A_jacques(mu_a_matrix, c_full[..., None], wavelengths, np.array([a]), np.array([b]), m1, m2, m3)
        return (A_jacques_t - A_jacques_ref)[:, 0] - delta_A
    
    def f_const_totcco(x, delta_A, A_jacques_ref, mu_a_matrix, wavelengths, m1, m2, m3, f_water, f_fat, a, b, c_totcco):
        c_full = np.concatenate((x, np.array([c_totcco - x[2], f_water, f_fat])), axis=0)
        A_jacques_t = A_jacques(mu_a_matrix, c_full[..., None], wavelengths, np.array([a]), np.array([b]), m1, m2, m3)
        return (A_jacques_t - A_jacques_ref)[:, 0] - delta_A
    
    num_spectra = delta_A.shape[-1]
    num_wavelengths, num_molecules = mu_a_matrix.shape

    assert num_molecules == 6

    if c_totcco is not None:
        num_variables = 3 # solve for c_Hb02, c_Hbb and c_oxCCO
        c = np.empty((3, num_spectra))
        assert c_ref[2] <= c_totcco # reference value for oxCCO needs to be smaller than total
        c_ref_full = np.concatenate((c_ref, np.array([c_totcco - c_ref[2], f_water, f_fat])), axis=0)
        f_optim = f_const_totcco
        right_bound = np.array([np.inf, np.inf, c_totcco])
        left_bound = np.zeros(3)
        mu_a_matrix = mu_a_matrix.copy() # the oxCCO will be multiplied with ox-red mu-a
        mu_a_matrix[:, 2] -= mu_a_matrix[:, 3]
    else:
        f_optim = f
        right_bound = np.ones(4) * np.inf
        c = np.empty((4, num_spectra))
        left_bound = np.zeros(4)
        num_variables = 4
        c_ref_full = np.concatenate((c_ref, np.array([f_water, f_fat])), axis=0)

 
    A_jacques_ref = A_jacques(mu_a_matrix, c_ref_full[:, None], wavelengths, np.array([a]), np.array([b]), m1, m2, m3)
    
    c[:, 0] = c_ref[:num_variables]
    cur_x = np.zeros(num_variables)
    if init_x is not None:
        cur_x = init_x
    else:
        cur_x = c_ref[:num_variables]
    errors = np.zeros((num_wavelengths, num_spectra))

    for t in range(num_spectra):
        args = (delta_A[:, t], A_jacques_ref, mu_a_matrix, wavelengths, m1, m2, m3, f_water, f_fat, a, b)
        if c_totcco is not None:
            args = args + (c_totcco, )
        
        res = least_squares(
            f_optim,
            cur_x,
            args=args,
            bounds=(left_bound, right_bound)
        )

        c[:, t] = res.x
        errors[:, t] = res.cost
        cur_x = res.x
    
    return c, errors


def fit_jacques_from_attenuation_change_const_scatter(delta_A, mu_a_matrix, wavelengths, f_fat, f_water, param_space, num_samples=10):
    
    num_wavelengths, num_spectra = delta_A.shape
    _, num_molecules = mu_a_matrix.shape

    def trainable(config):
        c_totcco = config["c_totcco"] if "c_totcco" in config else None
        if c_totcco is not None:
            num_molecules = 3
            assert config["c_ref_oxcco_perc"] < 1
            config["c_ref_oxcco"] = config["c_ref_oxcco_perc"] * c_totcco

        c_strs = ["c_ref_hb02", "c_ref_hbb", "c_ref_oxcco", "c_ref_redcco"]
        c_ref = [config[c_str] for c_str in c_strs[:num_molecules]]
        a_ref, b_ref = config["a_ref"], config["b_ref"]
        m1, m2, m3 = config["m1"], config["m2"], config["m3"]

        cur_x = np.empty((num_molecules + 2,))
        errors = np.zeros((num_wavelengths, num_spectra))

        # call calculate_concentr_diffs with ref spectrum and one spectrum
        # this weird stitching is done because we want to call train.report
        # after each iteration to incrementally report score
        # and potentially skip unpromising hyperparameters by scheduler
        ######  Update: I think this is completeley unnecessary
        for t in range(num_spectra):
            #delta_A_tmp = np.empty((num_wavelengths, 2))
            #delta_A_tmp[:, 0] = delta_A[:, 0] # reference
            #delta_A_tmp[:, 1] = delta_A[:, t]

            c_res, a_res, b_res, errors_res = concentr_jacques_from_attenuation_change(
                delta_A[:, np.newaxis, t],
                mu_a_matrix,
                wavelengths,
                c_ref,
                c_totcco,
                a_ref,
                b_ref,
                m1,
                m2,
                m3,
                max_a,
                max_b,
                init_x=cur_x if t != 0 else None
            )
            
            #errors[:, t] = errors_res[:, 1]

            #cur_x[:num_molecules] = c_res[:, 1]
            #cur_x[num_molecules] = a_res[1]
            #cur_x[-1] = b_res[1]

            errors[:, t] = errors_res[:, 0]
            cur_x[:num_molecules] = c_res[:, 0]
            cur_x[num_molecules] = a_res[0]
            cur_x[-1] = b_res[0]
            
            train.report({"score": np.sum(errors[:,t]**2.0)})


    scheduler = tune.schedulers.AsyncHyperBandScheduler(
        grace_period=300, # make sure to include time idxs, where hypoxia starts
        max_t=num_spectra,
    )

    tuner = tune.Tuner(
        trainable,
        param_space = param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="score",
            mode="min",
            scheduler=scheduler,
        )
    )
    res = tuner.fit()
    return res



