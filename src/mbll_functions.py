import numpy as np
import config
from sympy import lambdify, symbols
import pickle

def delta_A_no_scattering(delta_c, mu_a_matrix):
    return mu_a_matrix @ delta_c

def delta_A_scattering_1(delta_A_no_scattering, wavelengths, a_ref, b_ref, a_t, b_t):
    ###
    # delta_A_no_scattering: (num_wavelengths, num_spectra) or (num_wavelengths)
    # wavelengths: (num_wavelengths)
    # a_ref: scalar
    # b_ref: scalar
    # a_t: (num_spectra)
    # b_t: (num_spectra)
    ###
    one_dim_output = False
    if len(delta_A_no_scattering.shape) == 1:
        one_dim_output = True
        delta_A_no_scattering = delta_A_no_scattering[..., None]
    res = delta_A_no_scattering + (a_t[None, ...] * (wavelengths/500)[..., None]**(-b_t[None, ...]) - (a_ref * (wavelengths/500)**(-b_ref))[..., None])
    
    if one_dim_output:
        return res[:, 0]
    else:
        return res

def delta_A_scattering_2(delta_A_no_scattering, wavelengths, b_ref, b_t):
    num_wavelengths, num_spectra = delta_A_no_scattering.shape
    a_t = np.array([1.0] * num_spectra)
    return delta_A_scattering_1(delta_A_no_scattering, wavelengths, 1.0, b_ref, a_t, b_t)

def delta_A_scattering_3(delta_A_no_scattering, wavelengths, delta_a, b):
    num_wavelengths, num_spectra = delta_A_no_scattering.shape
    b = np.array([b] * num_spectra)
    return delta_A_scattering_1(delta_A_no_scattering, wavelengths, 0, 1.0, delta_a, b)

def delta_A_scattering_4(delta_A_no_scattering, wavelengths, a_t, b_t):
    return delta_A_scattering_1(delta_A_no_scattering, wavelengths, 0.0, 1.0, a_t, b_t)


def reduced_scattering(wavelengths, a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return a * (wavelengths/500)[:, None] ** (-b)

def mbll_new(wavelengths, mu_a_matrix, c, a, b, baseline_attenuation, baseline_c, baseline_a, baseline_b, pathlength, scatterlength, g):
    num_wavelengths, num_molecules = mu_a_matrix.shape
    pathlength = np.atleast_2d(pathlength).reshape(num_wavelengths, 1)
    scatterlength = np.atleast_2d(scatterlength).reshape(num_wavelengths, 1)
    c = np.atleast_2d(c).reshape(num_molecules, -1)
    baseline_c = np.atleast_2d(baseline_c).reshape(num_molecules, 1)
    baseline_attenuation = np.atleast_2d(baseline_attenuation).reshape(num_wavelengths, 1)
    mu_s_red = reduced_scattering(wavelengths, a, b)
    baseline_mu_s_red = reduced_scattering(wavelengths, baseline_a, baseline_b)
    # scatterlength is dA/dmu_s, but we need dA/dmu_s_red => multiply scatterlength with dmu_s/dmu_s_red = 1/(1-g)
    return baseline_attenuation + (mu_a_matrix @ (c-baseline_c)) * pathlength + scatterlength * (mu_s_red - baseline_mu_s_red) / (1-g)

###
# the variable A becomes ambiguous in the jacques model, because jacques
# defines its own A as A = m1 + m2*exp(ln(mu_a/mu_s)/m3)
# for clarification: this function returns -ln(R) = -ln(I(t)/I_0)
###
def A_jacques(mu_a_matrix, c, wavelengths, a, b, m1, m2, m3):
    mu_a = (mu_a_matrix @ c)
    mu_s_red = (a[None, ...] * (wavelengths/500)[..., None]**(-b[None, ...]))
    A_j = m1 + m2 * np.exp(np.log(mu_s_red/mu_a)/m3)
    theta = 1 / np.sqrt(3*mu_a*(mu_a + mu_s_red))
    return A_j * mu_a * theta


def A_jacques_concentrations(wavelengths, mu_a_matrix, c, a, b, m1, m2, m3):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c).reshape(mu_a_matrix.shape[1], -1)

    mu_a = mu_a_matrix @ c
    mu_s_red = a * (wavelengths/500)[:, None] ** (-b)
    A_j = m1 + m2 + np.exp(np.log(mu_s_red/mu_a)/m3)
    theta = 1 / np.sqrt(3*mu_a*(mu_a + mu_s_red))
    return A_j * mu_a * theta

# this function assumes, that the concentration array contains
# f_blood, stO2, c_oxCCO, c_redCCO, f_water, f_fat
def A_jacques_blood_fraction(wavelengths, mu_a_matrix, c, a, b, m1, m2, m3):
    c_new = blood_fraction_to_concentrations(c)
    return A_jacques_concentrations(wavelengths, mu_a_matrix, c_new, a, b, m1, m2, m3)

def A_carp(mu_a, mu_s, g, n):
    f = g*g
    g_star = g/(1+g)
    mu_s_star = mu_s*(1-f)
    mu_t_star = mu_a + mu_s_star
    mu_tr = (mu_a + mu_s * (1-g))
    mu_eff = np.sqrt(3 * mu_a * mu_tr)
    h = 2/(3*mu_tr)
    A = -0.13755 * np.power(n, 3) + 4.3390 * np.power(n, 2) - 4.90466 * n + 1.6896
    alpha = 3*mu_s_star*(mu_t_star + g_star * mu_a) / (mu_eff**2 - mu_t_star**2)
    beta = (-alpha * (1 + A*h*mu_t_star) - 3*A*h*g_star*mu_s_star) / (1 + A*h*mu_eff)
    R = (alpha + beta) / (2 * A)
    return -np.log(R)

def A_carp_concentrations(wavelengths, mu_a_matrix, c, a, b, g, n):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c).reshape(mu_a_matrix.shape[1], -1)
    mu_a = mu_a_matrix @ c
    mu_s_red = reduced_scattering(wavelengths, a, b)
    return A_carp(mu_a, mu_s_red / (1-g), g, n)


def A_carp_blood_fraction(wavelengths, mu_a_matrix, c, a, b, g, n):
    c = blood_fraction_to_concentrations(c)
    return A_carp_concentrations(wavelengths, mu_a_matrix, c, a, b, g, n)


# more rudimentary diffusion equation for planar source
def A_patterson(mu_a, mu_s_red, n):
    rd = -1.44 * np.power(n, -2) * 0.71 / n + 0.668 + 0.0636 * n
    k = (1 + rd) / (1 - rd)
    albedo = mu_s_red / (mu_a + mu_s_red)
    R = albedo / (1 + 2 * k * (1-albedo) + (1 + 2*k/3)*np.sqrt(3*(1-albedo)))
    return -np.log(R)

def A_patterson_concentrations(wavelengths, mu_a_matrix, c, a, b, n):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c).reshape(mu_a_matrix.shape[1], -1)
    mu_a = mu_a_matrix @ c
    mu_s_red = reduced_scattering(wavelengths, a, b)
    mu_a = mu_a_matrix @ c
    mu_s_red = reduced_scattering(wavelengths, a, b)
    return A_patterson(mu_a, mu_s_red, n)


def A_patterson_blood_fraction(wavelengths, mu_a_matrix, a, b, n):
    c = blood_fraction_to_concentrations(c)
    return A_patterson_concentrations(wavelengths, mu_a_matrix, c, a, b, n)

try:
    mu_a, mu_s_red, g = symbols("mu_a mu_s_red g", positive=True)
    k = symbols("k")
    with open(config.diffusion_derivative_dir / "carp.pickle", "rb") as f:
        A_carp_diff_mu_a_symbolic, A_carp_diff_mu_s_symbolic = pickle.load(f)
        A_carp_diff_mu_a_lambdified = lambdify((mu_a, mu_s_red, g, k), A_carp_diff_mu_a_symbolic)
        A_carp_diff_mu_s_red_lambdified = lambdify((mu_a, mu_s_red, g, k), A_carp_diff_mu_s_symbolic)
        
        def A_carp_pathlength(wavelengths, mu_a_matrix, c, a, b, g, n):
            mu_a = mu_a_matrix @ c
            mu_s_red = reduced_scattering(wavelengths, a, b)
            k = -0.13755 * (n**3) + 4.3390 * (n**2) - 4.90466 * n + 1.6896
            return A_carp_diff_mu_a_lambdified(mu_a[..., None], mu_s_red, g, k)

        def A_carp_scatterlength(wavelengths, mu_a_matrix, c, a, b, g, n):
            mu_a = mu_a_matrix @ c
            mu_s_red = reduced_scattering(wavelengths, a, b)
            k = -0.13755 * (n**3) + 4.3390 * (n**2) - 4.90466 * n + 1.6896
            return A_carp_diff_mu_s_red_lambdified(mu_a[..., None], mu_s_red, g, k) * (1-g)

except FileNotFoundError:
    print("Could load function data to define diffusion derivatives.")

# the first and second row of c should contain f_blood and st02, respectively
# keeps shape of input
def blood_fraction_to_concentrations(c):
    f_blood, stO2 = c[:2, ...]
    c_hb02 = config.c_pure_HbT * f_blood * stO2
    c_hbb = config.c_pure_HbT * f_blood * (1.0 - stO2)
    c_new = np.concatenate((c_hb02[None, ...], c_hbb[None, ...], c[2:]), axis=0)
    return c_new

def concentrations_to_blood_fraction(c):
    c_hbo2, c_hbb = c[:2, ...]
    f_blood = (c_hbo2 + c_hbb) / config.c_pure_HbT
    st02 = c_hbo2 / (c_hbo2 + c_hbb)
    c_new = np.concatenate((f_blood[None, ...], st02[None, ...], c[2:]), axis=0)
    
    return c_new

# convert f_blood and st02 percentages (in [0, 1]) to concentrations (hb02, hbb, background) in mM
# 150 g/L / 64500 g/mol = 150/64500 mol/L = 150/64500 * 1e3 mM
def blood_back_model_concentrations(f_blood, stO2):
    c_hb02 = (150.0 / 64500.0) * 1e3 * f_blood * stO2
    c_hbb = (150.0 / 64500.0) * 1e3 * f_blood * (1.0 - stO2)
    c_back = (1.0 - f_blood)
    c = np.array([c_hb02, c_hbb, c_back])
    return c


def blood_back_model_concentrations_inverse(c_hbo2, c_hbb, c_back):
    f_blood = 1.0 - c_back
    stO2 = c_hbo2 / ((150.0 / 64500.0) * 1e3 * f_blood)

    return f_blood, stO2


