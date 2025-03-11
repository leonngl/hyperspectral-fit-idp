import numpy as np
import config

def mbll_new(wavelengths, mu_a_matrix, c, pl):
    pl = np.atleast_2d(pl).reshape(wavelengths.shape[0], -1)
    c = np.atleast_2d(c).reshape(mu_a_matrix.shape[1], -1)
    return (mu_a_matrix @ c) * pl

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


def A_jacques_new(wavelengths, mu_a_matrix, c, a, b, m1, m2, m3):
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
    return A_jacques_new(wavelengths, mu_a_matrix, c_new, a, b, m1, m2, m3)

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
    stO2 = c_hbO2 / ((150.0 / 64500.0) * 1e3 * f_blood)

    return f_blood, stO2


