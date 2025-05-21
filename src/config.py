from pathlib import Path

# main path variables
spectra_dir = Path(__file__).parent.parent / "dataset/UCL-NIR-Spectra/spectra/"
caredda_spectra = spectra_dir / "caredda-spectra"
data_dir = Path("/media/leon/T7/IDP/")
#data_dir = Path("/home/leon_ivan/data/")


# derived path variables
pl_dir = spectra_dir.parent.parent / "mc_pl_simulations"
m_params_path = spectra_dir.parent.parent / "m_parameters.pickle"

dataset_dir = data_dir / "dataset/HELICoiD/HSI_Human_Brain_Database_IEEE_Access"
simulated_dataset_dir = data_dir / "dataset/simulated"
result_dir = data_dir / "results"
mcs_func_path = data_dir / "mcs_function_data/function_data.npz"
diffusion_derivative_dir = spectra_dir.parent.parent / "diffusion_derivatives"

eval_dir = data_dir / "evaluation"
reference_params_path = eval_dir / "reference_params.pickle"



# hemoglobin molecular concentration
mol_weight_HbT = 64500 # [g/mol]
density_HbT = 150 # [g/L]
c_pure_HbT = (density_HbT / mol_weight_HbT) * 1e3 # [mM = mmol/L]


gpuid = 1