from pathlib import Path

# ain path variables
spectra_dir = Path(__file__).parent.parent / "dataset/UCL-NIR-Spectra/spectra/"
data_dir = Path("/home/leon_ivan/data/")

# derived path variables
pl_dir = spectra_dir.parent.parent / "mc_pl_simulations"
m_params_path = spectra_dir.parent.parent / "m_parameters.pickle"

dataset_dir = data_dir / "dataset/HELICoiD/HSI_Human_Brain_Database_IEEE_Access"
result_dir = data_dir / "results"
mcs_func_path = data_dir / "mcs_function_data/function_data.npz"



# hemoglobin molecular concentration
mol_weight_HbT = 64500 # [g/mol]
density_HbT = 150 # [g/L]
c_pure_HbT = (density_HbT / mol_weight_HbT) * 1e3 # [mM = mmol/L]


gpuid = 2