"""Constants for the prediction script."""

# Target parameters.
TARGETS: list[str] = ["SMR", "SMR00", "SMR06", "SMR12", "SMR18"]

# Features for plotting.
FEATURES: dict = {
    "BGSM Components": ["BGSM_x", "BGSM_y", "BGSM_z"],
    "Solar Wind Parameters": ["Magnitude", "Np", "T"],
    "Velocity Components": ["V_GSM_x", "V_GSM_y", "V_GSM_z"]
}

# Features for saving preprocessed data.
DATA_FEATURES: list[str] = [
    "Magnitude", "BGSM_x", "BGSM_y", "BGSM_z", "Np", "Tpr", "V_GSM_x", 
    "V_GSM_y", "V_GSM_z", "SC_pos_GSM_Re_x", "SC_pos_GSM_Re_y", 
    "SC_pos_GSM_Re_z", "T", "SMR_lag1", "SMR_lag2", "SMR_lag3", "SMR_lag5", 
    "SMR00_lag1", "SMR00_lag2", "SMR00_lag3", "SMR00_lag5", "SMR06_lag1", 
    "SMR06_lag2", "SMR06_lag3", "SMR06_lag5", "SMR12_lag1", "SMR12_lag2", 
    "SMR12_lag3", "SMR12_lag5", "SMR18_lag1", "SMR18_lag2", "SMR18_lag3", 
    "SMR18_lag5",
]

# Prediction outlier threshold (nT).
THRESHOLD: int = 1.5

# Date interval for CDAWeb data.
START: str = "2020-01-01 00:00:00"
END: str = "2020-12-31 23:59:59"

# Project folder.
FOLDER: str = "your_folder"

# SuperMAG data path and filename.
PATH: str = f"{FOLDER}/mag_data/"
FILE: str = "file.csv"

# XGBoost model path.
MODEL_PATH: str = f"{FOLDER}/saved_models/xgboost_model.json"

# Preprocessed data path.
DATA_PATH: str = f"{FOLDER}/processed_data/data.parquet"

# ArgParser description.
DESCRIPTION: str = "Load or save data and load or train a model."

# XGBoost parameters.
N_ESTIMATORS: int = 200
EARLY_STOPPING_ROUNDS: int = 100
XGB_PARAMS: dict = {
    "device": "cuda",
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "rmse",
    "lambda": 10.0,
    "alpha": 0.1,
    "multi_strategy": "one_output_per_tree" #"multi_output_tree",
}

# SuperMAG data fill value.
FILL: int = 999999

# CDAWeb datasets (keys) parameters (values).
CDAWEB_PARAMS: dict[str, list[str]] = {
    "ace_mfi": ["Magnitude", "BGSM"],                # AC_H0_MFI (position from SWE)
    "ace_swe": ["Np", "Tpr", "V_GSM", "SC_pos_GSM"], # AC_H0_SWE + position for MFI
    "wind_mfi": ["Magnitude", "BGSM", "SC_pos_GSM"], # WI_H0_MFI
    "wind_swe": ["V_GSM", "Np", "Tpr"],              # WI_K0_SWE (Tpr for thermal speed)
}

# Earth radius (km).
RE: float = 6378.0
# Boltzmann constant (J/K).
K: float = 1.380649e-23
# Proton mass (kg).
MP: float = 1.6726219e-27
