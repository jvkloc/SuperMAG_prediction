"""Constants for the prediction script."""

# Target parameters.
TARGETS: list[str] = ["SMR", "SMR00", "SMR06", "SMR12", "SMR18"]

# Features for plotting.
FEATURES: dict = {
    'BGSM Components': ["BGSM_x", "BGSM_y", "BGSM_z"],
    'Solar Wind Parameters': ["Magnitude", "Np", "T"],
    'Velocity Components': ["V_GSM_x", "V_GSM_y", "V_GSM_z"]
}

# Prediction outlier threshold (nT).
THRESHOLD: int = 1.5

# Date interval for CDAWeb data.
START: str = "1999-01-01 00:00:00"
END: str = "2022-01-31 23:59:59"

# Dates for cross-validation folds.
VALIDATION_WEEKS: list[str] = [
    "2019-04-01", "2019-07-01", "2019-10-01", "2020-01-01", "2020-04-01", 
    "2020-07-01", "2020-10-01", "2021-01-01", "2021-04-01", "2022-01-01",
]

# Project folder.
FOLDER: str = "/home/jvkloc/Downloads/Computational Statistics/lecture_exercises/cs/gradu"

# SuperMAG data paths and filename.
SMAG_PATH: str = f"{FOLDER}/mag_data/downloads/"
PATH: str = f"{FOLDER}/mag_data/"
FILE: str = "supermag.csv"

# XGBoost model path.
MODEL_PATH: str = f"{FOLDER}/saved_models/xgboost_model.json"

# Preprocessed data paths.
DATA_PATH: str = f"{FOLDER}/processed_data/training_data.parquet"

# ArgParser description.
DESCRIPTION: str = "Download/check data and/or train a model or predict with a loaded model."

# XGBoost parameters.
N_ESTIMATORS: int = 750 # 1000
EARLY_STOPPING_ROUNDS: int = 150 # 450
XGB_PARAMS: dict = {
    "device": "cuda",
    "objective": "reg:squarederror", #"reg:absoluteerror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "rmse", #"mae",  
    "lambda": 10.0,
    "alpha": 0.1,
    "multi_strategy": "one_output_per_tree",
}
SINGLE_TARGET_PARAMS: dict = {
    "device": "cuda",
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "rmse",
    "lambda": 10.0,
    "alpha": 0.1,
}
XGB_METRIC: str = XGB_PARAMS['eval_metric']

# SuperMAG data fill value.
FILL: int = 999999

#CDAWeb data fill value.
CDAWEB_FILL: float = -1e31

# CDAWeb datasets (keys) and parameters (values).
CDAWEB_PARAMS: dict[str, list[str]] = {
    'ace_mfi': ["Magnitude", "BGSM"],                # AC_H0_MFI (position from SWE)
    'ace_swe': ["Np", "Tpr", "V_GSM", "SC_pos_GSM"], # AC_H0_SWE + position for MFI
    'wind_mfi': ["Magnitude", "BGSM", "SC_pos_GSM"], # WI_H0_MFI
    'wind_swe': ["V_GSM", "Np", "Tpr"],              # WI_K0_SWE (Tpr for thermal speed)
}

# CDAWeb .cdf folders path.
CDAWEB_PATH: str = "gradu/spedas_data"

# Earth radius (km).
RE: float = 6378.0
# Boltzmann constant (J/K).
K: float = 1.380649e-23
# Proton mass (kg).
MP: float = 1.6726219e-27
