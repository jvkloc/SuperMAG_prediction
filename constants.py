"""Constants for the prediction script."""

# Target parameters.
TARGETS: list[str] = ["SMR00", "SMR06", "SMR12", "SMR18"]
# SuperMAG data fill value.
FILL: int = 999999
# Prediction outlier threshold (nT).
THRESHOLD: int = 1.5

# Earth radius (km).
RE: float = 6378.0
# Boltzmann constant (J/K).
K: float = 1.380649e-23
# Proton mass (kg).
MP: float = 1.6726219e-27

# CDAWeb datasets (keys) parameters (values).
CDAWEB_PARAMS: dict[str, list[str]] = {
    "ace_mfi": ["Magnitude", "BGSM"],                # AC_H0_MFI (position from SWE)
    "ace_swe": ["Np", "Tpr", "V_GSM", "SC_pos_GSM"], # AC_H0_SWE + position for MFI
    "wind_mfi": ["Magnitude", "BGSM", "SC_pos_GSM"], # WI_H0_MFI
    "wind_swe": ["V_GSM", "Np", "Tpr"],              # WI_K0_SWE (Tpr for thermal speed)
}

# Date interval for CDAWeb data.
START: str = "2020-01-01 00:00:00"
END: str = "2020-12-31 23:59:59"

# Project folder.
FOLDER: str = "your_folder"

# SuperMAG data path and filename.
PATH: str = FOLDER + "/mag_data/"
FILE: str = "2020-supermag.csv"

# XGBoost parameters.
N_ESTIMATORS: int = 200
EARLY_STOPPING_ROUNDS: int = 100
XGB_PARAMS: dict = {
    "device": "cuda",
    "objective": "reg:squarederror",
    "max_depth": 12,
    "learning_rate": 0.1,
    "eval_metric": "rmse",
    "lambda": 5.0,
    "alpha": 0.1,
    "multi_strategy": "one_output_per_tree" #"multi_output_tree",
}

# XGBoost model path.
MODEL: str = FOLDER + "/saved_models/xgboost_model.json"
