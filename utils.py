"""Utility functions for the main script."""

from os import environ
from os.path import abspath, dirname, join
from time import perf_counter

from numpy import (
    all as npall,
    any as npany,
    nan,
    ndarray,
    unique,
)
from pandas import DataFrame, read_csv, Series, to_datetime
from pytplot import get_data
from xgboost import Booster

from constants import TARGETS


def add_lagged_features(combined: DataFrame, targets: list[str]) -> DataFrame:
    """Adds lagged SMR values as features to the data DataFrame."""
    for target in targets:
        combined[f'{target}_lag1'] = combined[target].shift(1)
        combined[f'{target}_lag2'] = combined[target].shift(2)
        combined[f'{target}_lag3'] = combined[target].shift(3)
        combined[f'{target}_lag5'] = combined[target].shift(5)
    # Remove rows with NaNs introduced by shifting
    pruned: DataFrame = combined.dropna()
    print(f"Combined DataFrame shape with lags: {pruned.shape}")
    return pruned


def get_cdaweb_data(parameters: dict, re: float, k: float, mp: float) -> dict:
    """Processes CDAWeb data and returns it in a dictionary. 
    Arguments
    --------- 
        re: Earth radius (km)
        k: Boltzmann constant (J/k)
        mp: proton mass (kg)
    """
    data: dict = {}
    for dataset, params in parameters.items():
        for param in params:
            values = get_data(param)
            # CDAWeb fill value is -1e31.
            if values is not None and not npall(values.y == -1e31):  
                if npany(values.y == -1e31):
                    print(f"Warning: {param} from {dataset} has some fill values (-1e31)")
                # Deduplicate timestamps, keeping the first occurrence.
                unique_times, unique_indices = unique(values.times, return_index=True)
                if "SC_pos_GSM" in param:
                    data_in_re: ndarray = values.y / re
                    data[f"{dataset}_{param}_Re_x"] = Series(data_in_re[:, 0], index=unique_times)
                    data[f"{dataset}_{param}_Re_y"] = Series(data_in_re[:, 1], index=unique_times)
                    data[f"{dataset}_{param}_Re_z"] = Series(data_in_re[:, 2], index=unique_times)
                elif param == "Tpr" and dataset == "wind_swe": # Convert thermal speed to Kelvin.
                    tpr: ndarray = values.y
                    kelvin: ndarray = (mp * tpr**2) / (2 * k)
                    data[f"{dataset}_T"] = Series(kelvin, index=values.times)
                elif len(values.y.shape) > 1:  # Vector data (e.g., BGSM, V_GSM).
                    data[f"{dataset}_{param}_x"] = Series(values.y[unique_indices, 0], index=unique_times)
                    data[f"{dataset}_{param}_y"] = Series(values.y[unique_indices, 1], index=unique_times)
                    data[f"{dataset}_{param}_z"] = Series(values.y[unique_indices, 2], index=unique_times)
                else:  # Scalar data (e.g., Magnitude, Np).
                    data[F"{dataset}_{param}"] = Series(values.y[unique_indices], index=unique_times)
    
    # Return the data dictionary.
    return data


def get_features(data: DataFrame, SuperMAG: DataFrame) -> list[str]:
    """Returns a list of model features."""
    return [col for col in data.columns if col not in SuperMAG.columns]


def get_rolling_basis_cv_splits(
    X: DataFrame, y: DataFrame, n_folds: int = 10
) -> list[tuple]:
    """Returns n_folds rolling basis cross-validation splits."""
    n_samples: int = len(X)
    fold_size: int = n_samples // n_folds
    cv_splits: list[tuple] = []
    
    print("10 train-test splits:")
    for i in range(1, n_folds):
        train_end: int = i * fold_size
        test_start: int = train_end
        test_end: int = (i + 1) * fold_size
        
        X_train: ndarray = X.iloc[:train_end].to_numpy()
        y_train: ndarray = y.iloc[:train_end].to_numpy()
        X_test: ndarray = X.iloc[test_start:test_end].to_numpy()
        y_test: ndarray = y.iloc[test_start:test_end].to_numpy()

        print(f"Fold {i}: Train {y.index[0]} to {y.index[train_end-1]} ({len(y_train)}), ", end='')
        print(f"Test {y.index[test_start]} to {y.index[test_end-1]} ({len(y_test)})")

        cv_splits.append((X_train, y_train, X_test, y_test))
        
    return cv_splits


def get_supermag_data(path: str, file: str, fill: int, targets: list[str]) -> DataFrame:
    """Returns SuperMAG data in a pandas DataFrame."""
    superMAG: DataFrame = read_csv(path + file, sep=",")
    superMAG["Date_UTC"] = to_datetime(superMAG["Date_UTC"])
    superMAG.set_index("Date_UTC", inplace=True)
    
    # Print stats for each target.
    for target in targets:
        t: Series = superMAG[target]
        print(f"{target} stats: mean={t.mean():.2f}, std={t.std():.2f}, min={t.min():.2f}, max={t.max():.2f}")
    
    # Filter out rows where any target has the fill value.
    mask = npall(superMAG[targets] != fill, axis=1)
    superMAG = superMAG[mask]
    print(f"SuperMAG shape after removing {fill} values from {targets}: {superMAG.shape}")
    
    return superMAG


def load_model(path: str) -> Booster:
    """Returns an XGBoost model."""
    model = Booster()
    model.load_model(path)
    return model


def merge_data(CDAWeb: DataFrame, SuperMAG: DataFrame) -> DataFrame:
    """Returns the CDAWeb and SuperMAG data merged into a DataFrame."""
    merge: DataFrame = CDAWeb.join(SuperMAG, how="inner")
    print(f"Merged data DataFrame shape: {merge.shape}")
    # Add lagged target features.
    data: DataFrame = add_lagged_features(merge, TARGETS)
    return data


def print_data_gaps(data: ndarray, limit: int = 300) -> None:
    """Prints data gaps longer than limit seconds."""
    gaps: ndarray = data.index.to_series().diff().dt.total_seconds()
    longer: ndarray = gaps[gaps > limit].value_counts()
    print(f"Gaps over {limit} mins in the dataset: {longer}")


def print_target_stats(targets: list[str], y_train: ndarray, y_test: ndarray, y_val: ndarray) -> None:
    for i, target in enumerate(targets):
        print(f"Train {target} mean: {y_train[:, i].mean():.2f}, std: {y_train[:, i].std():.2f}, max: {y_train[:, i].max():.2f}")
        print(f"Val {target} mean: {y_val[:, i].mean():.2f}, std: {y_val[:, i].std():.2f}, max: {y_val[:, i].max():.2f}")
        print(f"Test {target} mean: {y_test[:, i].mean():.2f}, std: {y_test[:, i].std():.2f}, max: {y_test[:, i].max():.2f}")


def resample_cdaweb_data(cdaweb_data: dict) -> DataFrame:
    """Returns a pandas DataFrame with the CDAWeb data resampled to one minute
    intervals."""
    merged: dict = {}
    for key in cdaweb_data:
        base_key: str = key.split('_', 2)[-1] # e.g. 'Magnitude'
        if base_key not in merged:
            merged[base_key] = cdaweb_data[key]
        else:
            merged[base_key] = merged[base_key].combine_first(
                cdaweb_data[key]
            ).fillna(cdaweb_data[key])
    framed = DataFrame(merged)
    # Set fill values to numpy nan.
    framed.replace(-1e31, nan, inplace=True)
    # Convert Unix time to datetime.
    framed.index = to_datetime(framed.index, unit="s")
    # Resample to one minute intervals to match SuperMAG data.
    resampled: DataFrame = framed.resample("1min").mean().interpolate()
    print(f"CDAWeb data shape: {resampled.shape}")
    
    # Return the processed CDAWeb data.
    return resampled


def set_environment_variable(data_dir: str = "spedas_data") -> None:
    """Sets SPEDAS_DATA_DIR environment variable."""
    download_dir: str = join(dirname(abspath(__file__)), data_dir)
    environ["SPEDAS_DATA_DIR"] = download_dir


def split_data(
    data: DataFrame, 
    features: list[str], 
    targets: list[str] = TARGETS,
) -> tuple[ndarray, ndarray]:
    """Splits the processed data into X (features) and y (targets)."""
    X: DataFrame = data[features]
    y: DataFrame = data[targets]
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    # Check the training data for gaps.
    print("Training data gap check:")
    print_data_gaps(X)
    # Check the target data for gaps.
    print("Target data gap check:")
    print_data_gaps(X)
    # Return the train-test-split
    return X, y


def stop_timing(script_start: float) -> None:
    """Stops timing the script and prints the elapsed time in minutes."""
    script_end: float = perf_counter()
    script_time: float = (script_end - script_start) / 60
    print(f"The script finished in {script_time:.2f} minutes.")
    
