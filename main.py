"""Script for training an XGBoost model for SuperMAG SMR index prediction with 
rolling basis cross-validation.
"""

from time import perf_counter

from numpy import ndarray
from pandas import DataFrame
from xgboost import Booster

from download_data import load_cdaweb_data
from metrics import plot_prediction, print_average_metrics, print_feature_importances
from training import training_loop
from utils import (
    set_environment_variable,
    get_supermag_data,
    get_cdaweb_data,
    resample_cdaweb_data,
    merge_data,
    get_features,
    split_data,
    get_rolling_basis_cv_splits,
    stop_timing,
)


def main() -> None:    
    # Set PySPEDAS download directory.
    set_environment_variable()
    
    # Download CDAWeb data with timing. TODO: use this via command line arg.
    load_cdaweb_data()

    # Start timing the rest of the script.
    script_start: float = perf_counter()

    # Get SuperMAG data.
    SuperMAG: DataFrame = get_supermag_data()

    # Get CDAWeb data.
    CDAWeb_data: dict = get_cdaweb_data()
    # Resample the CDAWeb data to one minute intervals to match SuperMAG data.
    CDAWeb: DataFrame = resample_cdaweb_data(CDAWeb_data)

    # Merge the CDAWeb data with the SuperMAG data and add lagged features.
    data: DataFrame = merge_data(CDAWeb, SuperMAG)
    features: list[str] = get_features(data, SuperMAG)

    # Features/targets split.
    X: DataFrame; y: DataFrame
    X, y = split_data(data, features)

    # Get rolling basis cross-validation splits.
    cv_splits: list[tuple] = get_rolling_basis_cv_splits(X, y)

    # Train and evaluate a model for each fold.
    metrics_per_fold: list[dict] = []
    model: Booster; y_pred: ndarray; y_test: ndarray
    # TODO: command line arg for loading the model.
    model, y_pred, y_test = training_loop(cv_splits, metrics_per_fold)

    # Print average metrics across folds.
    print_average_metrics(metrics_per_fold)

    # Print feature importances from the final model.
    print_feature_importances(model, features)

    # Stop timing the script and print the elapsed time.
    stop_timing(script_start)

    # Plot predictions from final fold.
    plot_prediction(y_pred, y_test)


if __name__ == "__main__":
    main()
