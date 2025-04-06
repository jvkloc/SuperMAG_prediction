"""The main function for training an XGBoost model for SuperMAG SMR index 
prediction with rolling basis cross-validation. The script can be run also 
without training, using a saved model and saved preprocessed data.
"""

from argparse import ArgumentParser, Namespace
from time import perf_counter

from numpy import ndarray
from pandas import DataFrame
from xgboost import Booster

from constants import DESCRIPTION, DATA_FEATURES
from load_data import load_cdaweb_data, load_processed_data
from metrics import print_average_metrics, print_feature_importances
from plotting import (
    plot_evals_result,
    plot_features_time_series,
    plot_smr_histograms,
    prediction_scatter_plot,
    global_prediction_scatter_plot,
    prediction_time_series,
    global_prediction_time_series,
)
from load_model import load_model, predict_with_loaded_model
from training import (
    get_prediciton_dataframe,
    get_prediction_metrics,
    training_loop,
)
from utils import (
    add_arguments,
    get_cdaweb_data,
    get_initial_features,
    get_rolling_basis_cv_splits,
    get_supermag_data,
    merge_data,
    print_data_gaps,
    resample_cdaweb_data,
    set_environment_variable,
    split_data,
    stop_timing,
)


def main(description: str = DESCRIPTION, features: str = DATA_FEATURES) -> None:
    """Main function. TODO: args, assuming load model and load data or 
    download data and train a model using --train.
    """
    
    # Start timing.
    script_start: float = perf_counter()
    
    # Parse command-line arguments.
    parser = ArgumentParser(description=description)
    add_arguments(parser)
    args: Namespace = parser.parse_args()

    if args.train: 
        # Download data / check that it is current and train a model.
        
        # Set PySPEDAS environment variable.
        set_environment_variable()
        
        # Download CDAWeb data.
        load_cdaweb_data()

        # Get SuperMAG data from a file.
        SuperMAG: DataFrame = get_supermag_data()
        
        # Set CDAWeb data to a dictionary.
        CDAWeb_data: dict = get_cdaweb_data()
        # Resample CDAWeb data to match SuperMAG data time intervals.
        CDAWeb: DataFrame = resample_cdaweb_data(CDAWeb_data)

        # Combine both data to one DataFrame.
        data: DataFrame = merge_data(CDAWeb, SuperMAG)

        # Get initial features (lagged features are not added yet).
        initial_features: list[str] = get_initial_features(data, SuperMAG)
        
        # Get features (X) and targets (y). 
        X: DataFrame; y: DataFrame
        X, y = split_data(data, initial_features)

        # Get rolling basis cross validation splits.
        cv_splits: list[tuple] = get_rolling_basis_cv_splits(X, y)
        
        # Train the model.
        metrics_per_fold: list[dict] = []
        model: Booster; y_pred: DataFrame; y_test: DataFrame; 
        X_test: DataFrame; features: list[str]; evals: dict
        model, y_pred, y_test, X_test, features, evals = training_loop(
            cv_splits, metrics_per_fold
        )
        
        # Plot train and test root mean square errors.
        plot_evals_result(evals, model.best_iteration)

        # Print model metrics.
        print_average_metrics(metrics_per_fold)

    else: 
        # Default to loading preprocessed data and a saved model.
        
        # Load the preprocessed data.
        data: DataFrame = load_processed_data(args.data_path)
        
        # Check possible gaps in the loaded data.
        print_data_gaps(data)
        
        # Get features (X) and targets (y).
        X: DataFrame; y: DataFrame
        X, y = split_data(data, features)
        
        # X_test and y_test for functions.
        X_test: DataFrame; y_test: DataFrame
        X_test, y_test = X, y

        # Load the model.
        model: Booster = load_model(args.model_path)

        # Predict.
        prediction: ndarray = predict_with_loaded_model(model, X_test, y_test)
        y_pred: DataFrame = get_prediciton_dataframe(prediction, y_test)

        # Print prediction metrics.
        get_prediction_metrics(y_test, y_pred, model)

    # Print feature importances from the final model.
    print_feature_importances(model, features)

    # Stop timing the script and print the elapsed time.
    stop_timing(script_start)

    # Plot SMR density histograms.
    plot_smr_histograms(y) 

    # Plot SMR MLT predictions vs. true values scatter plot from final fold.
    prediction_scatter_plot(y_pred, y_test)

    # Plot SMR predictions vs. true values scatter plot from final fold.
    global_prediction_scatter_plot(y_pred, y_test)

    # Plot feature time series.
    plot_features_time_series(X_test)

    # Plot SMR MLT prediction vs. true values time series from final fold.
    prediction_time_series(y_pred, y_test)

    # Plot SMR prediction vs. true values time series from final fold.
    global_prediction_time_series(y_pred, y_test)


if __name__ == "__main__":
    main()
