"""The main function for training an XGBoost model for SuperMAG SMR index 
prediction with rolling basis cross-validation. The script can be run also 
without training, using a saved model for predicting."""

from argparse import ArgumentParser, Namespace
from time import perf_counter

from numpy import ndarray
from polars import DataFrame as PolarsFrame, LazyFrame
from xgboost import Booster

from constants import DESCRIPTION, PATH, XGB_METRIC
from data_utils import (
    load_cdaweb_data,
    load_cdaweb_from_disk,
    load_preprocessed_data,
    get_cdaweb_data,
    get_supermag_data,
    merge_data,
    resample_cdaweb_data,
)
from load_model import load_xgb_model, predict_with_loaded_model
from metrics import *
from plotting import *
from training import add_lagged_features, get_prediction_dataframe, training_loop
from utils import (
    add_arguments,
    get_initial_features,
    get_rolling_basis_cv_folds,
    print_data_gaps,
    set_environment_variable,
    split_data,
    stop_timing,
)


def main(
    description: str = DESCRIPTION,
    smag_path: str = PATH,
    model_metric: str = XGB_METRIC
) -> None:
    """Main function. Runnig with --newdata downloads (if necessary) the data
    and trains a model and saves it. Running with --train loads preprocessed 
    data and trains a model wiht that. No command line argument results in a 
    prediction with a saved model. TODO: Note that the dates need to be set 
    manually in each case. Use args instead."""
    
    # Start timing.
    script_start: float = perf_counter()
    
    # Parse command-line arguments.
    parser = ArgumentParser(description=description)
    add_arguments(parser)
    args: Namespace = parser.parse_args()

    # Set PySPEDAS environment variable.
    set_environment_variable()
    
    # Dictionary for model statistics. 
    evals: dict | None = None

    if args.newdata: # Download/check CDAWeb data and train a model.
        
        # Download CDAWeb data.
        load_cdaweb_data()
        # Set CDAWeb data to a dictionary.
        CDAWeb_data: LazyFrame = get_cdaweb_data()
        # Resample CDAWeb data to match SuperMAG data time intervals.
        CDAWeb: LazyFrame = resample_cdaweb_data(CDAWeb_data)
        
        # Get SuperMAG data from a file.
        SuperMAG: LazyFrame = get_supermag_data(path=smag_path)

        # Combine both data to one DataFrame.
        data: LazyFrame = merge_data(CDAWeb, SuperMAG)
        
        # Get initial features (lagged features do not exist yet).
        initial_features: list[str] = get_initial_features(data, SuperMAG)
        
        # Split the data into features (X) and targets (y). 
        X: PolarsFrame; y: PolarsFrame; gaps: dict
        X, y, gaps = split_data(data, initial_features)
        
        # Get rolling basis cross validation splits.
        train: PolarsFrame; validation: PolarsFrame; 
        cv_splits: list[tuple]; extreme: dict
        train, validation, cv_splits, extreme = get_rolling_basis_cv_folds(X, y)
        
        # Train the model.
        metrics_per_fold: list[dict] = []
        model: Booster; y_pred: PolarsFrame; y_test: PolarsFrame; 
        X_test: PolarsFrame; features: list[str]; evals: dict
        model, y_pred, y_test, X_test, features, evals = training_loop(
            cv_splits, metrics_per_fold, extreme, newdata=args.newdata
        )
        
        # Get model metrics.
        metrics: dict = compute_average_metrics(metrics_per_fold)

        # Print the metrics.
        print_metrics(metrics)
    
    elif args.train: # Train a model with preprocessed data.
        
        # Get the data.
        data: LazyFrame = load_preprocessed_data()
        
        # Get SuperMAG data
        SuperMAG: LazyFrame = get_supermag_data(path=smag_path)
        
        # Get initial features (lagged features do not exist yet).
        initial_features: list[str] = get_initial_features(data, SuperMAG)
        
        # Split the data into features (X) and targets (y). 
        X: PolarsFrame; y: PolarsFrame; gaps: dict
        X, y, gaps = split_data(data, initial_features)
        
        # Get rolling basis cross validation splits.
        train: PolarsFrame; validation: PolarsFrame; 
        cv_splits: list[tuple]; extreme: dict
        train, validation, cv_splits, extreme = get_rolling_basis_cv_folds(X, y)
        
        # Train the model.
        metrics_per_fold: list[dict] = []
        model: Booster; y_pred: PolarsFrame; y_test: PolarsFrame; 
        X_test: PolarsFrame; features: list[str]; evals: dict
        model, y_pred, y_test, X_test, features, evals = training_loop(
            cv_splits, metrics_per_fold, extreme
        )
        
        # Get model metrics.
        metrics: dict = compute_average_metrics(metrics_per_fold)

        # Print the metrics.
        print_metrics(metrics)

    else: # Load future data and a saved model for predicting it.

        # Download CDAWeb data.
        start: str = "2023-01-01 00:00:00"
        end: str = "2023-01-07 23:59:59"
        load_cdaweb_data(start=start, end=end)
        
        # Set CDAWeb data to a dictionary.
        CDAWeb_data: LazyFrame = get_cdaweb_data()
        # Resample CDAWeb data to match SuperMAG data time intervals.
        CDAWeb: LazyFrame = resample_cdaweb_data(CDAWeb_data, start=start, end=end)
        
        # Get SuperMAG data from a file.
        file: str = "2023-jan-firstweek.csv"
        SuperMAG: LazyFrame = get_supermag_data(file=file, path=smag_path)
        
        # Combine both data to one DataFrame.
        merged: LazyFrame = merge_data(CDAWeb, SuperMAG)
        
        #print("CDAWeb_data columns:", CDAWeb_data.collect().columns)
        #print("Merged data columns:", merged.collect().columns)
        
        # Add lags to the data.
        data: LazyFrame = add_lagged_features(merged)
        
        # Load the model.
        model: Booster = load_xgb_model(args.modelpath)
        # Get model features.
        features: list[str] = model.feature_names
        #print("Model features:", model.feature_names)
        #print("DataFrame columns:", data.collect().columns)
        
        # Split the data into features (X) and targets (y). 
        X_test: PolarsFrame; y_test: PolarsFrame; gaps: dict
        X_test, y_test, gaps = split_data(data, features)
        
        # Predict.
        prediction: ndarray = predict_with_loaded_model(model, X_test, y_test)
        y_pred: PolarsFrame = get_prediction_dataframe(prediction, y_test)

        # Print prediction metrics.
        print_prediction_metrics(y_test, y_pred)
    
    # Print data gaps.
    print_data_gaps(gaps)

    # Print feature importance from the final model.
    print_feature_importance(model, features)
    
    # Drop index from true and predicted values.
    true: PolarsFrame; pred: PolarsFrame
    true, pred = y_test.drop("index"), y_pred.drop("index")
    
    # Get residuals.
    residuals: PolarsFrame = true - pred
    
    # Stop timing the script and print the elapsed time.
    stop_timing(script_start)
    
    if evals is not None: # Script ran in training mode.
        # Plot train and test root mean square errors.
        plot_evals_result(evals, model_metric, model.best_iteration)
        # Print extreme week models' MAE and RMSE.
        evaluate_extreme_performance(model, validation)
        # Plot train and test data SMR density histograms.
        plot_smr_histograms(train, "Train")
        plot_smr_histograms(validation, "Validation")

    # Print the number of unique predicion values and true values.
    print_unique_values(true, pred)

    # Print prediction statistics of the extreme values.
    print_extreme_predictions(true, pred)

    # Plot residuals' density.
    #residual_density_histogram(residuals)

    # Plot residuals vs. predicted values. Should be random around zero line.
    #residuals_vs_predicted_scatter(residuals, y_pred)

    # Q-Q plot for residual normality check. XGBoost does not require normality.
    #q_q_plot(residuals) 

    # Plot test data feature time series.
    plot_features_time_series(X_test)
    
    # Plot preditions vs. true values.
    
    # Plot SMR MLT predictions vs. true values scatter plot from final fold.
    prediction_scatter_plot(y_pred, y_test)

    # Plot SMR predictions vs. true values scatter plot from final fold.
    global_prediction_scatter_plot(y_pred, y_test)

    # Plot SMR MLT prediction vs. true values time series from final fold.
    prediction_time_series(y_pred, y_test)

    # Plot SMR prediction vs. true values time series from final fold.
    global_prediction_time_series(y_pred, y_test)


if __name__ == "__main__":
    main()
