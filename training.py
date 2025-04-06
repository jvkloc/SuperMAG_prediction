"""Training loop and training loop utility functions."""

from numpy import ndarray, sqrt
from pandas import concat, DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import Booster, DMatrix, train

from constants import (
    TARGETS,
    THRESHOLD,
    XGB_PARAMS,
    N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    MODEL_PATH
)
from load_data import save_processed_data, unsplit_data
from utils import get_features


def add_lagged_features(data: DataFrame, targets: list[str] = TARGETS) -> DataFrame:
    """Adds lagged SMR values as features to the data DataFrame."""
    for target in targets:
        data[f"{target}_lag1"] = data[target].shift(1)
        data[f"{target}_lag2"] = data[target].shift(2)
        data[f"{target}_lag3"] = data[target].shift(3)
        data[f"{target}_lag5"] = data[target].shift(5)
    # Return the data with lagged features.
    return data


def get_lagged_split(
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_test: DataFrame,
    targets: list[str] = TARGETS,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Returns train-test split with lagged features."""
    # Add lagged features to training data.
    train: DataFrame = concat([X_train, y_train], axis=1)
    lagged_train: DataFrame = add_lagged_features(train, targets)
    print(f"Training data shape with lags: {lagged_train.shape}")

    # Add lagged features to test data.
    test: DataFrame = concat([X_test, y_test], axis=1)
    full_data: DataFrame = concat([train, test], axis=0).dropna(how="all") 
    lagged_full_data: DataFrame = add_lagged_features(full_data, targets)
    print(f"Full data shape with lags: {lagged_full_data.shape}")
    lagged_test: DataFrame = lagged_full_data.iloc[len(train):]

    # Split the data back into X and y.
    lagged_X_train: DataFrame = lagged_train.drop(columns=targets)
    lagged_y_train: DataFrame = lagged_train[targets]
    lagged_X_test: DataFrame = lagged_test.drop(columns=targets)
    lagged_y_test: DataFrame = lagged_test[targets]

    print(f"Training samples: {len(lagged_X_train)}, Testing samples: {len(lagged_X_test)}")

    return lagged_X_train, lagged_y_train, lagged_X_test, lagged_y_test


def get_prediciton_dataframe(
    prediction: ndarray,
    lagged_y_test: DataFrame,
    targets: list[str] = TARGETS,
) -> DataFrame:
    """Returns columns for predicted values."""
    index: DatetimeIndex = lagged_y_test.index
    return DataFrame(prediction, columns=targets, index=index)


def get_prediction_metrics(
    y_test: DataFrame,
    y_pred: DataFrame,
    model: Booster,
    targets: list[str] = TARGETS,
) -> dict:
    """Prints prediction metrics and returns them in a dictionary."""
    metrics: dict[str, dict] = {target: {} for target in targets}
    
    for target in targets:
        print(f"\nMetrics for {target}:")
        r2: float = r2_score(y_test[target], y_pred[target])
        mse: float = mean_squared_error(y_test[target], y_pred[target])
        rmse: float = sqrt(mse)
        mae: float = mean_absolute_error(y_test[target], y_pred[target])
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        # Add metrics to the dictionary.
        metrics[target] = {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    
    print(f"\nBest Iteration: {model.best_iteration}")
    print(f"Validation RMSE at best iteration: {model.best_score:.2f}")
    
    return metrics


def get_prediction_outliers(
    y_test: DataFrame,
    y_pred: DataFrame,
    threshold: int = THRESHOLD,
    targets: list[str] = TARGETS,
) -> list[DatetimeIndex]:
    """Returns prediction outliers based on the threshold."""

    if y_test.shape != y_pred.shape or not all(y_test.columns == y_pred.columns):
        raise ValueError("y_test and y_pred must have the same shape and columns")

    outliers: DataFrame = (y_test - y_pred).abs()

    print(f"Max absolute errors: {outliers.max().to_dict()}")

    return [
        outliers.index[outliers.iloc[:, i] > threshold] 
        for i in range(len(targets))
    ]


def print_outliers(
    outliers: list[DatetimeIndex], 
    fold_nbr: int,
    threshold: float = THRESHOLD,
    targets: list[str] = TARGETS,
) -> None:
    """Prints prediction outliers."""
    for i, target in enumerate(targets):
        nbr: int = len(outliers[i])
        print(f"Found {nbr} outliers (error > {threshold} nT) for {target} in fold {fold_nbr}")


def print_results(y_test: DataFrame, y_pred: DataFrame, targets: list[str] = TARGETS) -> None:
    """Prints prediction results."""
    for i in range(0, len(y_test), 100):
        for j, target in enumerate(targets):
            print(f"{y_test.index[i]} {target} True: {y_test.iloc[i, j]:.2f}, Pred: {y_pred.iloc[i, j]:.2f}")


def print_training_rmse(model: Booster, dtrain: DMatrix, y_train: DataFrame) -> None:
    """Prints the training root mean square error."""
    y_train_pred: ndarray = model.predict(dtrain)
    train_rmse: float = mean_squared_error(y_train, y_train_pred)
    print(f"Train RMSE: {sqrt(train_rmse):.2f}")


def train_xgboost(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
    params: dict = XGB_PARAMS,
    estimators: int = N_ESTIMATORS,
    early_stop: int = EARLY_STOPPING_ROUNDS,
) -> tuple[Booster, DMatrix, DMatrix, dict]:
    """Creates and trains an XGBoost model. Returns the model, train and test 
    data and train and test root mean square error values."""
    # Set parameters.
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)
    evals: list[tuple] = [(dtrain, "train"), (dtest, "test")]
    evals_result: dict = {}
    # Train model.
    model: Booster = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=estimators,
        evals=evals,
        evals_result = evals_result,
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )
    # Return the trained model and test data.
    return model, dtrain, dtest, evals_result


def training_loop(
    cv_splits: list[tuple],
    metrics_per_fold: list[dict],
    model_path: str = MODEL_PATH,
) -> tuple[Booster, DataFrame, DataFrame, DataFrame, dict]:
    """Trains an XGBoost model for each rolling basis training fold. Saves and
    returns the last model which uses all the training data. Returns also 
    predictions, true test values, X test values and train and test root 
    mean square error values."""

    total_folds: int = len(cv_splits)
    for i, (Xtrain, ytrain, Xtest, ytest) in enumerate(cv_splits):
        fold_nbr: int = i + 1
        print(f"\nFold {fold_nbr}/{total_folds}")
        
        # Add lagged features.
        X_train: DataFrame; y_train: DataFrame; X_test: DataFrame; y_test: DataFrame
        X_train, y_train, X_test, y_test = get_lagged_split(
            Xtrain, ytrain, Xtest, ytest
        )

        # Create and train the model.
        model: Booster; dtrain: DMatrix; dtest: DMatrix; evals: dict 
        model, dtrain, dtest, evals = train_xgboost(X_train, X_test, y_train, y_test)
        
        # Predict.
        iter_range: tuple = (0, model.best_iteration + 1)
        prediction: ndarray = model.predict(dtest, iteration_range=iter_range)
        y_pred: DataFrame = get_prediciton_dataframe(prediction, y_test)

        # Print metrics and append them to the metrics list.
        fold_metrics: dict = get_prediction_metrics(y_test, y_pred, model)
        metrics_per_fold.append(fold_metrics)

        # Print training RMSE to check for overfitting.
        print_training_rmse(model, dtrain, y_train)
        
        # Print outliers.
        outliers: ndarray = get_prediction_outliers(y_pred, y_test)
        print_outliers(outliers, fold_nbr)

        if fold_nbr == total_folds:
            # Save the model trained on the whole training data.
            print(f"Saving the model from fold {fold_nbr}/{total_folds}...")
            model.save_model(model_path)
            print(f"Model saved to {model_path}")
            # Merge the data back to one DataFrame for saving.
            training_data: DataFrame; full_data: DataFrame 
            training_data, full_data = unsplit_data(X_train, y_train, X_test, y_test)
            # Save the processed data.
            save_processed_data(full_data)
            # Get features.
            features: list[str] = get_features(training_data)
            # Return the model, prediction, true values, features and evaluation.
            return model, y_pred, y_test, X_test, features, evals
