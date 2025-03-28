"""Training loop and training loop utility functions for the main script."""

from numpy import abs as npabs, ndarray, where
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import Booster, DMatrix, train

from constants import TARGETS, THRESHOLD, XGB_PARAMS, N_ESTIMATORS, EARLY_STOPPING_ROUNDS


def get_prediction_metrics(
    y_test: ndarray, y_pred: ndarray, model: Booster, targets: list[str] = TARGETS
) -> dict:
    """Prints prediction metrics and returns them in a dictionary."""
    metrics: dict[str, dict] = {target: {} for target in targets}
    
    for i, target in enumerate(targets):
        print(f"\nMetrics for {target}:")
        r2: float = r2_score(y_test[:, i], y_pred[:, i])
        mse: float = mean_squared_error(y_test[:, i], y_pred[:, i])
        mae: float = mean_absolute_error(y_test[:, i], y_pred[:, i])
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        # Add metrics to the dictionary.
        metrics[target] = {'R2': r2, 'MSE': mse, 'MAE': mae}
    
    print(f"Best Iteration: {model.best_iteration}")
    print(f"Validation RMSE at best iteration: {model.best_score:.2f}")
    
    return metrics


def get_prediction_outliers(
    y_test: ndarray,
    y_pred: ndarray,
    threshold: int = THRESHOLD,
    targets: list[str] = TARGETS,
) -> ndarray:
    """Returns prediction outliers based on the threshold."""
    errors: ndarray = npabs(y_test - y_pred)  # Shape: (n_test, 4)
    return [where(errors[:, i] > threshold)[0] for i in range(len(targets))]


def predict_with_loaded_model(model: Booster, X_test: ndarray) -> ndarray:
    """Predict with best iteration of the given model."""
    dtest = DMatrix(X_test)
    y_pred: ndarray = model.predict(dtest, ntree_limit=model.best_iteration)
    print(f"Loaded model best iteration: {model.best_iteration}")
    return y_pred


def print_outliers(
    outliers: ndarray, 
    fold_nbr: int,
    threshold: float = THRESHOLD,
    targets: list[str] = TARGETS,
) -> None:
    """Prints prediction outliers."""
    for i, target in enumerate(targets):
        nbr: int = len(outliers[i])
        print(f"Found {nbr} outliers (error > {threshold} nT) for {target} in fold {fold_nbr}")


def print_results(y_test: ndarray, y_pred: ndarray, targets: list[str] = TARGETS) -> None:
    """Prints prediction results."""
    y_test_df = DataFrame(y_test, columns=targets)
    for i in range(0, len(y_test), 100):
        for j, target in enumerate(targets):
            print(f"{y_test_df.index[i]} {target} True: {y_test[i,j]:.2f}, Pred: {y_pred[i,j]:.2f}")


def train_xgboost(
    X_train: ndarray,
    X_test: ndarray,
    y_train: ndarray,
    y_test: ndarray,
    params: dict = XGB_PARAMS,
    estimators: int = N_ESTIMATORS,
    early_stop: int = EARLY_STOPPING_ROUNDS,
) -> tuple[Booster, DMatrix]:
    """Creates and trains an XGBoost model. Returns the model and test data."""
    # Set parameters.
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)
    evals: list[tuple] = [(dtest, "test")]
    # Train model.
    model: Booster = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=estimators,
        evals=evals,
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )
    # Return the trained model and test data.
    return model, dtest


def training_loop(
    cv_splits: list[tuple],
    metrics_per_fold: list[dict],
    model_path: str = MODEL_PATH,
) -> tuple[Booster, ndarray, ndarray]:
    """Trains an XGBoost model for each rolling basis training fold. Saves and
    returns the last model which uses all the training data, as well as 
    predictions and true values."""

    total_folds: int = len(cv_splits)
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_splits):
        fold_nbr: int = i + 1
        print(f"\nFold {fold_nbr}/{total_folds}")
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # Create and train the model.
        model: Booster; dtest: DMatrix 
        model, dtest = train_xgboost(X_train, X_test, y_train, y_test)
        
        # Predict.
        iteration_range: tuple = (0, model.best_iteration + 1)
        y_pred: ndarray = model.predict(dtest, iteration_range=iteration_range)
        
        # Print metrics and append them to the metrics list.
        fold_metrics: dict = get_prediction_metrics(y_test, y_pred, model)
        metrics_per_fold.append(fold_metrics)
        
        # Print outliers.
        outliers: ndarray = get_prediction_outliers(y_pred, y_test)
        print_outliers(outliers, fold_nbr)

        if fold_nbr == total_folds:
            # Save the model trained on the whole training data.
            print(f"Saving the model from fold {fold_nbr}/{total_folds}...")
            model.save_model(model_path)
            print(f"Model saved to {model_path}")
            # Return the model, predicion and the true values.
            return model, y_pred, y_test
