"""Functions for evaluating the model."""

from numpy import mean as npmean, std as npstd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import Booster

from constants import TARGETS


def print_average_metrics(
    all_metrics: list[dict], targets: list[str] = TARGETS
) -> None:
    """Prints the average metrics across all training folds."""
    print("\nAverage metrics across all folds:")
    # Initialize a dictionary for each target's metrics.
    avg_metrics: dict[str, dict] = {
        target: {'R2': [], 'MSE': [], 'RMSE': [], 'MAE': []} for target in targets
    }
    
    # Set the values into the dictionary.
    for fold_metrics in all_metrics:
        for target in targets:
            avg_metrics[target]['R2'].append(fold_metrics[target]['R2'])
            avg_metrics[target]['MSE'].append(fold_metrics[target]['MSE'])
            avg_metrics[target]['RMSE'].append(fold_metrics[target]['RMSE'])
            avg_metrics[target]['MAE'].append(fold_metrics[target]['MAE'])

    # Print the values from the dictionary.
    for target in targets:
        print(f"\nAverage metrics for {target}:")
        print(f"R²: {npmean(avg_metrics[target]['R2']):.4f} (±{npstd(avg_metrics[target]['R2']):.4f})")
        print(f"MSE: {npmean(avg_metrics[target]['MSE']):.2f} (±{npstd(avg_metrics[target]['MSE']):.2f})")
        print(f"RMSE: {npmean(avg_metrics[target]['RMSE']):.2f} (±{npstd(avg_metrics[target]['RMSE']):.2f})")
        print(f"MAE: {npmean(avg_metrics[target]['MAE']):.2f} (±{npstd(avg_metrics[target]['MAE']):.2f})")


def print_feature_importances(model: Booster, features: list[str]) -> None:
    """Prints feature importances."""
    try:
        importances: dict = model.get_score(importance_type="weight")
    except Exception as e:
        print(f"get_score failed: {e}")
    if not importances:
        print("\nNo feature importances available.")
        return
    total_splits: int = sum(importances.values())
    print("\nFeature importances:\n")
    for feature in features:
        importance: float = importances.get(feature, 0) / total_splits
        print(f"{feature}: {importance:.4f}")


def print_prediction_metrics(
    y_test: DataFrame,
    y_pred: DataFrame,
    model: Booster,
    targets: list[str] = TARGETS
) -> None:
    """Prints prediction metrics."""
    for i, target in enumerate(targets):
        print(f"\nMetrics for {target}:")
        print(f"R²: {r2_score(y_test[:, i], y_pred[:, i]):.4f}")
        print(f"MSE: {mean_squared_error(y_test[target], y_pred[target]):.2f}")
        print(f"MAE: {mean_absolute_error(y_test[target], y_pred[target]):.2f}")
    print(f"Best Iteration: {model.best_iteration}")
    print(f"Test RMSE at best iteration: {model.best_score:.2f}")
