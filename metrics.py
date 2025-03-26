"""Functions for evaluating model predictions."""

from matplotlib.pyplot import show, subplots, tight_layout
from numpy import mean as npmean, ndarray, std as npstd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import Booster

from constants import TARGETS


def plot_prediction(targets: list[str], y_test: ndarray, y_pred: ndarray) -> None:
    """Plots the prediction vs. true values."""
    a, axs = subplots(2, 2, figsize=(10, 10))
    print(type(axs), type(a))
    axes: ndarray = axs.flatten()
    print(type(axes))
    for i, (target, ax) in enumerate(zip(targets, axes)):
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
        ax.plot(
            [y_test[:, i].min(), y_test[:, i].max()], 
            [y_test[:, i].min(), y_test[:, i].max()], 
            "r--",
        )
        ax.set_xlim(y_test[:, i].min() - 5, y_test[:, i].max() + 5)
        ax.set_ylim(y_pred[:, i].min() - 5, y_pred[:, i].max() + 5)
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"Actual vs Predicted {target}")
    tight_layout()
    show()


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
    if total_splits == 0:
        print("\nTotal splits is zero, no importances to compute.")
        return
    print("\nFeature importances:")
    for i, feat in enumerate(features):
        imp: float = importances.get(f"f{i}", 0) / total_splits
        print(f"{feat}: {imp:.4f}")


def print_prediction_metrics(
    targets: list[str], y_test: ndarray, y_pred: ndarray, model: Booster,
) -> None:
    """Prints prediction metrics."""
    # Print prediction metrics.
    for i, target in enumerate(targets):
        print(f"\nMetrics for {target}:")
        print(f"R²: {r2_score(y_test[:, i], y_pred[:, i]):.4f}")
        print(f"MSE: {mean_squared_error(y_test[:, i], y_pred[:, i]):.2f}")
        print(f"MAE: {mean_absolute_error(y_test[:, i], y_pred[:, i]):.2f}")
    print(f"Best Iteration: {model.best_iteration}")
    print(f"Validation RMSE at best iteration: {model.best_score:.2f}")


def print_average_metrics(all_metrics: list[dict]) -> None:
    """Prints the average metrics across all training folds."""
    print("\nAverage metrics across all folds:")
    # Initialise a dictionary for each target's metrics.
    avg_metrics: dict[str, dict] = {
        target: {'R2': [], 'MSE': [], 'MAE': []} for target in TARGETS
    }
    
    # Set the values into the dictionary.
    for fold_metrics in all_metrics:
        for target in TARGETS:
            avg_metrics[target]['R2'].append(fold_metrics[target]['R2'])
            avg_metrics[target]['MSE'].append(fold_metrics[target]['MSE'])
            avg_metrics[target]['MAE'].append(fold_metrics[target]['MAE'])

    # Print the values from the dictionary.
    for target in TARGETS:
        print(f"\nAverage metrics for {target}:")
        print(f"R²: {npmean(avg_metrics[target]['R2']):.4f} (±{npstd(avg_metrics[target]['R2']):.4f})")
        print(f"MSE: {npmean(avg_metrics[target]['MSE']):.2f} (±{npstd(avg_metrics[target]['MSE']):.2f})")
        print(f"MAE: {npmean(avg_metrics[target]['MAE']):.2f} (±{npstd(avg_metrics[target]['MAE']):.2f})")
