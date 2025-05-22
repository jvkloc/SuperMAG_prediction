"""Plotting functions."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from polars import DataFrame as PolarsFrame
from scipy.stats import probplot

from constants import FEATURES, TARGETS


def plot_features_time_series(X_test: PolarsFrame, features: dict = FEATURES) -> None:
    """Plots feature time series in three figures, each with three subplots."""
    time: ndarray = X_test["index"].to_numpy()

    for group_name, group_features in features.items():
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes: ndarray = ax.flatten()
        
        colors: list[str] = ["blue", "orange", "green"]

        for i, (feature, color) in enumerate(zip(group_features, colors)):
            values: ndarray = X_test[feature].to_numpy()
            
            axes[i].plot(time, values, color=color, label=feature, alpha=0.7)
            axes[i].set_ylabel(f"{feature} value")
            axes[i].set_title(f"{feature}")
            axes[i].legend()
            axes[i].grid(True)

        axes[2].set_xlabel("Time")
        fig.suptitle(f"{group_name} time series", y=1.02)
        plt.tight_layout()
        plt.show()


def plot_evals_result(evals_result: dict, metric: str, best_iteration: int) -> None:
    """Plot train and validation evaluation metric as a scatterplot."""
    train_rmse: list[float] = evals_result['train'][metric]
    test_rmse: list[float] = evals_result['test'][metric]
    iterations = range(len(train_rmse))  # X-axis: iteration numbers
    plt.figure(figsize=(10, 6))
    plt.scatter(iterations, train_rmse, label=f"Training {metric}", color="blue")
    plt.scatter(iterations, test_rmse, label=f"Validation {metric}", color="orange")
    # Set vertical line to best iteration number on x-axis.
    plt.axvline(
        x=best_iteration,
        color="green",
        linestyle="--",
        label=f"Best iteration ({best_iteration})",
    )
    # Set axes' labels. 
    plt.xlabel("Iteration")
    plt.ylabel(metric.upper())
    # Set plot title and legend.
    plt.title(f"Training vs. validation {metric.upper()}")
    plt.legend()
    # Plot image.
    plt.grid(True)
    plt.show()


def prediction_scatter_plot(
    y_pred: PolarsFrame,
    y_test: PolarsFrame,
    targets: list[str] = TARGETS
) -> None:
    """Plots prediction vs. true values as a scatter plot."""
    _: ndarray; axs: Figure
    _, axs = plt.subplots(2, 2, figsize=(10, 10))
    axes: ndarray = axs.flatten()
    # Loop over the targets.
    for target, ax in zip(targets[1:], axes):
        ax.scatter(y_test[target], y_pred[target], marker='.', s=5, alpha=0.5)
        ax.plot(
            [y_test[target].min(), y_test[target].max()], 
            [y_test[target].min(), y_test[target].max()], 
            "r--",
        )
        # Set axes' limits.
        ax.set_xlim(y_test[target].min() - 5, y_test[target].max() + 5)
        ax.set_ylim(y_pred[target].min() - 5, y_pred[target].max() + 5)
        # Set axes' labels.
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        # set title.
        ax.set_title(f"True vs. predicted {target}")
    # Adjust layout and display,
    plt.tight_layout()
    plt.show()


def global_prediction_scatter_plot(
    y_pred: PolarsFrame,
    y_test: PolarsFrame,
    target: str = "SMR",
) -> None:
    """Plots prediction vs. true values as a scatter plot."""
    _: ndarray; axes: Figure
    _, axes = plt.subplots(figsize=(8, 6))
    axes.scatter(y_test[target], y_pred[target], marker='.', s=5, alpha=0.5)
    axes.plot(
        [y_test[target].min(), y_test[target].max()], 
        [y_test[target].min(), y_test[target].max()], 
        "r--",
    )
    # Set axes' limits.
    axes.set_xlim(y_test[target].min() - 5, y_test[target].max() + 5)
    axes.set_ylim(y_pred[target].min() - 5, y_pred[target].max() + 5)
    # Set labels and title.
    axes.set_xlabel(f"True {target}")
    axes.set_ylabel(f"Predicted {target}")
    axes.set_title(f"True vs. predicted {target}")
    # Adjust layout and display,
    plt.tight_layout()
    plt.show()


def prediction_time_series(
    y_pred: PolarsFrame,
    y_test: PolarsFrame,
    targets: list[str] = TARGETS,
) -> None:
    """Plots local SMR predictions vs. true values as time series."""
    _, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes: ndarray = axs.flatten()

    true_color: str = "blue"
    pred_color: str = "orange"
    time: ndarray = y_test["index"].to_numpy()

    for i, target in enumerate(targets[1:]):
        ax: int = axes[i]
        ax.plot(
            time,
            y_test[target].to_numpy(),
            color=true_color,
            label=f"True {target}",
            alpha=0.7,
        )
        ax.plot(
            time,
            y_pred[target].to_numpy(),
            color=pred_color,
            linestyle="--",
            label=f"Pred {target}",
            alpha=0.7,
        )
        ax.set_ylabel(f"{target} (nT)")
        ax.set_title(f"{target}: True vs. predicted")
        ax.legend()
        ax.grid(True)

    axes[2].set_xlabel("Time")
    axes[3].set_xlabel("Time")

    plt.tight_layout()
    plt.show()


def global_prediction_time_series(
    y_pred: PolarsFrame,
    y_test: PolarsFrame,
    target: str = "SMR",
) -> None:
    """Plots global SMR prediction vs. true values as time series."""
    plt.figure(figsize=(10, 6))
    
    # Extract time axis and values
    time: ndarray = y_test["index"].to_numpy()
    true_vals: ndarray = y_test[target].to_numpy()
    pred_vals: ndarray = y_pred[target].to_numpy()

    # Plot
    plt.plot(time, true_vals, color="blue", label=f"True {target}", alpha=0.7)
    plt.plot(time, pred_vals, color="orange", linestyle="--", label=f"Pred {target}", alpha=0.7)

    plt.xlabel("Time")
    plt.ylabel(f"{target} (nT)")
    plt.title(f"{target}: True vs. predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_smr_histograms(
    data: PolarsFrame, label: str, targets: list[str] = TARGETS
) -> None:
    """Plots density histograms of SMR and SMR MLT values from the data."""
    _: ndarray; axes: Figure
    _, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes: ndarray = axes.flatten()
    
    # A color for each target.
    colors = ["gray", "blue", "orange", "green", "red"]
    
    # Plot histograms for each target.
    for i, (target, color) in enumerate(zip(targets, colors)):
        axes[i].hist(data[target], bins=50, color=color, alpha=0.7, density=True)
        axes[i].set_title(f"{target}")
        axes[i].set_xlabel(f"{target} (nT)")
        axes[i].set_ylabel(f"{label} data SMR densities")
        axes[i].grid(True)
    
    # Hide the unused subplot (bottom right).
    axes[5].axis("off")
    # Display the images.
    plt.tight_layout()
    plt.show()


def residual_density_histogram(residuals: PolarsFrame) -> None:
    """Plots residuals in a normalized histogram."""
    values: ndarray = residuals.select(residuals.columns[0]).to_series().to_numpy()
    plt.hist(values, bins=100, edgecolor="black", density=True)
    plt.title("Residual histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def residuals_vs_predicted_scatter(residuals: PolarsFrame, y_pred: PolarsFrame) -> None:
    """Scatter plot of residuals vs. predicted values. Residuals should be
    randomly scattered around zero line."""
    for col in residuals.columns:
        # Convert to NumPy arrays.
        y: ndarray = y_pred[col].to_numpy()
        r: ndarray = residuals[col].to_numpy()
        # Plot.
        plt.figure(figsize=(10, 6))
        plt.scatter(y, r, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle="--")
        plt.title(f"Residuals vs. predicted: {col}")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.tight_layout()
        plt.show()


def q_q_plot(residuals: PolarsFrame) -> None:
    """Q-Q plot for checking normality of the residuals. Note that XGBoost 
    does not require normality."""
    for col in residuals.columns:
        plt.figure(figsize=(10, 6))
        probplot(residuals[col], dist="norm", plot=plt)
        plt.title(f"{col} residual Q-Q plot")
        plt.show()
