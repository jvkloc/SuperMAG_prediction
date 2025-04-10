"""Plotting functions."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import probplot

from constants import FEATURES, TARGETS


def plot_features_time_series(X_test: DataFrame, features: dict = FEATURES) -> None:
    """Plots feature time series in three figures, each with three subplots."""
    for group_name, features in features.items():
        fig: ndarray; ax: Figure
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes: ndarray = ax.flatten()
        
        # Colors for features.
        colors = ["blue", "orange", "green"]
        
        # Plot each feature in its subplot.
        for i, (feature, color) in enumerate(zip(features, colors)):
            ax: ndarray = axes[i]
            ax.plot(X_test.index, X_test[feature], color=color, label=feature, alpha=0.7)
            ax.set_ylabel(f"{feature} value")
            ax.set_title(f"{feature}")
            ax.legend()
            ax.grid(True)
        
        # Set x-label only for bottom subplot.
        axes[2].set_xlabel("Time")
        # Title above all subplots
        fig.suptitle(f"{group_name} time series", y=1.02)
        # Display.
        plt.tight_layout()
        plt.show()


def plot_evals_result(evals_result: dict, best_iteration: int) -> None:
    """Plot training and test RMSE as a scatterplot."""
    train_rmse: list[float] = evals_result['train']['rmse']
    test_rmse: list[float] = evals_result['test']['rmse']
    iterations = range(len(train_rmse))  # X-axis: iteration numbers
    plt.figure(figsize=(10, 6))
    plt.scatter(iterations, train_rmse, label="Training RMSE", color="blue")
    plt.scatter(iterations, test_rmse, label="Test RMSE", color="orange")
    # Set vertical line to best iteration number on x-axis.
    plt.axvline(
        x=best_iteration,
        color="green",
        linestyle="--",
        label=f"Best iteration ({best_iteration})",
    )
    # Set axes' labels. 
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    # Set plot title and legend.
    plt.title("Training vs. test RMSE")
    plt.legend()
    # Plot image.
    plt.grid(True)
    plt.show()


def prediction_scatter_plot(
    y_pred: DataFrame,
    y_test: DataFrame,
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
    y_pred: DataFrame,
    y_test: DataFrame,
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
    y_pred: DataFrame,
    y_test: DataFrame,
    targets: list[str] = TARGETS,
) -> None:
    """Plots prediction vs. true values as time series."""
    _: ndarray; axs: Figure
    _, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes: ndarray = axs.flatten()
    # Colors for true (solid) and predicted (dashed)
    true_color: str = "blue"
    pred_color: str = "orange"
    # Plot each target in its subplot.
    for i, target in enumerate(targets[1:]):
        ax: ndarray = axes[i]
        ax.plot(
            y_test.index,
            y_test[target],
            color=true_color,
            label=f"True {target}",
            alpha=0.7,
        )
        ax.plot(
            y_test.index,
            y_pred[target],
            color=pred_color,
            linestyle="--",
            label=f"Pred {target}",
            alpha=0.7,
        )
        ax.set_ylabel(f"{target} (nT)")
        ax.set_title(f"{target}: True vs. predicted")
        ax.legend()
        ax.grid(True)
    # Set x-label only for bottom row.
    axes[2].set_xlabel("Time")
    axes[3].set_xlabel("Time")
    # Adjust layout and display,
    plt.tight_layout()
    plt.show()


def global_prediction_time_series(
    y_pred: DataFrame,
    y_test: DataFrame,
    target: str = "SMR",
) -> None:
    """Plots prediction vs. true values as time series."""
    plt.figure(figsize=(10, 6))
    # Colors for true (solid) and predicted (dashed).
    true_color: str = "blue"
    pred_color: str = "orange"
    # Plot the target.
    plt.plot(
        y_test.index,
        y_test[target],
        color=true_color,
        label=f"True {target}",
        alpha=0.7,
    )
    plt.plot(
        y_test.index,
        y_pred[target],
        color=pred_color,
        linestyle="--",
        label=f"Pred {target}",
        alpha=0.7,
    )
    # Set labels and title.
    plt.xlabel("Time")
    plt.ylabel(f"{target} (nT)")
    plt.title(f"{target}: True vs. predicted")
    # Add legend and grid.
    plt.legend()
    plt.grid(True)
    # Adjust layout and display,
    plt.tight_layout()
    plt.show()


def plot_smr_histograms(data: DataFrame, targets: list[str] = TARGETS) -> None:
    """Plots density histograms of SMR and SMR MLT values from all data."""
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
        axes[i].set_ylabel("Density")
        axes[i].grid(True)
    
    # Hide the unused subplot (bottom right).
    axes[5].axis("off")
    # Display the images.
    plt.tight_layout()
    plt.show()


def residual_density_histogram(residuals: DataFrame) -> None:
    """Plots residuals in a normalized histogram."""
    residuals.hist(bins=100, edgecolor="black", figsize=(10, 6), density=True)
    plt.title("Residual histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()


def residuals_vs_predicted_scatter(
        residuals: DataFrame, y_pred: DataFrame
) -> None:
    """Scatter plot of residuals vs. predicted values. Residuals should be
    randomly scattered around zero line."""
    for col in residuals.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred[col], residuals[col], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle="--")
        plt.title(f"Residuals vs. predicted {col}")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.show()


def q_q_plot(residuals: DataFrame) -> None:
    """Q-Q plot for checking normality of the residuals. Note that XGBoost 
    does not require normality."""
    for col in residuals.columns:
        plt.figure(figsize=(10, 6))
        probplot(residuals[col], dist="norm", plot=plt)
        plt.title(f"{col} residual Q-Q plot")
        plt.show()
