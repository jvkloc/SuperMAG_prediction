"""Functions for evaluating the model."""

from numpy import ndarray, sqrt
from polars import col, DataFrame as PolarsFrame, Series
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import Booster, DMatrix

from constants import TARGETS


def compute_average_metrics(
    all_metrics: list[dict], targets: list[str] = TARGETS
) -> dict[str, dict]:
    """Returns the average metrics in a dictionary across all training folds."""
    # Dictionary for computed metrics.
    computed: dict[str, dict] = {target: {} for target in targets}

    for target in computed:
        # Filter metrics for the specific target.
        target_metrics = [m for m in all_metrics if m['target'] == target]
        
        # Extract metrics for the target across all relevant folds.
        metrics: PolarsFrame = PolarsFrame({
            'MSE': [m[f'{target}_MSE'] for m in target_metrics],
            'R2': [m[f'{target}_R2'] for m in target_metrics],
            'MAE': [m[f'{target}_MAE'] for m in target_metrics],
            'RMSE': [m[f'{target}_RMSE'] for m in target_metrics],
            'training_RMSE': [
                m.get(f'{target}_training_RMSE', m.get('training_RMSE')) 
                for m in target_metrics
            ]
        })

        # Compute mean and std for each metric.
        agg_stats: PolarsFrame = metrics.select([
            col("MSE").mean().alias("MSE_mean"),
            col("MSE").std(ddof=1).alias("MSE_std"),
            col("R2").mean().alias("R2_mean"),
            col("R2").std(ddof=1).alias("R2_std"),
            col("MAE").mean().alias("MAE_mean"),
            col("MAE").std(ddof=1).alias("MAE_std"),
            col("RMSE").mean().alias("RMSE_mean"),
            col("RMSE").std(ddof=1).alias("RMSE_std"),
            col("training_RMSE").mean().alias("training_RMSE_mean"),
            col("training_RMSE").std(ddof=1).alias("training_RMSE_std")
        ]).to_dicts()[0]

        # Assign computed stats to the result dictionary.
        keys: list[str] = ["R2", "MSE", "RMSE", "MAE"]
        for key in keys:
            computed[target][key] = agg_stats[f"{key}_mean"]
            computed[target][f"{key}_std"] = agg_stats[f"{key}_std"]

        # Include training_RMSE if available.
        if agg_stats["training_RMSE_mean"] is not None:
            computed[target]["training_RMSE"] = agg_stats["training_RMSE_mean"]
            computed[target]["training_RMSE_std"] = agg_stats["training_RMSE_std"]

    return computed


def print_metrics(metrics: dict[str, dict], targets: list[str] = TARGETS) -> None:
    """Prints the average model metrics from the dictionary argument."""
    for target in targets:
        print(f"\nAverage metrics for {target}:")
        print(f"R²: {metrics[target]['R2']:.4f} (±{metrics[target]['R2_std']:.4f})")
        print(f"MSE: {metrics[target]['MSE']:.2f} (±{metrics[target]['MSE_std']:.2f})")
        print(f"RMSE: {metrics[target]['RMSE']:.2f} (±{metrics[target]['RMSE_std']:.2f})")
        print(f"MAE: {metrics[target]['MAE']:.2f} (±{metrics[target]['MAE_std']:.2f})")
        if 'training_RMSE' in metrics[target]:
            print(f"Training RMSE: {metrics[target]['training_RMSE']:.2f} (±{metrics[target]['training_RMSE_std']:.2f})")


def print_feature_importance(model: Booster, features: list[str]) -> None:
    """Prints feature importances in descending order."""
    # Get the importances.
    try:
        importances: dict = model.get_score(importance_type="gain")
    except Exception as e:
        print(f"model.get_score(): {e}")
    if not importances:
        print("\nNo feature importances available.")
        return
    total_splits: int = sum(importances.values())
    
    # Set the importances to a list.
    sorted: list[tuple[str, float]] = [
        (feature, importances.get(feature, 0) / total_splits)
        for feature in features
    ]

    # Sort the list by importance, the most important first.
    sorted.sort(key=lambda x: x[1], reverse=True)
    
    # Print the sorted importances.
    print("\nFeature importances:")
    for feature, importance in sorted:
        print(f"{feature}: {importance:.4f}")


def print_prediction_metrics(
    y_test: PolarsFrame,
    y_pred: PolarsFrame,
    targets: list[str] = TARGETS
) -> None:
    """Prints prediction metrics."""
    for target in targets:
        if target in y_test.columns and target in y_pred.columns:
            print(f"\nMetrics for {target}:")
            # Get target columns as NumPy arrays for metrics.
            true_val: ndarray = y_test.select(target).to_numpy().flatten()
            prediction: ndarray = y_pred.select(target).to_numpy().flatten()
            mse: float = mean_squared_error(true_val, prediction)

            # Calculate and print metrics
            print(f"R²: {r2_score(true_val, prediction):.4f}")
            print(f"MSE: {mse:.2f}")
            print(f"MAE: {mean_absolute_error(true_val, prediction):.2f}")
            print(f"RMSE: {sqrt(mse):.2f}")
        else:
            print(f"Warning: Target {target} not found in y_test or y_pred")


def print_unique_values(
    true: PolarsFrame,
    pred: PolarsFrame,
    targets: list[str] = TARGETS
) -> None:
    """Prints the number of unique true and predicted values."""
    print("Number of unique values per target:")
    print("Target | True | Predicted")
    print("-" * 40)
    for t in targets:
        unique_true: int = true[t].n_unique()
        unique_pred: int = pred[t].n_unique()
        print(f"{t:<6} | {unique_true:<11} | {unique_pred}")


def print_extreme_predictions(
    true_df: PolarsFrame,
    pred_df: PolarsFrame,
    percentile: float = 0.01,
    max_display: int = 50,
    targets: list[str] = TARGETS
) -> None:
    """Prints a table comparing model performance for the top and bottom 
    'percentile' of unique true values. Adjusting 'max_display' prints less or 
    more of unique true values and their predictions."""     
    
    for t in targets:
        print(f"\nColumn: {t}")
        print("-" * 50)
        
        ts: Series = true_df[t]
        print(f"{t} validation stats: min={ts.min():.2f}, max={ts.max():.2f}, count={ts.len()}")
        combined: PolarsFrame = true_df.select(col(t).alias("true")).with_columns(pred=pred_df[t])
        low_threshold: float = ts.quantile(percentile)
        high_threshold: float = ts.quantile(1 - percentile)
        
        values: Series = combined.filter(
            (col("true") <= low_threshold) | (col("true") >= high_threshold)
        )["true"].unique().sort()
        n_unique: int = combined["true"].unique().len()
        values: Series = values.head(max_display // 2).extend(
            values.tail(max_display // 2)
        ).unique().sort()
        
        print(f"Total unique true values: {n_unique}")
        print(f"Selected {len(values)} extreme true values (below {percentile*100}th/above {(1-percentile)*100}th percentile, capped at {max_display})")
        print("True Value | Mean Pred | RMSE  | MAE   | Count")
        print("-" * 50)
        
        for true_val in values:
            group: PolarsFrame = combined.filter(col("true") == true_val)
            count: int = group.height
            mean_pred: float = group["pred"].mean()
            rmse: float = ((group["true"] - group["pred"]) ** 2).mean() ** 0.5
            mae: float = (group["true"] - group["pred"]).abs().mean()
            
            print(f"{true_val:<10.2f} | {mean_pred:<9.2f} | {rmse:<5.2f} | {mae:<5.2f} | {count}")
        
        low_rmse: float = combined.filter(col("true") <= low_threshold).select(
            ((col("true") - col("pred")) ** 2).mean() ** 0.5
        ).item()
        high_rmse: float = combined.filter(col("true") >= high_threshold).select(
            ((col("true") - col("pred")) ** 2).mean() ** 0.5
        ).item()

        print(f"\nLow extreme values average RMSE: {low_rmse:.2f}")
        print(f"High extreme values average RMSE: {high_rmse:.2f}")
        
        low_mae: float = combined.filter(col("true") <= low_threshold).select(
            (col("true") - col("pred")).abs().mean()
        ).item()
        high_mae: float = combined.filter(col("true") >= high_threshold).select(
            (col("true") - col("pred")).abs().mean()
        ).item()

        print(f"Low extreme values average MAE: {low_mae:.2f}")
        print(f"High extreme values average MAE: {high_mae:.2f}")


def evaluate_extreme_performance(
    model: Booster, 
    validation: PolarsFrame, 
    targets: list[str] = TARGETS, 
    quantile: float = 0.001
) -> dict[str, float]:
    """Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) 
    for each target on extreme values (top and bottom quantiles). Prints MAE 
    and RMSE for each target and returns a dictionary with both metrics."""
    
    results: dict[str, float] = {}
    for target in targets:
        extreme_val: PolarsFrame = validation.filter(
            col(target).is_in(
                col(target).quantile(quantile).append(
                    col(target).quantile(1 - quantile)
                )
            )
        )
        X_val = DMatrix(
            extreme_val.drop(targets).to_numpy(),
            feature_names=extreme_val.drop(targets).columns
        )
        y_true: ndarray = extreme_val[target].to_numpy()
        y_pred: ndarray = model.predict(X_val)[:, targets.index(target)]
        mae: float = mean_absolute_error(y_true, y_pred)
        rmse: float = sqrt(mean_squared_error(y_true, y_pred))
        results[target] = mae
        print(f"Extreme values validation for {target}: MAE {mae:.4f}, RMSE {rmse:.4f}")
    return results
