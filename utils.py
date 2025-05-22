"""Utility functions."""

from argparse import ArgumentParser
from datetime import datetime, timedelta
from os import environ
from os.path import abspath, dirname, join
from time import perf_counter

from numpy import ndarray
from pandas import DataFrame
from polars import (
    col, 
    concat, 
    DataFrame as PolarsFrame,
    Float32,
    LazyFrame, 
    lit,
    Series,
    when,
)

from constants import TARGETS, DATA_PATH, MODEL_PATH, VALIDATION_WEEKS


def add_arguments(
    parser: ArgumentParser, data: str = DATA_PATH, model: str = MODEL_PATH
) -> None:
    """Command line arguments."""
    parser.add_argument(
        "--newdata",
        action="store_true",
        help="Download/check and preprocess data and train and save a model."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model using preprocessed data from --datapath."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=data,
        help="Load/save preprocessed training data from/to path."
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default=model,
        help="Load/save model from/to path."
    )


def get_data_gaps(data: PolarsFrame, limit: int = 300) -> dict:
    """Returns a dictionary of gaps longer than limit seconds."""
    gaps: PolarsFrame = (
        data
        .select(
            col("index").diff().dt.total_seconds().alias("gap_seconds"),
            col("index")
        )
        .filter(col("gap_seconds") > limit)
    )
    return {
        'limit': limit,
        'longer_gaps': gaps.get_column("gap_seconds"),
        'index_values': gaps.get_column("index"),
    }


def get_features(X: PolarsFrame) -> list[str]:
    """Returns a list of model features."""
    return [col for col in X.columns if col not in TARGETS and col != "index"]


def get_initial_features(data: LazyFrame, SuperMAG: LazyFrame) -> list[str]:
    """Returns a list of model features."""
    return [
        col for col in data.collect_schema().names() 
        if col not in SuperMAG.collect_schema().names()
    ]


def get_extreme_weeks(
    target: str, y: PolarsFrame, weeks: set[str], n: int = 3
) -> list[datetime]:
    """Helper function for get_rolling_basis_cv_folds(). Returns three unique 
    extreme weeks for each target."""

    selected_weeks: list[datetime] = []
    
    # Quantile-based extreme weeks
    quantiles: PolarsFrame = y.select([
        col(target).quantile(0.001).alias("q_low"),
        col(target).quantile(0.999).alias("q_high")
    ]).to_dict()
    q_low: float = quantiles["q_low"][0]
    q_high: float = quantiles["q_high"][0]
    
    extreme_data: PolarsFrame = y.filter(
        (col(target) <= q_low) | (col(target) >= q_high)
    ).group_by_dynamic("index", every="1w").agg(
        col("index").min().alias("week_start")
    )
    
    # Get unique weeks.
    available_extreme: PolarsFrame = extreme_data.filter(
        ~col("week_start").cast(str).is_in(weeks)
    )
    if available_extreme.height > 0:
        sample_size: int = min(n, available_extreme.height)
        selected_weeks.extend(
            available_extreme.sample(
                n=sample_size, with_replacement=False, shuffle=True
            )["week_start"].to_list()
        )
    
    if len(selected_weeks) < n: # Fewer than n weeks.
        # Use deviation from mean to get extreme value weeks.
        remaining: int = n - len(selected_weeks)
        mean_val: float = y.select(col(target).mean())[0, 0]
        all_weeks: PolarsFrame = y.group_by_dynamic("index", every="1w").agg(
            col("index").min().alias("week_start"),
            col(target).apply(lambda x: abs(x - mean_val).mean()).alias("deviation")
        ).sort("deviation", descending=True)
        
        # Get unique weeks.
        used_weeks_str: set[str] = {str(w) for w in weeks}
        selected_weeks_str: set[str] = {str(w) for w in selected_weeks}
        available_weeks: PolarsFrame = all_weeks.filter(
            ~col("week_start").cast(str).is_in(used_weeks_str | selected_weeks_str)
        )
        
        if available_weeks.height > 0: # Append the new weeks.
            selected_weeks.extend(
                available_weeks.head(remaining)["week_start"].to_list()
            )
    
    # Ensure exactly n weeks.
    return selected_weeks[:n]


def get_rolling_basis_cv_folds(
    X_data: PolarsFrame,
    y_data: PolarsFrame,
    val_weeks: list[str] = VALIDATION_WEEKS,
    targets: list[str] = TARGETS
) -> tuple[list[tuple], PolarsFrame, PolarsFrame, dict]:
    """Returns training-validation folds for all val_weeks, three extreme 
    target values week folds for each target, the whole training data, the 
    validation data of the last week, and a dictionary of weeks with extreme 
    target values per target."""
    
    X: PolarsFrame = X_data.sort("index")
    y: PolarsFrame = y_data.sort("index")
    folds: list[tuple[PolarsFrame, PolarsFrame, PolarsFrame, PolarsFrame]] = []
    xtreme_weeks: dict[str, list[str]] = {target: [] for target in targets}
    extreme_weeks: set[str] = set()

    
    # Three extreme values folds for each target.
    for target in targets:
        weeks: list[datetime] = get_extreme_weeks(target, y, extreme_weeks)
        for i, week_start in enumerate(weeks, 1):
            week_start_date: str = week_start.date().strftime("%Y-%m-%d")
            if week_start_date not in extreme_weeks:
                extreme_weeks.add(week_start_date)
                xtreme_weeks[target].append(week_start_date)
                week_end_dt: datetime = week_start + timedelta(days=7)
                
                X_train: PolarsFrame = X.filter(col("index") < week_start)
                y_train: PolarsFrame = y.filter(col("index") < week_start)
                X_test: PolarsFrame = X.filter(
                    (col("index") >= week_start) & (col("index") < week_end_dt)
                )
                y_test: PolarsFrame = y.filter(
                    (col("index") >= week_start) & (col("index") < week_end_dt)
                )
                
                folds.append((X_train, y_train, X_test, y_test))
                print(f"Extreme Fold {i} for {target}: Train until {week_start}, Val {week_start} to {week_end_dt}, Val rows: {y_test.height}")
            else:
                print(f"Skipped duplicate week {week_start_date} for {target}")

    # Regular folds.
    for i, week_start in enumerate(val_weeks, 1):
        week_start_dt: datetime = datetime.strptime(week_start, "%Y-%m-%d")
        week_end_dt: datetime = week_start_dt + timedelta(days=7)
        
        X_train: PolarsFrame = X.filter(col("index") < week_start_dt)
        y_train: PolarsFrame = y.filter(col("index") < week_start_dt)
        X_test: PolarsFrame = X.filter(
            (col("index") >= week_start_dt) & (col("index") < week_end_dt)
        )
        y_test: PolarsFrame = y.filter(
            (col("index") >= week_start_dt) & (col("index") < week_end_dt)
        )
        
        folds.append((X_train, y_train, X_test, y_test))
        print(f"Fold {i}: Train until {week_start}, Val {week_start} to {week_end_dt}, Val rows: {y_test.height}")
       
    # Training data.
    last_start: datetime = datetime.strptime(val_weeks[-2], "%Y-%m-%d")
    last_end: datetime = last_start + timedelta(days=7)
    train_X: PolarsFrame = X.filter(col("index") < last_start)
    train_y: PolarsFrame = y.filter(col("index") < last_start)
    training: PolarsFrame = train_y.join(train_X, how="inner", on="index").drop("index")

    # Validation data of the last regular fold.
    val_X: PolarsFrame = X.filter(
        (col("index") >= last_start) & (col("index") < last_end)
    )
    val_y: PolarsFrame = y.filter(
        (col("index") >= last_start) & (col("index") < last_end)
    )
    validation: PolarsFrame = val_y.join(val_X, how="inner", on="index").drop("index")

    return training, validation, folds, xtreme_weeks


def get_rolling_basis_cv_folds(
    X_data: PolarsFrame,
    y_data: PolarsFrame,
    val_weeks: list[str] = VALIDATION_WEEKS,
    targets: list[str] = TARGETS
) -> tuple[list[tuple], PolarsFrame, PolarsFrame, dict]:
    """Returns training-validation folds for all val_weeks, three extreme 
    target values week folds for each target, the whole training data, the 
    validation data of the last week, and a dictionary of weeks with extreme 
    target values per target."""
    
    X: PolarsFrame = X_data.sort("index")
    y: PolarsFrame = y_data.sort("index")
    folds: list[tuple[PolarsFrame, PolarsFrame, PolarsFrame, PolarsFrame]] = []
    xtreme_weeks: dict[str, list[str]] = {target: [] for target in targets}

    # Three extreme folds for each target.
    extreme_weeks: set = set()
    for target in targets:
        all_weeks: PolarsFrame = y.filter(
            col(target).is_in(col(target).quantile(0.001).append(col(target).quantile(0.999)))
        ).group_by_dynamic("index", every="1w").agg(col("index").min().alias("week_start"))
        # Randomly sample three weeks.
        weeks: PolarsFrame = all_weeks.sample(
            n=3, with_replacement=False, shuffle=True
        )["week_start"].to_list()
        for i, week_start in enumerate(weeks, 1):
            week_start_date: str = week_start.date().strftime("%Y-%m-%d")
            if week_start_date not in extreme_weeks:
                extreme_weeks.add(week_start_date)
                xtreme_weeks[target].append(week_start_date)
                week_end_dt: datetime = week_start + timedelta(days=7)
                
                X_train: PolarsFrame = X.filter(col("index") < week_start)
                y_train: PolarsFrame = y.filter(col("index") < week_start)
                X_test: PolarsFrame = X.filter(
                    (col("index") >= week_start) & (col("index") < week_end_dt)
                )
                y_test: PolarsFrame = y.filter(
                    (col("index") >= week_start) & (col("index") < week_end_dt)
                )
                
                folds.append((X_train, y_train, X_test, y_test))
            print(f"Extreme Fold {i}: Train until {week_start}, Val {week_start} to {week_end_dt}, Val rows: {y_test.height}")

    # Regular folds.
    for i, week_start in enumerate(val_weeks, 1):
        week_start_dt: datetime = datetime.strptime(week_start, "%Y-%m-%d")
        week_end_dt: datetime = week_start_dt + timedelta(days=7)
        
        X_train: PolarsFrame = X.filter(col("index") < week_start_dt)
        y_train: PolarsFrame = y.filter(col("index") < week_start_dt)
        X_test: PolarsFrame = X.filter(
            (col("index") >= week_start_dt) & (col("index") < week_end_dt)
        )
        y_test: PolarsFrame = y.filter(
            (col("index") >= week_start_dt) & (col("index") < week_end_dt)
        )
        
        folds.append((X_train, y_train, X_test, y_test))
        print(f"Fold {i}: Train until {week_start}, Val {week_start} to {week_end_dt}, Val rows: {y_test.height}")
       
    # Training data.
    last_start = datetime.strptime(val_weeks[- 2], "%Y-%m-%d")
    last_end: datetime = last_start + timedelta(days=7)
    train_X: PolarsFrame = X.filter(col("index") < last_start)
    train_y: PolarsFrame = y.filter(col("index") < last_start)
    training: PolarsFrame = train_y.join(train_X, how="inner", on="index").drop("index")

    # Validation data of the last regular fold.
    val_X: PolarsFrame = X.filter(
        (col("index") >= last_start) & (col("index") < last_end)
    )
    val_y: PolarsFrame = y.filter(
        (col("index") >= last_start) & (col("index") < last_end)
    )
    validation: PolarsFrame = val_y.join(val_X, how="inner", on="index").drop("index")

    return training, validation, folds, xtreme_weeks


def get_train_label(y_train: PolarsFrame, target: str) -> ndarray:
    """Returns the 'label' param for XGBoost training."""
    label: PolarsFrame = y_train.select(target)
    label: PolarsFrame = label.drop_nulls()
    return label.to_numpy().flatten()


def get_test_label(y_test: PolarsFrame, target: str) -> ndarray:
    """Returns the 'label' param for XGBoost validation."""
    label: PolarsFrame = y_test.select(target)
    label: PolarsFrame = label.drop_nulls()
    return label.to_numpy().flatten()


def get_multitarget_label(y_data: PolarsFrame) -> ndarray:
    """Returns 'label' param for multi-target XGBoost training."""
    label: PolarsFrame = y_data.drop("index")
    label: PolarsFrame = label.drop_nulls()
    return label.to_numpy()


def print_data_gaps(gap_info: dict) -> None:
    """Prints data gaps from the given dictionary."""
    print("")
    for data in gap_info:
        longer_gaps: Series = gap_info[data]['longer_gaps']
        index_values: Series = gap_info[data]['index_values']
        limit: int = gap_info[data]['limit']
        if len(longer_gaps) > 0:
            print(f"Gaps over {limit} seconds in the {data} data:")
            for gap_size, idx in zip(longer_gaps, index_values):
                try:
                    pos: int = index_values.eq(idx).arg_true()[0]
                    if pos > 0:
                        start = index_values[pos - 1]
                        print(f"Gap of {gap_size} seconds between {start} and {idx}")
                    else:
                        print(f"Gap of {gap_size} seconds starts at index {idx}")
                except IndexError:
                    continue
        else:
            print(f"No gaps over {limit} seconds found from {data} data.")


def print_fold_datetimes(
    i: int, y: PolarsFrame, train_end: int, test_start: int, test_end: int
)-> None:
    """Prints training fold data datetimes."""
    datetimes: Series = y["index"]
    y_first: int = datetimes[0]
    y_train_last: int = datetimes[train_end - 1] 
    y_test_first: int = datetimes[test_start]
    y_test_last: int = datetimes[test_end - 1]
    print(f"Fold {i}: Train {y_first} to {y_train_last} ({train_end}), "
            f"Test {y_test_first} to {y_test_last} ({test_end - test_start})")


def print_target_stats(
    y_train: DataFrame,
    y_test: DataFrame,
    y_val: DataFrame,
    targets: list[str] = TARGETS
) -> None:
    """Prints statistics of the targets."""
    for target in targets:
        print(f"Train {target} mean: {y_train[target].mean():.2f}, std: {y_train[target].std():.2f}, max: {y_train[target].max():.2f}")
        print(f"Val {target} mean: {y_val[target].mean():.2f}, std: {y_val[target].std():.2f}, max: {y_val[target].max():.2f}")
        print(f"Test {target} mean: {y_test[target].mean():.2f}, std: {y_test[target].std():.2f}, max: {y_test[target].max():.2f}")


def set_environment_variable(data_dir: str = "spedas_data") -> None:
    """Sets SPEDAS_DATA_DIR environment variable."""
    download_dir: str = join(dirname(abspath(__file__)), data_dir)
    environ["SPEDAS_DATA_DIR"] = download_dir
    print(f"environ['SPEDAS_DATA_DIR'] = {download_dir}\n")


def split_data(
    data: LazyFrame, 
    features: list[str], 
    targets: list[str] = TARGETS,
) -> tuple[PolarsFrame, PolarsFrame, dict]:
    """Returns the processed data as X (features) and y (targets). Also, 
    returns possible data gaps in X and y in a dictionary."""
    pf: PolarsFrame = (
        data
        .unique(subset=["index"], keep="first")
        .sort("index")
    ).collect()
    
    X: PolarsFrame = pf.select(features + ["index"]).with_columns(
        [col(f).cast(Float32) for f in features]
    )
    
    y: PolarsFrame = pf.select(targets + ["index"]).with_columns(
        [col(t).cast(Float32) for t in targets]
    )
    
    gaps: dict[str, dict] = {'X': get_data_gaps(X), 'y': get_data_gaps(y)}
    
    return X, y, gaps


def stop_timing(script_start: float) -> None:
    """Stops timing the script and prints the elapsed time in minutes."""
    script_end: float = perf_counter()
    script_time: float = (script_end - script_start) / 60
    print(f"\nThe script finished in {script_time:.2f} minutes.")


def print_results(y_test: DataFrame, y_pred: DataFrame, targets: list[str] = TARGETS) -> None:
    """Prints prediction results."""
    for i in range(0, len(y_test), 100):
        for j, target in enumerate(targets):
            print(f"{y_test.index[i]} {target} True: {y_test.iloc[i, j]:.2f}, Pred: {y_pred.iloc[i, j]:.2f}")


def unsplit_data(
    X_train: PolarsFrame, y_train: PolarsFrame, X_test: PolarsFrame, y_test: PolarsFrame
) -> tuple[PolarsFrame, PolarsFrame]:
    """Recombines a train-test split into one DataFrame and returns it."""
    # Combine the training data.
    train: PolarsFrame = X_train.join(y_train, how="inner", on="index")
    # Combine the test data.
    test: PolarsFrame = X_test.join(y_test, how="inner", on="index")
    # Combine train and test.
    full_data: PolarsFrame = concat([train, test], how="vertical")
    # Return the recombined training data and full data.
    return train, full_data


def get_weights(full_data: PolarsFrame, xtreme_weeks: dict) -> PolarsFrame:
    """Returns 'sample_weights' param for XGBoost model."""
    sample_weights: PolarsFrame = full_data.with_columns(weight=lit(1.0))
    
    for target, weeks in xtreme_weeks.items():
        week_starts: list[datetime] = [
            datetime.strptime(w, "%Y-%m-%d") for w in weeks
        ]
        for week_start in week_starts:
            week_end: datetime = week_start + timedelta(days=7)
            sample_weights: PolarsFrame = sample_weights.with_columns(
                weight=when(
                    (col("index") >= week_start) & (col("index") < week_end)
                ).then(5.0).otherwise(col("weight"))
            )
    
    return sample_weights


def add_interaction_features(
    full_data: PolarsFrame,
    importances: dict,
    features: list[str],
    targets: list[str] = TARGETS
) -> PolarsFrame:
    """Adds interaction features to 'full_data' based on top two most 
    important features of the extreme values weeks."""
    
    print("\nInteraction variables for final model:")
    for target in targets:
        top_features: list[str] = [
            feat for imp in importances[target] for feat, _ in imp
        ]
        top_features = list(dict.fromkeys(top_features))[:2]                
        # Log(x+1) transform for the feature.
        full_data: PolarsFrame = full_data.with_columns(
            ((col(top_features[0]) * col(top_features[1])).abs().log1p() * 
             (col(top_features[0]) * col(top_features[1])).sign()
            ).alias(f"{target}_interaction")
        )
        features.append(f"{target}_interaction")
        print(f"{target}: {top_features[0]} * {top_features[1]}")

    return full_data
