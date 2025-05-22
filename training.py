"""Training loop and training loop utility functions."""

from gc import collect

from numpy import ndarray, sqrt
from pandas import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from polars import (
    col, 
    concat,  
    DataFrame as PolarsFrame, 
    Expr,
    Float32,
    LazyFrame,
    len as p_len,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import Booster, DMatrix, train

from constants import (
    TARGETS,
    THRESHOLD,
    XGB_PARAMS,
    SINGLE_TARGET_PARAMS,
    N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    MODEL_PATH,
)
from data_utils import save_processed_data
from utils import (
    add_interaction_features,
    get_features,
    get_test_label,
    get_train_label,
    get_multitarget_label,
    get_weights,
    unsplit_data,
)


def add_lagged_features(
    data: LazyFrame | PolarsFrame, targets: list[str] = TARGETS
) -> LazyFrame | PolarsFrame:
    """Adds lagged SMR values to the data and removes rows with NaN values. 
    Returns the same type of frame as the argument 'data' is."""
    data: LazyFrame | PolarsFrame = data.sort("index")
    exprs: list[Expr] = []
    for target in targets:
        exprs.extend([
            col(target).shift(1).alias(f"{target}_lag1"),
            col(target).shift(2).alias(f"{target}_lag2"),
            col(target).shift(3).alias(f"{target}_lag3"),
            col(target).shift(5).alias(f"{target}_lag5")
        ])
    data: LazyFrame | PolarsFrame = data.with_columns(exprs)
    cleaned: LazyFrame | PolarsFrame = data.drop_nulls()
    return cleaned


def get_lagged_data(
    X_train: PolarsFrame,
    y_train: PolarsFrame,
    X_test: PolarsFrame,
    y_test: PolarsFrame,
    targets: list[str] = TARGETS,
) -> tuple[PolarsFrame, PolarsFrame, PolarsFrame, PolarsFrame]:
    """Returns the train-test split with lagged features."""
    # Get train, test and full data combinations for lagging.
    train: PolarsFrame = X_train.join(y_train, how="inner", on="index")
    test: PolarsFrame = X_test.join(y_test, how="inner", on="index")
    full_data: PolarsFrame = concat([train, test], how="vertical")

    # Get lagged training data.
    lagged_train: PolarsFrame = add_lagged_features(train)
    
    # Get lagged full data.
    lagged_full: PolarsFrame = add_lagged_features(full_data)
    
    # Get lagged test data start index.
    train_len: int = lagged_train.select(p_len()).item()
    # Get lagged test data.
    lagged_test: PolarsFrame = lagged_full.slice(train_len)

    # Split the data back into X and y.
    lagged_X_train: PolarsFrame = lagged_train.drop(targets)
    lagged_y_train: PolarsFrame = lagged_train.select(targets + ["index"])
    lagged_X_test: PolarsFrame = lagged_test.drop(targets)
    lagged_y_test: PolarsFrame = lagged_test.select(targets + ["index"])
    
    return lagged_X_train, lagged_y_train, lagged_X_test, lagged_y_test


def get_prediction_dataframe(
    pred: ndarray, y_test: PolarsFrame, target: str | None = None
) -> PolarsFrame:
    """Returns Polars dataframe with the given predictions and index column."""
    targets: list[str] = [col for col in y_test.columns if col != "index"]
    index_col: ndarray = y_test["index"].to_numpy()
    
    if pred.ndim == 1:  # Single-target prediction.
        
        return PolarsFrame(
            {"index": index_col, target: pred},
            schema={"index": y_test["index"].dtype, target: Float32}
        )
    
    else:  # Multi-target prediction.
        
        return PolarsFrame(
            {"index": index_col, **{target: pred[:, i] for i, target in enumerate(targets)}},
            schema={"index": y_test["index"].dtype, **{target: Float32 for target in targets}}
        )


def get_prediction_metrics(
    y_test: PolarsFrame,
    y_pred: PolarsFrame,
    model: Booster,
    d_train: DMatrix,
    y_train: PolarsFrame,
    target: str | None = None,
    targets: list[str] = TARGETS
) -> dict[str, float]:
    """Returns prediction metrics (MSE, R2, MAE, RMSE, training_RMSE) for a single target
    or each target in a multi-output model."""
    # Determine if single-target or multi-target.
    single: bool = len(y_test.columns) == 1
    metrics: dict[str, float] = {}
    
    if single: # Single-target.
        
        test_target: ndarray = y_test[target].to_numpy().flatten()
        pred_target: ndarray = y_pred[target].to_numpy().flatten()
        
        # Metrics.
        mse: float = mean_squared_error(test_target, pred_target)
        rmse: float = sqrt(mse)
        metrics.update({
            f'{target}_MSE': mse,
            f'{target}_RMSE': rmse,
            f'{target}_R2': r2_score(test_target, pred_target),
            f'{target}_MAE': mean_absolute_error(test_target, pred_target)
        })
        
        # Training RMSE.
        train_pred: ndarray = model.predict(d_train).flatten()
        train_true: ndarray = y_train[target].to_numpy().flatten()
        metrics['training_RMSE'] = sqrt(mean_squared_error(train_true, train_pred))
    
    else: # Multi-target case: compute metrics for the specified target.
        
        train_pred: ndarray = model.predict(d_train)
        for i, target in enumerate(targets):
            test_target: ndarray = y_test[target].to_numpy().flatten()
            pred_target: ndarray = y_pred[target].to_numpy().flatten()
            
            mse: float = mean_squared_error(test_target, pred_target)
            rmse: float = sqrt(mse)
            metrics.update({
                f'{target}_MSE': mse,
                f'{target}_RMSE': rmse,
                f'{target}_R2': r2_score(test_target, pred_target),
                f'{target}_MAE': mean_absolute_error(test_target, pred_target)
            })
            
            train_true: ndarray = y_train[target].to_numpy().flatten()
            train_pred_target: ndarray = train_pred[:, i]
            metrics[f'{target}_training_RMSE'] = sqrt(
                mean_squared_error(train_true, train_pred_target)
            )
    
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


def train_xgboost(
    X_train: PolarsFrame,
    X_test: PolarsFrame,
    train_label: ndarray,
    test_label: PolarsFrame,
    params: dict,
    estimators: int = N_ESTIMATORS,
    early_stop: int = EARLY_STOPPING_ROUNDS,
) -> tuple[Booster, DMatrix, DMatrix, dict]:
    """Initializes and trains an XGBoost model. Returns the model, train and 
    test data and train and test root mean square error values."""
    
    # Drop index columns.
    Xtrain: PolarsFrame = X_train.drop("index")
    Xtest: PolarsFrame = X_test.drop("index")
    # Drop missing values.
    Xtrain: PolarsFrame = Xtrain.drop_nulls()
    Xtest: PolarsFrame = Xtest.drop_nulls()

    # Set XGBoost parameters.
    dtrain = DMatrix(
        data=Xtrain.to_numpy(), 
        label=train_label, 
        feature_names=Xtrain.columns
    )
    dtest = DMatrix(
        data=Xtest.to_numpy(),
        label=test_label,
        feature_names=Xtrain.columns,
    )
    evals: list[tuple[DMatrix, str]] = [(dtrain, "train"), (dtest, "test")]
    evals_result: dict = {}

    # Train a model.
    model: Booster = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )
    
    # Return the trained model and test data.
    return model, dtrain, dtest, evals_result


def train_xgboost_full(
    X_train: PolarsFrame,
    y_train: PolarsFrame,
    params: dict = XGB_PARAMS,
    estimators: int = N_ESTIMATORS,
    sample_weights: ndarray | None = None
) -> Booster:
    """Initializes and trains an XGBoost model on the full dataset without 
    testing, evaluation or early stopping, with optional sample weights."""
    
    # Drop missing values.
    Xtrain: PolarsFrame = X_train.drop_nulls()
    ytrain: PolarsFrame = y_train.drop_nulls()
    
    # Set XGBoost parameters.
    dtrain = DMatrix(
        Xtrain.to_numpy(), 
        label=ytrain.to_numpy(), 
        feature_names=X_train.columns
    )
    
    # Set weights if available.
    if sample_weights is not None:
        dtrain.set_weight(sample_weights["weight"].to_numpy())
    
    # Train a model.
    model: Booster = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=estimators,
        verbose_eval=False,
    )
    
    # Return the trained model.
    return model


def training_loop(
    cv_folds: list[tuple],
    metrics_per_fold: list[dict],
    extreme_weeks_dict: dict[str, list[str]],
    model_path: str = MODEL_PATH,
    targets: list[str] = TARGETS,
    multiparams: dict = XGB_PARAMS,
    singleparams: dict = SINGLE_TARGET_PARAMS,
    newdata: bool = False,
) -> tuple[Booster, PolarsFrame, PolarsFrame, PolarsFrame, list[str], dict]:
    """Trains an XGBoost model per target for three random extreme value weeks 
    and a multi-output model for rolling basis CV folds. Uses feature 
    importance from extreme weeks' models to create interaction features and 
    applies weights to extreme value weeks in the final model. Returns the 
    last fold model, predictions, true values, X validation values, features, 
    and evals."""

    total_folds: int = len(cv_folds)
    feature_importance: dict[str, list[tuple]] = {t: [] for t in targets}
    print("\nTraining:")
    
    for fold, (Xtrain, ytrain, Xtest, ytest) in enumerate(cv_folds, start=1):
        print(f"Split {fold}/{total_folds}")
        
        # Add lagged features.
        X_train: PolarsFrame; y_train: PolarsFrame; X_test: PolarsFrame; y_test: PolarsFrame
        X_train, y_train, X_test, y_test = get_lagged_data(
            Xtrain, ytrain, Xtest, ytest
        )

        if fold <= len(targets) * 3:  # Extreme weeks: 3 weeks per target.
            
            for target in targets: # Train a model for each target.
                model: Booster; dtrain: DMatrix; dtest: DMatrix; evals: dict
                model, dtrain, dtest, evals = train_xgboost(
                    X_train,
                    X_test,
                    get_train_label(y_train, target),
                    get_test_label(y_test, target),
                    singleparams
                )
                
                # Predict and evaluate.
                iter_range: tuple = (0, model.best_iteration + 1)
                prediction: ndarray = model.predict(
                    dtest, iteration_range=iter_range
                )
                y_pred: PolarsFrame = get_prediction_dataframe(
                    prediction, y_test, target=target
                )

                # Store metrics.
                fold_metrics: dict = get_prediction_metrics(
                    y_test.select(target),
                    y_pred,
                    model,
                    dtrain,
                    y_train,
                    target=target
                )
                metrics_per_fold.append(
                    {**fold_metrics, "target": target, "fold": fold}
                )
                
                # The top two most important features from extreme folds.
                importance: dict = model.get_score(importance_type="gain")
                feature_importance[target].append(
                    sorted(
                        importance.items(), key=lambda x: x[1], reverse=True
                    )[:2]
                )

                # Free memory.
                del model, dtrain, dtest, prediction, y_pred, fold_metrics, importance
                collect()  # Garbage collection.
        
        else:  # Rolling basis CV folds, all targets.
            
            model: Booster; dtrain: DMatrix; dtest: DMatrix; evals: dict
            model, dtrain, dtest, evals = train_xgboost(
                X_train,
                X_test,
                get_multitarget_label(y_train),
                get_multitarget_label(y_test),
                multiparams
            )
            
            # Predict and evaluate.
            iter_range: tuple = (0, model.best_iteration + 1)
            prediction: ndarray = model.predict(dtest, iteration_range=iter_range)
            y_pred: PolarsFrame = get_prediction_dataframe(prediction, y_test)
            
            # Store metrics for each target.
            for target in targets:
                fold_metrics: dict = get_prediction_metrics(
                    y_test, y_pred, model, dtrain, y_train
                )
                metrics_per_fold.append(
                    {**fold_metrics, "target": target, "fold": fold}
                )
                # Free memory.
                del fold_metrics

            # Free memory.
            if fold != total_folds:
                del model, y_pred # These are returned.
            del dtrain, dtest, prediction
            collect()  # Garbage collection.
        
        if fold == total_folds: # The last rolling basis CV fold.
            
            # Merge the data back to one Polars DataFrame.
            training_data: PolarsFrame; full_data: PolarsFrame 
            training_data, full_data = unsplit_data(
                X_train, y_train, X_test, y_test
            )
            # Get features.
            features: list[str] = get_features(training_data)
            
            # Add interaction features based on extreme weeks' models.
            full_data: PolarsFrame = add_interaction_features(
                full_data, feature_importance, features
            )
                
            if newdata:  # Save the processed data.
                save_processed_data(full_data)
        
        if fold != total_folds:
            del X_test, y_test # Free memory.
            collect() # Garbage colletion.
    
    # Free memory.
    del X_train, y_train
    collect()  # Garbage collection.

    # X and y for the final model.
    X_train_full: PolarsFrame = full_data.drop(targets + ["index"])
    y_train_full: PolarsFrame = full_data[targets]
    
    # Weights from extreme value weeks for a final model.
    sample_weights: PolarsFrame = get_weights(full_data, extreme_weeks_dict)
    
    # Free memory.
    del training_data, full_data
    collect() # Garbage collection.

    # The final model trained with all the data and the weights.
    final_model: Booster = train_xgboost_full(
        X_train_full, y_train_full, sample_weights=sample_weights
    )

    # Save the final model.
    final_model.save_model(model_path)
    print(f"\nA model trained on all data with sample weights and saved to {model_path}")
    
    # Free memory.
    del final_model
    collect() # Garbage collection.

    # Return rolling basis cross-validation results from the last iteration.
    return model, y_pred, y_test, X_test, features, evals
