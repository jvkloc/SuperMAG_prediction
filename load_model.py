"""Functions for using a saved model."""

from numpy import ndarray
from pandas import DataFrame
from xgboost import Booster, DMatrix

from constants import MODEL_PATH


def load_model(path: str = MODEL_PATH) -> Booster:
    """Returns a saved XGBoost model from the path."""
    model = Booster()
    model.load_model(path)
    return model


def predict_with_loaded_model(
    model: Booster, X_test: DataFrame, y_test: DataFrame
) -> DataFrame:
    """Predict with best iteration of the given model."""
    best_iteration: int = model.best_iteration
    dtest = DMatrix(X_test, label=y_test)
    irange: tuple[int, int] = (0, best_iteration + 1)
    y_pred: ndarray = model.predict(dtest, iteration_range=irange)
    print(f"Loaded model best iteration: {best_iteration}")
    return y_pred
