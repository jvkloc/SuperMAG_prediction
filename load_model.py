"""Functions for using a saved model."""

from numpy import ndarray
from pandas import DataFrame
from xgboost import Booster, DMatrix

from constants import MODEL_PATH


def load_xgb_model(path: str = MODEL_PATH) -> Booster:
    """Returns a saved XGBoost model from the path."""
    model = Booster()
    model.load_model(path)
    return model


def predict_with_loaded_model(model: Booster, X_test: DataFrame) -> ndarray:
    """Returns a prediction by the given model."""
    dtest = DMatrix(X_test)
    y_pred: ndarray = model.predict(dtest)
    return y_pred
