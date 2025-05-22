"""Functions for using a saved model."""

from numpy import ndarray
from polars import DataFrame as PolarsFrame
from xgboost import Booster, DMatrix


def load_xgb_model(path: str) -> Booster:
    """Returns a saved XGBoost model from the path."""
    model = Booster()
    model.load_model(path)
    return model


def predict_with_loaded_model(
    model: Booster,
    X_test: PolarsFrame,
    y_test: PolarsFrame
) -> ndarray:
    """Returns a prediction by the given model based on the X_test data."""
    Xtest: PolarsFrame = X_test.drop("index")
    ytest: PolarsFrame = y_test.drop("index")
    dtest = DMatrix(
        data=Xtest.to_numpy(), 
        label=ytest.to_numpy().ravel(), 
        feature_names=Xtest.columns
    )
    y_pred: ndarray = model.predict(dtest)
    return y_pred
