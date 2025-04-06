"""Functions for downloading and processing data."""

from os import path
from time import perf_counter

from pandas import concat, DataFrame, read_parquet
from pyspedas import ace, wind

from constants import START, END, CDAWEB_PARAMS, DATA_PATH


def load_cdaweb_data(
    start: str = START, end: str = END, params: dict = CDAWEB_PARAMS
) -> None:
    """Downloads the parameter data from CDAWeb."""
    trange: list[str] = [start, end]
    # Start timing.
    download_start: float = perf_counter()
    # Start downloading.
    ace_mfi: list = ace.mfi(trange=trange, varnames=params["ace_mfi"], datatype="h0")
    ace_swe: list = ace.swe(trange=trange, varnames=params["ace_swe"], datatype="h0")
    wind_mfi: list = wind.mfi(trange=trange, varnames=params["wind_mfi"], datatype="h0")
    wind_swe: list = wind.swe(trange=trange, varnames=params["wind_swe"], datatype="k0")
    # Stop timing.
    download_end: float = perf_counter()
    # Get download time in minutes.
    download_time: float = (download_end - download_start) / 60
    # Print the download time.
    print(f"Loading the CDAWeb data took {download_time:.2f} minutes.")


def load_processed_data(data_path: str = DATA_PATH) -> DataFrame:
    """Loads a DataFrame from data_path."""
    if path.exists(data_path):
        data: DataFrame = read_parquet(data_path)
        print(f"Loaded data shape: {data.shape}")
        return data
    else:
        raise FileNotFoundError(f"No data found at {data_path}")
    

def save_processed_data(data: DataFrame, path: str = DATA_PATH) -> None:
    """Saves the preprocessed data to path."""
    print("Saving the processed data...")
    data.to_parquet(path)
    print(f"Data saved to {path}")


def unsplit_data(
    X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Recombines a train-test split into one DataFrame and returns it."""
    # Combine the training data.
    train: DataFrame = concat([X_train, y_train], axis=1)
    # Combine the test data.
    test: DataFrame = concat([X_test, y_test], axis=1)
    # Combine train and test vertically.
    data: DataFrame = concat([train, test], axis=0)

    # Reset index if needed
    #data = data.reset_index(drop=True)

    # Return the recombined data.
    return train, data
