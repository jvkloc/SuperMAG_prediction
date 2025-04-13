"""Functions for printing SuperMAG file info and combining SuperMAG .csv 
files."""

from gc import collect
from glob import glob
from os import path
from pandas import concat, DataFrame, read_csv

from constants import SMAG_PATH
from utils import get_supermag_data


def print_SuperMAG_info(smag_path: str = SMAG_PATH) -> None:
    """Prints info on the selected file from the SuperMAG folder."""
    file: str = "2020-supermag.csv"
    superMAG: DataFrame = read_csv(smag_path + file, sep=",")
    print(superMAG.head())
    superMAG.info()


def get_dataframe_list(csv_path: str = SMAG_PATH) -> list[DataFrame]:
    """Creates DataFrames from all .csv files in the given folder and returns 
    them in a list."""
    # Get .csv file path list from the folder.
    filepaths: list[str] = glob(f"{csv_path}*.csv")
    # Ensure chronological order according to which filenames are named.
    filepaths.sort()
    # Empty list for processed SuperMAG files.
    superMAGs: list[DataFrame] = []
    # Preprocess all the files.
    for filepath in filepaths:
        # Split the file path into folder and filename.
        folder, file = path.split(filepath)
        # Add / to the end of the path.
        folder += '/'
        # Create a DataFrame and append it to the list.
        superMAGs.append(get_supermag_data(path=folder, file=file))
    # Return the DataFrame list.
    return superMAGs


def combine_dataframes(frames: list[DataFrame]) -> None:
    """Returns all DataFrames from the given list combined to a new DataFrame.
    Prints the shape and memory usage of the new DataFrame. Deletes the
    DataFrame list and frees the memory allocated for it."""
    combined: DataFrame = concat(frames, ignore_index=True)
    print(combined.shape)
    print(combined.memory_usage(deep=True).sum() / (1024**2), "MB")
    del frames
    collect() # Free the memory allocated to the DataFrame list.
    return combined


def save_large_dataframe(
    frame: DataFrame, csv_path: str = SMAG_PATH, filename: str = "SuperMAG.csv"
) -> None:
    """Saves the given DataFrame to a .csv file into the given folder with
    the given file name."""
    # Add file name to the path.
    file_out: str = path.join(csv_path, filename)
    # Save the DataFrame to a .csv file.
    frame.to_csv(file_out, index=False, chunksize=100_000)


def main() -> None:
    
    # Print SuperMAG .csv file info.
    print_SuperMAG_info()
    
    # Get a list of all the .csv files converted to DataFrames.
    superMAGs: list[DataFrame] = get_dataframe_list()
    
    # Combine the processed files. 
    combined: DataFrame = combine_dataframes(superMAGs)

    # Save the new file to the same folder.
    save_large_dataframe(combined)


if __name__=="__main__":
    main()
