"""Functions for printing SuperMAG file info and loading and combining 
SuperMAG .csv files.
"""

from gc import collect
from glob import glob
from os import path

from pandas import DataFrame, read_csv
from polars import all_horizontal, col, concat, Datetime, Expr, LazyFrame, scan_csv, Float64

from constants import SMAG_PATH, PATH, FILE, FILL, TARGETS


def print_supermag_info(file: str, smag_path: str = SMAG_PATH) -> None:
    """Prints info on the given file from the SuperMAG folder."""
    superMAG: DataFrame = read_csv(smag_path + file, sep=",")
    print(superMAG.head())
    superMAG.info()


def load_supermag_data(
    path: str = SMAG_PATH,
    file: str = FILE,
    fill: int = FILL,
    targets: list[str] = TARGETS
) -> LazyFrame:
    """Returns SuperMAG data in a Polars LazyFrame."""
    # Load the .csv to LazyFrame.
    supermag: LazyFrame = (
        scan_csv(f"{path}{file}")
        .with_columns(
            col("Date_UTC")
            .str.strptime(Datetime, "%Y-%m-%d %H:%M:%S")
        )
        .set_sorted("Date_UTC")
    )
    
    # Filter out rows where any target has the fill value.
    filter: Expr = all_horizontal(col(targets) != fill)
    supermag: LazyFrame = supermag.filter(filter)
    
    # Return the LazyFrame.
    return supermag


def load_supermag_data2(
    path: str = SMAG_PATH,
    file: str = FILE,
    fill: int = FILL,
    targets: list[str] = TARGETS
) -> LazyFrame:
    """Returns SuperMAG data in a Polars LazyFrame."""
    schema_overrides = {
        "SMEr00": Float64,
        "SMEr02": Float64,
        "SMEr04": Float64,
        "SMEr06": Float64,
        "SMEr08": Float64,
        "SMEr10": Float64,
        "SMEr12": Float64,
        "SMEr14": Float64,
        "SMEr16": Float64,
        "SMEr18": Float64,
        "SMEr20": Float64,
        "SMEr22": Float64,
    }
    
    try:
        supermag: LazyFrame = (
            scan_csv(
                f"{path}{file}",
                schema_overrides=schema_overrides,
                infer_schema_length=10000,
                encoding="utf8",
                null_values=["-3486353637195406624423936", "N/A", "null"]
            )
            .with_columns(
                col("Date_UTC").str.strptime(Datetime, "%Y-%m-%d %H:%M:%S"),
                *[col(c).fill_null(0) for c in schema_overrides.keys()]
            )
            .set_sorted("Date_UTC")
        )
        
        filter: Expr = all_horizontal(col(targets) != fill)
        supermag: LazyFrame = supermag.filter(filter)
        
        return supermag
    except Exception as e:
        print(f"Error processing {file}: {e}")
        raise


def get_dataframe_list(csv_path: str = SMAG_PATH) -> list[LazyFrame]:
    """Creates LazyFrames from all .csv files in the given folder and returns 
    them in a list."""
    # Get .csv file path list from the folder.
    filepaths: list[str] = glob(f"{csv_path}*.csv")
    # Ensure chronological order according to which filenames are named.
    filepaths.sort()
    # Empty list for processed SuperMAG files.
    superMAGs: list[LazyFrame] = []
    # Preprocess all the files.
    for filepath in filepaths:
        # Split the file path into folder and filename.
        folder, file = path.split(filepath)
        # Add / to the end of the path.
        folder += '/'
        # Create a DataFrame and append it to the list.
        superMAGs.append(load_supermag_data2(file=file))
    # Return the DataFrame list.
    return superMAGs


def combine_dataframes(frames: list[LazyFrame]) -> LazyFrame:
    """Returns all LazyFrames from the given list combined to a new LazyFrame. 
    Deletes the LazyFrame list and frees the memory allocated for it."""
    combined: LazyFrame = concat(frames, how="vertical")
    del frames # Delete the list given as argument.
    collect()  # Free the memory allocated to the deleted DataFrames list.
    return combined


def save_large_dataframe(
    frame: LazyFrame, csv_path: str = PATH, filename: str = "super.csv"
) -> None:
    """Saves the DataFrame to a .csv file into the given folder with the given 
    file name."""
    # Add file name to the path.
    file_out: str = path.join(csv_path, filename)
    # Save the Dask DataFrame to a .csv file.
    frame.sink_csv(file_out)


def main() -> None:
    
    # Print SuperMAG .csv file info.
    #print_supermag_info("2014-supermag.csv")
    
    # Get all the .csv files converted to Dask DataFrames in a list.
    superMAGs: list[DataFrame] = get_dataframe_list()
    
    # Combine the Dask DataFrames. 
    combined: DataFrame = combine_dataframes(superMAGs)

    # Save the new DataFrame to the same folder as a .csv file.
    save_large_dataframe(combined)


if __name__=="__main__":
    main()
