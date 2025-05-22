"""Functions for loading and processing data."""

from glob import glob
from os import path
from time import perf_counter
from typing import NamedTuple

from numpy import ndarray
from polars import (
    all_horizontal,
    coalesce,
    col,
    DataFrame as PolarsFrame,
    Datetime,
    Expr,
    Float32,
    LazyFrame,
    len as p_len,
    lit,
    scan_csv,
    scan_parquet,
    Series,
) 
from pyspedas import ace, cdf_to_tplot, wind
from pytplot import get_data, del_data, tplot_names 

from constants import (
    TARGETS,
    START,
    END,
    CDAWEB_PARAMS,
    CDAWEB_FILL,
    CDAWEB_PATH,
    SMAG_PATH,
    FILE,
    DATA_PATH,
    FILL,
    RE,
    K,
    MP,
)


def load_cdaweb_data(
    start: str = START, end: str = END, params: dict = CDAWEB_PARAMS
) -> None:
    """Checks if existing files are up to date and downloads new files from 
    CDAWeb if necessary."""
    
    # Set time range.
    trange: list[str] = [start, end]
    # Start timing.
    start_time: float = perf_counter()
    # Check/download the files.
    _: list = ace.mfi(trange=trange, varnames=params["ace_mfi"], datatype="h0")
    _: list = ace.swe(trange=trange, varnames=params["ace_swe"], datatype="h0")
    _: list = wind.mfi(trange=trange, varnames=params["wind_mfi"], datatype="h0")
    _: list = wind.swe(trange=trange, varnames=params["wind_swe"], datatype="k0")
    # Stop timing.
    end_time: float = perf_counter()
    # Get loading time in minutes.
    loading_time: float = (end_time - start_time) / 60
    # Print the download time.
    print(f"Loading the CDAWeb data took {loading_time:.2f} minutes.\n")
    print(tplot_names())


def load_cdaweb_from_disk(
    start: str = START,
    end: str = END,
    params: dict = CDAWEB_PARAMS,
    base_path: str = CDAWEB_PATH
) -> None:
    """Loads local .cdf files from disk into tplot variables."""
    # Start timing.
    start_time: float = perf_counter()

    # Clear any existing tplot variables to avoid conflicts.
    del_data('*')
    
    # Define dataset paths and their corresponding PySPEDAS variable names.
    dataset_paths: dict[str, str] = {
        "ace_mfi": path.join(base_path, "ace/mag/level_2_cdaweb/mfi_h0"),
        "ace_swe": path.join(base_path, "ace/swepam/level_2_cdaweb/swe_h0"),
        "wind_mfi": path.join(base_path, "wind/mfi/mfi_h0"),
        "wind_swe": path.join(base_path, "wind/swe/swe_k0")
    }
    
    # Convert start and end times to years for filtering folders.
    start_year: int = int(start.split('-')[0])
    end_year: int = int(end.split('-')[0])
    
    # Load .cdf files for each dataset.
    for dataset, filepath in dataset_paths.items():
        # Get variable names from params.
        varnames: list[str] = params.get(dataset, []) if params is not None else []
        if not varnames:
            print(f"No parameters found for dataset {dataset}")
            continue
            
        # Iterate through years in the time range.
        for year in range(start_year, end_year + 1):
            year_path: str = path.join(filepath, str(year))
            if not path.exists(year_path):
                print(f"Directory not found: {year_path}")
                continue
                
            # Find all .cdf files in the year directory.
            cdf_files: list[str] = glob(path.join(year_path, "*.cdf"))
            if not cdf_files:
                print(f"No .cdf files found from {year_path}")
                continue
                
            # Load each .cdf file into tplot variables.
            try:
                cdf_to_tplot(cdf_files, varnames=varnames, merge=True)
                print(f"Loaded {len(cdf_files)} .cdf files from {year_path}")
            except Exception as e:
                print(f"Error loading .cdf files from {year_path}: {e}")
                continue
    
    # Verify loaded tplot variables.
    loaded_vars: list[str] = tplot_names()
    if not loaded_vars:
        raise ValueError("No tplot variables were loaded from .cdf files")
    
    # Stop timing and print duration.
    end_time: float = perf_counter()
    loading_time: float = (end_time - start_time) / 60
    print(f"Loading .cdf files from disk took {loading_time:.2f} minutes.")



def get_cdaweb_data(
    parameters: dict = CDAWEB_PARAMS,
    fill: float = CDAWEB_FILL,
    re: float = RE,
    k: float = K,
    mp: float = MP
) -> LazyFrame:
    """Processes CDAWeb data and returns it in a Polars LazyFrame.
    Arguments
    ---------
        parameters: CDAWeb datasets (keys) and parameters (values)
        fill: CDAWeb data fill value -1e31
        re: Earth radius (km)
        k: Boltzmann constant (J/k)
        mp: proton mass (kg)
    """
    frames: list[LazyFrame] = []

    for dataset, params in parameters.items():
        for param in params:
            # Get the data from tplot variables (in memory).
            result: NamedTuple = get_data(param, xarray=False, dt=True)
            if result is None:
                print(f"Variable {param} not found (or invalid) in {dataset}")
                continue
            # Extract time and values.
            times: ndarray = result.times
            y: ndarray = result.y
            
            # Convert time to Polars Datetime.
            time_series = Series(times).cast(Datetime).dt.round("1m")

            # Create a Polars DataFrame from the times and values.
            if len(y.shape) > 1:
                    df = PolarsFrame({
                        "time": time_series,
                        **{f"dim_{i}": y[:, i] for i in range(y.shape[1])},
                    })
                    columns: list[str] = [f"dim_{i}" for i in range(y.shape[1])]
            else:
                df = PolarsFrame({
                    "time": time_series,
                    "value": y
                })
                columns: list[str] = ["value"]
            
            # Set the data from the DataFrame to a LazyFrame.
            lf: LazyFrame = df.lazy()

            # Check fill values.
            all_fill: bool; any_fill: bool
            all_fill, any_fill = fill_values(lf, columns, fill)
            if all_fill:
                continue
            if any_fill:
                print(f"{param} from {dataset} has fill values ({fill})")

            # Deduplicate timestamps
            if len(y.shape) > 1:
                lf: LazyFrame = lf.group_by("time").agg([
                    col(f"dim_{i}").first().alias(f"dim_{i}") 
                    for i in range(y.shape[1])
                ])
            else:
                lf: LazyFrame = lf.group_by("time").agg(col("value").first())

            # Process the data.
            if "SC_pos_GSM" in param:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if len(y.shape) <= 1 or y[:, i].size == 0:
                        continue
                    lf_out: LazyFrame = lf.select(
                        col("time"),
                        (col(f"dim_{i}") / re).cast(Float32).alias(f"{dataset}_{param}_Re_{axis}")
                    )
                    frames.append(lf_out)
            elif param == "Tpr" and dataset == "wind_swe":
                if y.size == 0:
                    continue
                lf_out: LazyFrame = lf.select(
                    col("time"),
                    ((mp * col("value")**2) / (2 * k)).cast(Float32).alias(f"{dataset}_T")
                )
                frames.append(lf_out)
            elif len(y.shape) > 1:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if y[:, i].size == 0:
                        continue
                    lf_out: LazyFrame = lf.select(
                        col("time"),
                        col(f"dim_{i}").cast(Float32).alias(f"{dataset}_{param}_{axis}")
                    )
                    frames.append(lf_out)
            else:
                if y.size == 0:
                    continue
                lf_out: LazyFrame = lf.select(
                    col("time"),
                    col("value").cast(Float32).alias(f"{dataset}_{param}")
                )
                frames.append(lf_out)
    
    # Check for an empty list.
    if not frames:
        raise ValueError("No valid data was loaded")

    # Join all LazyFrames on time.
    result: LazyFrame = frames[0]
    for lf in frames[1:]:
        result: LazyFrame = result.join(lf, on="time", how="full", coalesce=True)

    print(f"CDAWeb row count: {result.select(p_len()).collect().item()}")

    # Sort the LazyFrame by time and return it.
    return result.sort("time")


def get_supermag_data(
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
            .str.strptime(Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
            .dt.round("1m")
            .alias("index")
        )
        .drop("Date_UTC")
        .set_sorted("index")
    )

    # Print statistics for each target.
    supermag_collected: PolarsFrame = supermag.collect()
    for target in targets:
        if target in supermag_collected.columns:
            t: Series = supermag_collected[target]
            print(
                f"{target} stats: mean={t.mean():.2f}, std={t.std():.2f}, median={t.median():.2f} min={t.min():.2f}, max={t.max():.2f}"
            )
        else:
            print(f"Target {target} not found in SuperMAG data")

    # Filter out rows where any target has the fill value.
    filter_expr: Expr = all_horizontal(col(targets) != fill)
    supermag_filtered: LazyFrame = supermag.filter(filter_expr)

    # Check the number of rows after filtering.
    rows: int = supermag_filtered.select(p_len()).collect().item()
    print(f"\nSuperMAG rows after removing fill values ({fill}) from {targets}: {rows}\n")

    # Return the LazyFrame.
    return supermag_filtered


def resample_cdaweb_data(
    cdaweb_data: LazyFrame,
    fill: float = CDAWEB_FILL,
    start: str = START,
    end: str = END
) -> LazyFrame:
    """Returns a Dask DataFrame with the CDAWeb data resampled to one minute
    intervals."""
    def get_key(col: str) -> str:
        return col.split('_', 2)[-1]  # e.g., 'Magnitude'
    
    cols: list[str] = cdaweb_data.collect_schema().names()
    keys: set = {get_key(col) for col in cols if col != "time"}
    exprs: list[Expr] = []
    
    for key in keys:
        # Get columns matching the base key.
        columns: list[str] = [col for col in cols if get_key(col) == key]
        if not columns:
            continue
        
        # Create coalesce expression for combining columns.
        if len(columns) == 1:
            combined: Expr = col(columns[0])
        else:
            combined: Expr = coalesce(columns)
        
        # Add to expressions with the base_key as new column name.
        exprs.append(combined.alias(key))
    
    # Process the data.
    processed: LazyFrame = (
        cdaweb_data
        # Combine columns and replace fill values.
        .with_columns(exprs)
        .select([col("time"), *keys])
        .with_columns(
            col(keys).replace(fill, None),
            col("time").cast(Datetime).dt.round("1m").alias("index")
        )
        # Filter to the specified time range
        .filter(
            (col("index") >= lit(start).str.to_datetime("%Y-%m-%d %H:%M:%S")) &
            (col("index") <= lit(end).str.to_datetime("%Y-%m-%d %H:%M:%S"))
        )
        # Drop the old index.
        .drop("time")
        # Group by and resample.
        .group_by("index").agg([col(key).mean() for key in keys])
        # Interpolate missing values.
        .with_columns(col(keys).interpolate(method="linear"))
        # Sort by index.
        .sort("index")
    )
    
    print(f"Resampled CDAWeb row count: {processed.select(p_len()).collect().item()}\n")

    # Return the resampled data.
    return processed


def merge_data(CDAWeb: LazyFrame, SuperMAG: LazyFrame) -> LazyFrame:
    """Returns the CDAWeb and SuperMAG data merged into a LazyFrame."""
    return CDAWeb.join(SuperMAG, on="index", how="inner")


def load_preprocessed_data(
    file_path: str = DATA_PATH, print_schema: bool = False
) -> LazyFrame:
    """Loads a LazyFrame from data_path."""
    if path.exists(file_path):
        data: LazyFrame = scan_parquet(file_path)
        if print_schema:
            print(f"Loaded data schema: {data.collect_schema()}")
        return data
    else:
        raise FileNotFoundError(f"No data found at {file_path}")
    

def save_processed_data(data: PolarsFrame, datapath: str = DATA_PATH) -> None:
    """Saves the data to datapath as a parquet file."""
    try:
        data.write_parquet(datapath)
        print(f"\nProcessed data saved to {datapath}")
    except Exception as e:
        print(f"\nData saving error: {e}")
        raise


def fill_values(lf: LazyFrame, columns: list[str], fill: float) -> tuple[bool, bool]:
    """Checks for fill values in specified columns of a LazyFrame.
    Returns
    -------
        all_fill: True if all values in all columns are fill.
        any_fill: True if any value in any column is fill.
    """
    checks: LazyFrame = lf.select(
        [col(c).eq(fill).all().alias(f"{c}_all") for c in columns] +
        [col(c).eq(fill).any().alias(f"{c}_any") for c in columns]
    ).collect()
    all_fill: bool = all(checks[f"{c}_all"][0] for c in columns)
    any_fill: bool = any(checks[f"{c}_any"][0] for c in columns)
    return all_fill, any_fill
