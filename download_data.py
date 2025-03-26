"""A function for downloading CDAWeb data."""

from time import perf_counter

from pyspedas import ace, wind


def load_cdaweb_data(start: str, end: str, params: dict) -> None:
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
