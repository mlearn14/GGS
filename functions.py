# author: matthew learn (matt.learn@marine.rutgers.edu)

import xarray as xr
import xesmf as xe
import numpy as np
import datetime as dt
from datetime import datetime


def print_starttime() -> datetime:
    """
    Prints the start time of the script.

        Args:
            - None

        Returns:
            - start_time (datetime): The start time of the script.
    """
    starttime = dt.datetime.now(dt.timezone.utc)
    print(f"Start time (UTC): {starttime}")

    return starttime


def print_endtime() -> datetime:
    """
    Prints the end time of the script.

        Args:
            None

        Returns:
            end_time (datetime): The end time of the script.
    """
    endtime = dt.datetime.now(dt.timezone.utc)
    print(f"End time (UTC): {endtime}")

    return endtime


def print_runtime(starttime: datetime, endtime: datetime) -> None:
    """
    Prints the runtime of the script.

        Args:
            - starttime (datetime): The start time of the script.
            - endtime (datetime): The end time of the script.

        Returns:
            - None
    """
    runtime = endtime - starttime
    print(f"Runtime: {runtime}")


def calculate_magnitude(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates the magnitude of the model data.

        Args:
            - ds (xr.Dataset): The model data.

        Returns:
            - ds (xr.Dataset): The model data with a new variable 'magnitude'.
    """
    model = ds.attrs["model"]

    print(f"{model}: Calculating magnitude...")
    starttime = print_starttime()

    # Calculate magnitude (derived from Pythagoras)
    magnitude = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)

    magnitude.attrs["model"] = model
    ds = ds.assign(magnitude=magnitude)
    ds = ds.chunk("auto")  # just to make sure

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return ds


def interpolate_depth(ds: xr.Dataset, max_depth: int = 1000) -> xr.Dataset:
    """
    Interpolates the model data to 1 meter depth intervals.

        Args:
            - ds (xr.Dataset): The model data.
            - max_depth (int, optional): The maximum depth to interpolate to. Defaults to 1000.

        Returns:
            - ds_interp (xr.Dataset): The interpolated model data.
    """
    model = ds.attrs["model"]

    # Define the depth range that will be interpolated to.
    z_range = np.arange(0, max_depth + 1, 1)

    u = ds["u"]
    v = ds["v"]

    print(f"{model}: Interpolating depth...")
    starttime = print_starttime()

    # .compute() is necessary because it actually computes the interpolation with parallel processing.
    u_interp = u.interp(depth=z_range).compute()
    v_interp = v.interp(depth=z_range).compute()

    ds_interp = xr.Dataset({"u": u_interp, "v": v_interp})
    ds_interp = ds_interp.chunk("auto")
    ds_interp.attrs["model"] = model

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return ds_interp


def regrid_ds(ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
    """
    Regrids the first dataset to the second dataset.

        Args:
            - ds1 (xr.Dataset): The first dataset. This is the dataset that will be regridded.
            - ds2 (xr.Dataset): The second dataset. This is the dataset that the first dataset will be regridded to.

        Returns:
            - ds1_regridded (xr.Dataset): The first dataset regridded to the second dataset.
    """
    model = ds1.attrs["model"]

    print(f"{model}: Regridding to {ds2.attrs['model']}...")
    starttime = print_starttime()

    # Code from Mike Smith.
    ds1_regridded = ds1.reindex_like(ds2, method="nearest")

    grid_out = xr.Dataset({"lat": ds2["lat"], "lon": ds2["lon"]})
    regridder = xe.Regridder(ds1, grid_out, "bilinear", extrap_method="nearest_s2d")

    ds1_regridded = regridder(ds1)
    ds1_regridded.attrs["model"] = model

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return ds1_regridded


def depth_average(ds: xr.Dataset) -> xr.Dataset:
    """
    Gets the depth integrated current velocities from the passed model data.

    Args:
        - ds (xr.Dataset): The model data.

    Returns:
        - ds_da (xr.Dataset): The depth averaged model data. Contains 'u', 'v', and 'magnitude' variables.
    """
    # TODO: remove magnitude from here and impliment it correctly
    model = ds.attrs["model"]

    print(f"{model}: Depth averaging...")
    starttime = print_starttime()

    ds_da = ds.mean(dim="depth", keep_attrs=True)

    ds_da.attrs["model"] = model

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return ds_da


def calculate_rmsd(
    model1: xr.Dataset, model2: xr.Dataset, regrid: bool = True
) -> xr.DataArray:
    """
    Calculates the root mean squared difference between two datasets.

    Args:
        - model1 (xr.Dataset): The first dataset.
        - model2 (xr.Dataset): The second dataset.
        - regrid (bool, optional): Whether to regrid model1 to model2. Defaults to True.

    Returns:
        - rmsd (xr.DataArray): The root mean squared difference between the two datasets.
    """
    model1name = model1.attrs["model"]
    model2name = model2.attrs["model"]

    print(f"{model1name} & {model2name}: Calculating RMSD...")
    starttime = print_starttime()

    if regrid:
        # Interpolate model2 to model1.
        model1 = regrid_ds(model1, model2)

    diff = model1.magnitude - model2.magnitude
    rmsd = np.sqrt(np.square(diff).mean(dim="depth"))

    rmsd.attrs["model"] = f"{model1name} & {model2name}"

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return rmsd


def calculate_mad(model1: xr.Dataset, model2: xr.Dataset) -> xr.DataArray:
    """
    Calculates the mean absolute difference between two datasets.

    Args:
        - model1 (xr.Dataset): The first dataset.
        - model2 (xr.Dataset): The second dataset.

    Returns:
        - mae (xr.DataArray): The mean absolute difference between the two datasets.
    """
    model1name = model1.attrs["model"]
    model2name = model2.attrs["model"]

    print(f"{model1name} & {model2name}: Calculating MAD...")
    starttime = print_starttime()

    # Interpolate model2 to model1.
    model2_interp = regrid_ds(model2, model1)

    diff = model1.magnitude - model2_interp.magnitude
    mad = np.abs(diff).mean(dim="depth")

    mad.attrs["model"] = f"{model1name} & {model2name}"

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return mad


def calculate_simple_mean(ds_list: list[xr.Dataset]) -> xr.Dataset:
    """
    Calculates the simple mean of a list of datasets. Returns a single xr.Dataset of the simple means.

    Args:
        - ds_list (list[xr.Dataset]): A list of xr.Datasets.

    Returns:
        - simple_mean (xr.Dataset): The simple mean of the list of datasets.
    """
    length = len(ds_list)
    total = sum(ds_list)
    simple_mean = total / length

    return simple_mean


def calculate_mean_difference(ds_list: list[xr.Dataset]) -> xr.Dataset:
    """
    Calculates the mean difference between a list of datasets. Returns a single xr.Dataset of the mean differences.

    Args:
        - ds_list (list[xr.Dataset]): A list of xr.Datasets.

    Returns:
        - mean_diff (xr.Dataset): The mean difference between the list of datasets.
    """
    length = len(ds_list)
    total = sum(ds_list)

    # TODO: find code that goes through every dataset combination. Is in a coding assignment!
