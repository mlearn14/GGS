from tracemalloc import start
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

    # # From Sal's Code, might not use
    # valid_mask = ~np.isnan(u) & ~np.isnan(v)

    # valid_depths = z[valid_mask]
    # u_valid, v_valid = u[valid_mask], v[valid_mask]

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


def calculate_rmse(model1: xr.Dataset, model2: xr.Dataset) -> xr.DataArray:
    """
    Calculates the root mean squared error between two datasets.

    Args:
        - model1 (xr.Dataset): The first dataset.
        - model2 (xr.Dataset): The second dataset.

    Returns:
        - rmse (xr.DataArray): The root mean squared error between the two datasets.
    """
    model1name = model1.attrs["model"]
    model2name = model2.attrs["model"]

    print(f"{model1name} & {model2name}: Calculating RMSE...")
    starttime = print_starttime()

    # Interpolate model2 to model1. Code from Mike Smith.
    grid_out = xr.Dataset({"lat": model1["lat"], "lon": model1["lon"]})
    regridder = xe.Regridder(model2, grid_out, "bilinear", extrap_method="nearest_s2d")
    model2_interp = regridder(model2)

    diff = model1.magnitude - model2_interp.magnitude
    rmse = np.sqrt(np.square(diff).mean(dim="depth"))

    rmse.attrs["model"] = f"{model1name} & {model2name}"

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return rmse


def calculate_mae(model1: xr.Dataset, model2: xr.Dataset) -> xr.DataArray:
    """
    Calculates the mean absolute error between two datasets.

    Args:
        - model1 (xr.Dataset): The first dataset.
        - model2 (xr.Dataset): The second dataset.

    Returns:
        - mae (xr.DataArray): The mean absolute error between the two datasets.
    """
    model1name = model1.attrs["model"]
    model2name = model2.attrs["model"]

    print(f"{model1name} & {model2name}: Calculating MAE...")
    starttime = print_starttime()

    # Interpolate model2 to model1. Code from Mike Smith.
    grid_out = xr.Dataset({"lat": model1["lat"], "lon": model1["lon"]})
    regridder = xe.Regridder(model2, grid_out, "bilinear", extrap_method="nearest_s2d")
    model2_interp = regridder(model2)

    diff = model1.magnitude - model2_interp.magnitude
    mae = np.abs(diff).mean(dim="depth")

    mae.attrs["model"] = f"{model1name} & {model2name}"

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return mae
