from tracemalloc import start
import xarray as xr
import numpy as np
import datetime as dt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# TODO: add RMSE and MAE


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

    # Pythagorean theorem
    ds = ds.assign(magnitude=np.sqrt(ds["u"] ** 2 + ds["v"] ** 2))

    # ds = ds.chunk("auto")  # just to make sure

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
    z = ds["depth"]

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


def interpolate_coords(ds: xr.Dataset, ds_common: xr.Dataset) -> xr.Dataset:
    """
    Interpolates the model data to a given set of coordinates.

    Args:
        - ds (xr.Dataset): The model data.
        - ds_common (xr.Dataset): The common coordinates to interpolate to.

    Returns:
        - ds_interp (xr.Dataset): The interpolated model data.
    """
    print("Interpolating coordinates...")
    starttime = print_starttime()

    if "RTOFS" in ds.attrs["model"]:
        print("ugh")
        ds_interp = "nothing yet!"  # TODO: add RTOFS interpolation
    else:
        ds_interp = ds.interp(lon=ds_common.lon, lat=ds_common.lat).compute()

    ds_interp = ds_interp.chunk("auto")

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
