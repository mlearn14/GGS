# author: matthew learn (matt.learn@marine.rutgers.edu)

import datetime as dt
from datetime import datetime
import itertools
import math
import numpy as np
import os
import xarray as xr
import xesmf as xe


# Helper functions
def print_starttime() -> datetime:
    """
    Prints the start time of the script.

    Args:
    ----------
        - `None`

    Returns:
    ----------
        - start_time (datetime): The start time of the script.
    """
    starttime = dt.datetime.now(dt.timezone.utc)
    print(f"Start time (UTC): {starttime}")

    return starttime


def print_endtime() -> datetime:
    """
    Prints the end time of the script.

    Args:
    ----------
        - `None`

    Returns:
    ----------
        - end_time (datetime): The end time of the script.
    """
    endtime = dt.datetime.now(dt.timezone.utc)
    print(f"End time (UTC): {endtime}")

    return endtime


def print_runtime(starttime: datetime, endtime: datetime) -> None:
    """
    Prints the runtime of the script.

    Args:
    ----------
        - starttime (datetime): The start time of the script.
        - endtime (datetime): The end time of the script.

    Returns:
    ----------
        - `None`
    """
    runtime = endtime - starttime
    print(f"Runtime: {runtime}")


def optimal_workers(power: float = 1.0) -> int:
    """
    Calculate the optimal number of workers for parallel processing based on the available CPU cores and a power factor.

    Args:
    ----------
        - power (float): The percentage of available resources to use in processing. Values should be between 0 and 1. Defaults to 1.

    Returns:
    ----------
        - num_workers (int): The optimal number of workers for parallel processing.
    """

    print(f"Allocating {power * 100}% of available CPU cores...")

    if not 0 <= power <= 1:
        raise ValueError("Power must be between 0 and 1.")

    total_cores = os.cpu_count()

    if total_cores is None:
        total_cores = 4

    num_workers = max(1, math.floor(total_cores * power))
    print(f"Number of workers: {num_workers}")

    return num_workers


def regrid_ds(ds1: xr.Dataset, ds2: xr.Dataset, diag_text: bool = True) -> xr.Dataset:
    """
    Regrids the first dataset to the second dataset.

    Args:
    ----------
        - ds1 (xr.Dataset): The first dataset. This is the dataset that will be regridded.
        - ds2 (xr.Dataset): The second dataset. This is the dataset that the first dataset will be regridded to.
        - diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns:
    ----------
        - ds1_regridded (xr.Dataset): The first dataset regridded to the second dataset.
    """
    text_name = ds1.attrs["text_name"]
    model_name = ds1.attrs["model_name"]

    if diag_text:
        print(f"{text_name}: Regridding to {ds2.attrs['text_name']}...")
        starttime = print_starttime()

    # Code from Mike Smith.
    ds1_regridded = ds1.reindex_like(ds2, method="nearest")

    grid_out = xr.Dataset({"lat": ds2["lat"], "lon": ds2["lon"]})
    regridder = xe.Regridder(ds1, grid_out, "bilinear", extrap_method="nearest_s2d")

    ds1_regridded = regridder(ds1)
    ds1_regridded.attrs["text_name"] = text_name
    ds1_regridded.attrs["model_name"] = model_name

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    return ds1_regridded


# Calculation funtions
def interpolate_depth(
    model: object, max_depth: int = 1000, diag_text: bool = True
) -> xr.Dataset:
    """
    Interpolates the model data to 1 meter depth intervals.

    Args:
    ----------
        - model (object): The model data.
        - max_depth (int, optional): The maximum depth to interpolate to. Defaults to 1000.
        - diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns:
    ----------
        - ds_interp (xr.Dataset): The interpolated model data.
    """
    ds = model.subset_data

    text_name = ds.attrs["text_name"]
    model_name = ds.attrs["model_name"]

    # Define the depth range that will be interpolated to.
    z_range = np.arange(0, max_depth + 1, 1)

    u = ds["u"]
    v = ds["v"]

    if diag_text:
        print(f"{text_name}: Interpolating depth...")
        starttime = print_starttime()

    # .compute() is necessary because it actually computes the interpolation with parallel processing.
    u_interp = u.interp(depth=z_range).compute()
    v_interp = v.interp(depth=z_range).compute()

    ds_interp = xr.Dataset({"u": u_interp, "v": v_interp})

    ds_interp.attrs["text_name"] = text_name
    ds_interp.attrs["model_name"] = model_name
    ds_interp = ds_interp.chunk("auto")

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    return ds_interp


def depth_average(model: object, diag_text: bool = True) -> xr.Dataset:
    """
    Gets the depth integrated current velocities from the passed model data.

    Args:
    ----------
        - model (object): The model data.
        - diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns:
    ----------
        - ds_da (xr.Dataset): The depth averaged model data. Contains 'u', 'v', and 'magnitude' variables.
    """
    ds = model.z_interpolated_data

    text_name = ds.attrs["text_name"]
    model_name = ds.attrs["model_name"]

    if diag_text:
        print(f"{text_name}: Depth averaging...")
        starttime = print_starttime()

    ds_da = ds.mean(dim="depth", keep_attrs=True)

    ds_da.attrs["text_name"] = text_name
    ds_da.attrs["model_name"] = model_name

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    return ds_da


def calculate_magnitude(model: object, diag_text: bool = True) -> xr.Dataset:
    """
    Calculates the magnitude of the model data.

    Args:
    ----------
        - model (object): The model data.
        - diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns:
    ----------
        - data_mag (xr.Dataset): The model data with a new variable 'magnitude'.
    """
    data = model.da_data

    text_name = data.attrs["text_name"]
    model_name = data.attrs["model_name"]

    if diag_text:
        print(f"{text_name}: Calculating magnitude...")
        starttime = print_starttime()

    # Calculate magnitude (derived from Pythagoras)
    magnitude = np.sqrt(data["u"] ** 2 + data["v"] ** 2)

    magnitude.attrs["text_name"] = text_name
    magnitude.attrs["model_name"] = model_name
    data = data.assign(magnitude=magnitude)
    data = data.chunk("auto")  # just to make sure

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    return data


# Comparison functions


def calculate_rmsd(model1: object, model2: object, regrid: bool = False) -> xr.Dataset:
    """
    Calculates the root mean squared difference between two datasets.

    Args:
    ----------
        - model1 (object): The first model.
        - model2 (object): The second model.
        - regrid (bool, optional): Whether to regrid datasets. Defaults to `False`.

    Returns:
    ----------
        - rmsd (xr.Dataset): The root mean squared difference between the two datasets.
    """
    data1 = model1.z_interpolated_data
    data2 = model2.z_interpolated_data
    model_list = [data1, data2]
    model_list.sort(key=lambda x: x.attrs["model_name"])  # sort datasets
    data1 = model_list[0]
    data2 = model_list[1]

    text_name1: str = data1.attrs["text_name"]
    text_name2: str = data2.attrs["text_name"]
    model_name1: str = data1.attrs["model_name"]
    model_name2: str = data2.attrs["model_name"]
    text_name = " & ".join([text_name1, text_name2])
    model_name = "+".join([model_name1, model_name2])

    print(f"{text_name}: Calculating RMSD...")
    starttime = print_starttime()

    if regrid:
        data2 = regrid_ds(data2, data1, diag_text=False)  # regrid model2 to model1.

    # Calculate RMSD
    delta_u = data1.u - data2.u
    delta_v = data1.v - data2.v

    rmsd_u = np.sqrt(np.square(delta_u).mean(dim="depth"))
    rmsd_v = np.sqrt(np.square(delta_v).mean(dim="depth"))
    rmsd_mag = np.sqrt(rmsd_u**2 + rmsd_v**2)

    rmsd = xr.Dataset(
        {"u": rmsd_u, "v": rmsd_v, "magnitude": rmsd_mag},
        attrs={"text_name": text_name, "model_name": model_name},
    )

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return rmsd


# Experimental Functions TODO: implement these
def calculate_simple_mean(model_list: list[object]) -> xr.Dataset:
    """
    Calculates the simple mean of a list of datasets. Returns a single xr.Dataset of the simple means.

    Args:
    ----------
        - model_list (list[object]): A list of xr.Datasets.

    Returns:
    ----------
        - simple_mean (xr.Dataset): The simple mean of the list of datasets.
    """
    print("Calculating simple mean of selected models...")
    starttime = print_starttime()

    datasets = [model.da_data for model in model_list]
    model_names = "_".join([dataset.attrs["model_name"] for dataset in datasets])
    text_names = ", ".join([dataset.attrs["text_name"] for dataset in datasets])

    combined_dataset = xr.concat(datasets, dim="datasets")
    simple_mean = combined_dataset.mean(dim="datasets")
    simple_mean.attrs["model_name"] = f"{model_names}_simple_mean"
    simple_mean.attrs["text_name"] = f"Simple Mean [{text_names}]"

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return simple_mean


def calculate_mean_diff(model_list: list[object]) -> xr.Dataset:
    """
    Calculates the mean of the differences of each non-repeating pair of models from the passed list of datasets.
    Returns a single xr.Dataset of the mean differences.

    Args:
    ----------
        - model_list (list[object]): A list of xr.Datasets.

    Returns:
    ----------
        - mean_diff (xr.Dataset): The mean difference of all selected models.
    """
    datasets = [model.da_data for model in model_list]
    model_names = "_".join([dataset.attrs["model_name"] for dataset in datasets])
    text_names = ", ".join([dataset.attrs["text_name"] for dataset in datasets])

    ds_combos = list(itertools.combinations(datasets, r=2))

    diff_list = []
    for ds1, ds2 in ds_combos:
        diff_list.append(ds1 - ds2)

    combined_ds = xr.concat(diff_list, dim="datasets", coords="minimal")
    mean_diff = combined_ds.mean(dim="datasets")
    mean_diff.attrs["model_name"] = f"{model_names}_meandiff"
    mean_diff.attrs["text_name"] = f"Mean Difference [{text_names}]"

    return mean_diff
