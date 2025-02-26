# author: matthew learn (matt.learn@marine.rutgers.edu)
# This file contains functions for processing model data. Used by 1_main.py
# Contains functions for processing individual model data and comparing model data.

import numpy as np
import xarray as xr

import datetime
import itertools

from .functions import *
from .models import CMEMS, ESPC
from .pathfinding import *


def process_common_grid(extent: tuple, depth: int) -> xr.Dataset:
    """
    Loads and subsets data from CMEMS to act as a common grid for all models. In the event of a failure, ESPC will be used as a common grid instead.

    Args:
    ----------
        - extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        - depth (int): The maximum depth in meters.

    Returns:
    ----------
        - common_grid (xr.Dataset): Common grid data.
    """
    print("Setting up COMMON_GRID...")
    starttime = print_starttime()

    try:
        temp = CMEMS()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        today = datetime.today().strftime("%Y-%m-%d")
        temp.subset((today, today), extent, depth, diag_text=False)
        common_grid = temp.subset_data
    except Exception as e:
        print(f"ERROR: Failed to process CMEMS COMMON GRID data due to: {e}\n")
        print("Processing ESPC COMMON GRID instead...\n")
        temp = ESPC()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        today = datetime.today().strftime("%Y-%m-%d")
        temp.subset((today, today), extent, depth, diag_text=False)
        common_grid = temp.subset_data

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return common_grid


def process_individual_model(
    model: object,
    common_grid: xr.Dataset,
    dates: tuple,
    extent: tuple,
    depth: int,
    single_date: bool,
    pathfinding: dict,
    mission_name: str = None,
) -> None:
    """
    Processes individual model data. Assigns regridded subset data,
    1 meter interval interpolated data, & depth averaged to model class attributes.

    Args:
    ----------
        - model (object): The model data.
        - common_grid (xr.Dataset): Common grid data.
        - dates (tuple): A tuple of (date_min, date_max) in datetime format.
        - extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        - depth (int): The maximum depth in meters.
        - single_date (bool): Boolean indicating whether to subset data to a single datetime.
        - pathfinding (dict): Dictionary of pathfinding parameters.
    """
    # subset
    model.subset(dates, extent, depth)
    if single_date:
        model.subset_data = model.subset_data.isel(time=0)
    model.subset_data = regrid_ds(model.subset_data, common_grid)

    # interpolate depth
    model.z_interpolated_data = interpolate_depth(model, depth)

    # depth average
    model.da_data = depth_average(model)
    model.da_data = calculate_magnitude(model)

    # pathfinding
    if pathfinding["ENABLE"]:
        model.waypoints = pathfinding["WAYPOINTS"]
        model.optimal_path = compute_a_star_path(
            pathfinding["WAYPOINTS"],
            model,
            pathfinding["GLIDER_RAW_SPEED"],
            mission_name,
        )
    else:
        model.waypoints = None
        model.optimal_path = None


def calculate_speed_diff(model1: object, model2: object) -> xr.Dataset:
    """
    Calculates the simple difference of the speed between two datasets. Returns a single xr.Dataset of the simple difference.

    Args:
    ----------
        - model1 (object): The first model.
        - model2 (object): The second model.

    Returns:
    ----------
        - simple_diff (xr.Dataset): The simple difference between the two datasets.
    """
    data1 = model1.da_data
    data2 = model2.da_data
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

    print(f"{text_name}: Calculating Speed Difference...")
    starttime = print_starttime()

    simple_diff = data1 - data2
    simple_diff.attrs["model_name"] = f"{model_name}_speed_diff"
    simple_diff.attrs["text_name"] = f"Speed Difference [{text_name}]"
    simple_diff.attrs["model1_name"] = data1.attrs["text_name"]
    simple_diff.attrs["model2_name"] = data2.attrs["text_name"]

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return simple_diff


def calculate_rmsd_profile(
    model1: object, model2: object, regrid: bool = False
) -> xr.Dataset:
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
    print("Calculating mean difference of selected models...")
    starttime = print_starttime()

    datasets = [model.da_data for model in model_list]
    model_names = "_".join([dataset.attrs["model_name"] for dataset in datasets])
    text_names = ", ".join([dataset.attrs["text_name"] for dataset in datasets])

    ds_combos = list(itertools.combinations(datasets, r=2))

    diff_list = []
    for ds1, ds2 in ds_combos:
        diff_list.append(abs(ds1 - ds2))

    combined_ds = xr.concat(diff_list, dim="datasets", coords="minimal")
    mean_diff = combined_ds.mean(dim="datasets")
    mean_diff.attrs["model_name"] = f"{model_names}_meandiff"
    mean_diff.attrs["text_name"] = f"Mean Difference [{text_names}]"

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return mean_diff
