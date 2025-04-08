# author: matthew learn (matt.learn@marine.rutgers.edu)
# This file contains functions for processing model data. Used by 1_main.py
# Contains functions for processing individual model data and comparing model data.

import numpy as np
import xarray as xr
import xesmf as xe

from dask.diagnostics import ProgressBar
from datetime import date
from datetime import datetime as dt
import itertools

from .util import (
    print_starttime,
    print_endtime,
    print_runtime,
    generate_data_filename,
    save_data,
)
from .models import CMEMS, ESPC
from .pathfinding import *


"""
Section 1: Individual Model Processing Functions
"""


def regrid_ds(ds1: xr.Dataset, ds2: xr.Dataset, diag_text: bool = True) -> xr.Dataset:
    """
    Regrids the first dataset to the second dataset.

    Args
    ----------
        ds1 (xr.Dataset): The first dataset. This is the dataset that will be regridded.
        ds2 (xr.Dataset): The second dataset. This is the dataset that the first dataset will be regridded to.
        diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns
    ----------
        ds1_regridded (xr.Dataset)
            The first dataset regridded to the second dataset.
    """
    text_name = ds1.attrs["text_name"]
    model_name = ds1.attrs["model_name"]
    fname = ds1.attrs["fname"]

    if diag_text:
        print(f"{text_name}: Regridding to {ds2.attrs['text_name']}...")
        starttime = print_starttime()

    # ds1 = ds1.drop_vars(["time"])

    # Code from Mike Smith.
    ds1_regridded = ds1.reindex_like(ds2, method="nearest")

    grid_out = xr.Dataset({"lat": ds2["lat"], "lon": ds2["lon"]})
    regridder = xe.Regridder(ds1, grid_out, "bilinear", extrap_method="nearest_s2d")

    ds1_regridded = regridder(ds1)
    ds1_regridded.attrs["text_name"] = text_name
    ds1_regridded.attrs["model_name"] = model_name
    ds1_regridded.attrs["fname"] = fname

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return ds1_regridded


def interpolate_depth(
    model: object,
    max_depth: int = 1000,
    diag_text: bool = True,
) -> xr.Dataset:
    """
    Interpolates the model data to 1 meter depth intervals.

    Args
    ----------
        model (object): The model data.
        max_depth (int, optional): The maximum depth to interpolate to. Defaults to 1000.
        common_grid (xr.Dataset): The common grid to interpolate to (CMEMS ONLY).
        diag_text (bool, optional): Print diagnostic text. Defaults to True.

    Returns:
    ----------
        ds_interp (xr.Dataset): The interpolated model data.
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

    u_interp = u.interp(depth=z_range)
    v_interp = v.interp(depth=z_range)

    ds_interp = xr.Dataset({"u": u_interp, "v": v_interp})

    ds_interp.attrs["text_name"] = text_name
    ds_interp.attrs["model_name"] = model_name
    ds_interp.attrs["fname"] = f"{model_name}_zinterp"
    ds_interp = ds_interp.chunk("auto")

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

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

    ds_da = ds.mean(dim="depth", keep_attrs=False)

    ds_da.attrs["text_name"] = text_name
    ds_da.attrs["model_name"] = model_name
    ds_da.attrs["fname"] = f"{model_name}_dac"

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

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
    fname = data.attrs["fname"]

    if diag_text:
        print(f"{text_name}: Calculating magnitude...")
        starttime = print_starttime()

    # Calculate magnitude (derived from Pythagoras)
    magnitude = np.sqrt(data["u"] ** 2 + data["v"] ** 2)

    data = data.assign(magnitude=magnitude)
    data.attrs["text_name"] = text_name
    data.attrs["model_name"] = model_name
    data.attrs["fname"] = fname
    data = data.chunk("auto")  # just to make sure

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return data


"""
Section 2: Model Comparison Calculations
"""


def calculate_simple_diff(
    model1: object, model2: object, diag_text: bool = True
) -> xr.Dataset:
    """
    Calculates the simple difference between two datasets. Returns a single xr.Dataset of the simple difference.

    Args
    ----------
        model1 (object): The first model.
        model2 (object): The second model.
        diag_text (bool, optional): Print diagnostic text.


    Returns
    ----------
        simple_diff (xr.Dataset): The simple difference between the two datasets.
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

    if diag_text:
        print(f"{text_name}: Calculating Simple Difference...")
        starttime = print_starttime()

    simple_diff = data1 - data2
    simple_diff.attrs["model_name"] = f"{model_name}_speed_diff"
    simple_diff.attrs["text_name"] = f"Speed Difference [{text_name}]"
    simple_diff.attrs["fname"] = simple_diff.attrs["model_name"]
    simple_diff.attrs["model1_name"] = data1.attrs["text_name"]
    simple_diff.attrs["model2_name"] = data2.attrs["text_name"]

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return simple_diff


def calculate_rms_vertical_diff(
    model1: object, model2: object, regrid: bool = False, diag_text: bool = True
) -> xr.Dataset:
    """
    Calculates the vertical root mean squared difference between two datasets.

    Args
    ----------
        model1 (object): The first model.
        model2 (object): The second model.
        regrid (bool, optional): Whether to regrid datasets. Defaults to `False`.\
        diag_text (bool, optional): Print diagnostic text.

    Returns:
    ----------
        vrmsd (xr.Dataset): The vertical root mean squared difference between the two datasets.
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

    if diag_text:
        print(f"{text_name}: Calculating RMSD...")
        starttime = print_starttime()

    if regrid:
        data2 = regrid_ds(data2, data1, diag_text=False)  # regrid model2 to model1.

    # Calculate RMSD
    delta_u = data1.u - data2.u
    delta_v = data1.v - data2.v

    vrmsd_u = np.sqrt(np.square(delta_u).mean(dim="depth"))
    vrmsd_v = np.sqrt(np.square(delta_v).mean(dim="depth"))
    vrmsd_mag = np.sqrt(vrmsd_u**2 + vrmsd_v**2)

    vrmsd = xr.Dataset(
        {"u": vrmsd_u, "v": vrmsd_v, "magnitude": vrmsd_mag},
        attrs={"text_name": text_name, "model_name": model_name, "fname": model_name},
    )

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return vrmsd


def calculate_simple_mean(
    model_list: list[object], diag_text: bool = True
) -> xr.Dataset:
    """
    Calculates the simple mean of a list of datasets. Returns a single xr.Dataset of the simple means.

    Args
    ----------
        model_list (list[object]): A list of xr.Datasets.
        diag_text (bool, optional): Print diagnostic text.

    Returns
    ----------
        simple_mean (xr.Dataset): The simple mean of the list of datasets.
    """
    if diag_text:
        print("Calculating simple mean of selected models...")
        starttime = print_starttime()

    datasets = [model.da_data for model in model_list]
    model_names = "_".join([dataset.attrs["model_name"] for dataset in datasets])
    text_names = ", ".join([dataset.attrs["text_name"] for dataset in datasets])

    combined_dataset = xr.concat(datasets, dim="datasets")
    simple_mean = combined_dataset.mean(dim="datasets")
    simple_mean.attrs["model_name"] = f"{model_names}_simple_mean"
    simple_mean.attrs["text_name"] = f"Simple Mean [{text_names}]"
    simple_mean.attrs["fname"] = simple_mean.attrs["model_name"]

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return simple_mean


def calculate_mean_diff(model_list: list[object], diag_text: bool = True) -> xr.Dataset:
    """
    Calculates the mean of the differences of each non-repeating pair of models from the passed list of datasets.
    Returns a single xr.Dataset of the mean differences.

    Args
    ----------
        model_list (list[object]): A list of xr.Datasets.
        diag_text (bool, optional): Print diagnostic text.

    Returns
    ----------
        mean_diff (xr.Dataset): The mean difference of all selected models.
    """
    if diag_text:
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
    mean_diff.attrs["fname"] = mean_diff.attrs["model_name"]

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return mean_diff


"""
Section 3: Model Processing Functions
"""


def process_common_grid(
    dates: tuple[str, str], extent: tuple[float, float, float, float], depth: int
) -> xr.Dataset:
    """
    Loads and subsets data from ESPC to act as a common grid for all models. In the event of a failure, CMEMS will be used as a common grid instead.

    Args
    ----------
        dates (tuple[str, str]): A tuple of (start_date, end_date).
        extent (tuple[float, float, float, float]): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        depth (int): The maximum depth in meters.

    Returns
    ----------
        common_grid (xr.Dataset): Common grid data.
    """
    print("Setting up COMMON_GRID...")
    starttime = print_starttime()

    try:
        temp = ESPC()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        today = date.today()
        temp.subset((today, today), extent, depth, diag_text=False)
        temp.subset_data["time"] = [np.datetime64(dates[0])]
        temp.subset_data = temp.subset_data.isel(time=0)
        common_grid = temp.subset_data
    except Exception as e:
        print(f"ERROR: Failed to process ESPC COMMON GRID data due to: {e}\n")
        print("Processing CMEMS COMMON GRID instead...\n")
        temp = CMEMS()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        temp.subset(dates, extent, depth, diag_text=False)
        common_grid = temp.subset_data

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)

    return common_grid


def process_individual_model(
    model: object,
    common_grid: xr.Dataset,
    dates: tuple[str, str],
    extent: tuple[float, float, float, float],
    depth: int,
    single_date: bool,
    pathfinding: bool,
    heuristic: str,
    waypoints: list[tuple[float, float]] = None,
    glider_speed: float = None,
    mission_name: str = None,
    save: bool = False,
) -> None:
    """
    Processes individual model data. Assigns regridded subset data,
    1 meter interval interpolated data, & depth averaged to model class attributes.

    Args
    ----------
        model (object): The model data.
        common_grid (xr.Dataset): Common grid data.
        dates (tuple[str, str]): A tuple of (date_min, date_max) in datetime format.
        extent (tuple[float, float, float, float]): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        depth (int): The maximum depth in meters.
        single_date (bool): Boolean indicating whether to subset data to a single datetime.
        pathfinding (dict): Dictionary of pathfinding parameters.
        heuristic (str): Pathfinding heuristic. Options: "drift_aware", "haversine".
        waypoints (list[tuple[float, float]]): List of waypoints for the A* computation.
        mission_name (str): Name of the mission.
        save (bool): Save each data to netCDF.
    """
    print(f"{model.name}: Processing data...")
    starttime = print_starttime()

    # subset
    model.subset(dates, extent, depth, diag_text=False)

    if "time" in model.subset_data.dims:
        # Select first timestep, drop time as dimension
        model.subset_data = model.subset_data.isel(time=0).drop_vars("time")
        # Add as coordinate
        model.subset_data = model.subset_data.assign_coords(
            time=("time", [np.datetime64(dates[0])])
        )
    model.subset_data = regrid_ds(model.subset_data, common_grid, diag_text=False)

    # interpolate depth
    model.z_interpolated_data = interpolate_depth(model, depth, diag_text=False)
    with ProgressBar(minimum=1):
        model.z_interpolated_data = model.z_interpolated_data.persist()

    # depth average
    model.da_data = depth_average(model, diag_text=False)
    model.da_data = calculate_magnitude(model, diag_text=False)
    with ProgressBar(minimum=1):
        # TODO: would it be faster for A* to be computed with it loaded?
        model.da_data = model.da_data.compute()

    if save:
        fdate = model.da_data.time.dt.strftime("%Y%m%d%H").values
        ddate = model.da_data.time.dt.strftime("%Y_%m_%d").values
        fname_zi = model.z_interpolated_data.attrs["fname"]
        fname_da = model.da_data.attrs["fname"]

        full_fname_zi = generate_data_filename(mission_name, fdate, fname_zi)
        full_fname_da = generate_data_filename(mission_name, fdate, fname_da)

        # save_data(model.z_interpolated_data, full_fname_zi, ddate)
        save_data(model.da_data, full_fname_da, ddate)

    print("Processing Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)

    # pathfinding
    if pathfinding:
        model.waypoints = waypoints
        model.optimal_path = compute_a_star_path(
            waypoints,
            model,
            heuristic,
            glider_speed,
            mission_name,
        )
    else:
        model.waypoints = None
        model.optimal_path = None
