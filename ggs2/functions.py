# author: matthew learn (matt.learn@marine.rutgers.edu)
# This script contains general helper functions and individual model processing functions.

import numpy as np
import xarray as xr
import xesmf as xe

import datetime as dt
from datetime import datetime
import math
import os

"""
SECTION 1: General Helper Functions

This section contains general helper functions for the script.
"""


def logo_text() -> None:
    """Prints the GGS2 logo text."""
    print(
        rf"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~          
 ~~~~~/\\\\\\\\\\\\~~~~~~/\\\\\\\\\\\\~~~~~~/\\\\\\\\\\\~~~~~~~/\\\\\\\\\~~~~~         
  ~~~/\\\//////////~~~~~/\\\//////////~~~~~/\\\/////////\\\~~~/\\\///////\\\~~~        
   ~~/\\\~~~~~~~~~~~~~~~/\\\~~~~~~~~~~~~~~~\//\\\~~~~~~\///~~~\///~~~~~~\//\\\~~       
    ~\/\\\~~~~/\\\\\\\~~\/\\\~~~~/\\\\\\\~~~~\////\\\~~~~~~~~~~~~~~~~~~~~/\\\/~~~      
     ~\/\\\~~~\/////\\\~~\/\\\~~~\/////\\\~~~~~~~\////\\\~~~~~~~~~~~~~~/\\\//~~~~~     
      ~\/\\\~~~~~~~\/\\\~~\/\\\~~~~~~~\/\\\~~~~~~~~~~\////\\\~~~~~~~~/\\\//~~~~~~~~    
       ~\/\\\~~~~~~~\/\\\~~\/\\\~~~~~~~\/\\\~~~/\\\~~~~~~\//\\\~~~~~/\\\/~~~~~~~~~~~   
        ~\//\\\\\\\\\\\\/~~~\//\\\\\\\\\\\\/~~~\///\\\\\\\\\\\/~~~~~/\\\\\\\\\\\\\\\~  
         ~~\////////////~~~~~~\////////////~~~~~~~\///////////~~~~~~\///////////////~~ 
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                                    Glider Guidance System 2
                                          Version 1.1.0
                                    Created by Matthew Learn

                      Need help? Send an email to matt.learn@marine.rutgers.edu
        """
    )


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


def ticket_report(params: dict) -> None:
    """Prints the ticket report for the selected parameters."""
    contour_dict = {"magnitude": "Magnitude", "threshold": "Magnitude Threshold"}
    vector_dict = {"quiver": "Quiver", "streamplot": "Streamplot"}
    comp_dict = {
        "simple_diff": "Simple Difference",
        "mean_diff": "Mean Difference",
        "simple_mean": "Simple Mean",
        "rmsd_profile": "RMS Profile Difference",
    }
    ticket = {
        "Mission Name": params["mission_name"],
        "Start Date": params["start_date"],
        "End Date": params["end_date"],
        "Southwest Point": f"({params['extent'][0]}°, {params['extent'][1]}°)",
        "Northeast Point": f"({params['extent'][2]}°, {params['extent'][3]}°)",
        "Depth": params["depth"],
        "Models": ", ".join([model.name for model in params["models"]]),
        "Pathfinding": params["pathfinding"],
        "Algorithm": params["algorithm"],
        "Heuristic": params["heuristic"],
        "Waypoints": ",\n\t   ".join([f"({y}°, {x}°)" for x, y in params["waypoints"]]),
        "Glider Raw Speed": f"{params["glider_raw_speed"]} m/s",
        "Individual Plots": params["indv_plots"],
        "Contours": ", ".join(contour_dict[plot] for plot in params["contours"]),
        "Vectors": vector_dict[params["vectors"]["TYPE"]],
        "Streamline Density": params["vectors"]["STREAMLINE_DENSITY"],
        "Quiver Downscaling Value": params["vectors"]["QUIVER_DOWNSCALING"],
        "Comparison Plots": ", ".join(
            [comp_dict[plot] for plot in params["comp_plots"]]
        ),
        "Plot Optimal Path": params["plot_opt_path"],
        "Save Figures": params["save_figs"],
    }

    print(
        "----------------------------\nTicket Information:\n----------------------------"
    )
    for key, value in ticket.items():
        print(f"{key}: {value}")


def model_raw_report(model_list: list[object]) -> None:
    """Prints the raw report for the selected models."""
    min_date_list = [model.raw_data.time.min().values for model in model_list]
    max_date_list = [model.raw_data.time.max().values for model in model_list]
    print(
        "----------------------------\nModel Raw Report:\n----------------------------"
    )
    # prints the range of valid dates that can be used.
    print(f"Minimum date: {max(min_date_list).astype(str).split('T')[0]}")
    print(f"Maximum date: {min(max_date_list).astype(str).split('T')[0]}")


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


"""
SECTION 2: Individual Model Processing Functions

This section contains individual model processing functions. These functions regrid the model data to a common grid, 
interpolate depth, calculate magnitude, and depth average.
"""


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
