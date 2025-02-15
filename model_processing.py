# author: matthew learn (matt.learn@marine.rutgers.edu)
from functions import *
from models import *
from pathfinding import *
from plotting import *
import itertools


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

    return common_grid



