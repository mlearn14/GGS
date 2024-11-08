import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cool_maps.plot as cplt

# TODO: add plotting functions


def plot_magnitude(ds: xr.Dataset, extent: tuple) -> None:
    """
    Plot current magnitudes.
    """
    pass

def plot_threshold() -> None:
    """
    Plot current threshold.
    """
    pass

def test_plot(extent: tuple, data, path: str) -> None:
    lon_min, lat_min, lon_max, lat_max = extent
    cplt.create([lon_min, lon_max, lat_min, lat_max], proj=ccrs.PlateCarree())

    plt.savefig(path)
