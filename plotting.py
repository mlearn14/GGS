import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import cool_maps.plot as cplt

from functions import *


def plot_streamlines(
    ds: xr.Dataset, lon: np.ndarray, lat: np.ndarray, density: int = 4
) -> object:
    """
    Plot current streamlines.

    Args:
        ds (xr.Dataset): xarray dataset containing current data.
        lon (np.ndarray): 2D array of longitudes.
        lat (np.ndarray): 2D array of latitudes.
        density (int, optional): Density of streamlines. Defaults to 4.

    Returns:
        streamplot (object): Streamplot object.
    """
    u = ds["u"]
    v = ds["v"]
    # mag = ds["magnitude"]  # might not keep!

    streamplot = plt.streamplot(
        lon,
        lat,
        u,
        v,
        color="black",
        linewidth=0.5,
        density=density,
        transform=ccrs.PlateCarree(),
    )

    return streamplot


def plot_quiver(
    ds: xr.Dataset, lon: np.ndarray, lat: np.ndarray, scalar: int = 2
) -> object:
    """
    Plot current quiver.

    Args:
        ds (xr.Dataset): xarray dataset containing current data.
        lon (np.ndarray): 2D array of longitudes.
        lat (np.ndarray): 2D array of latitudes.
        scalar (int, optional): Scalar for subsampling data. Defaults to 2.

    Returns:
        quiver (object): Quiver plot object.
    """
    u = ds["u"]
    v = ds["v"]

    # subsample data to clean plot up
    u_sub = u[::scalar, ::scalar]
    v_sub = v[::scalar, ::scalar]
    lon_sub = lon[::scalar, ::scalar]
    lat_sub = lat[::scalar, ::scalar]

    quiver = plt.quiver(
        lon_sub,
        lat_sub,
        u_sub,
        v_sub,
        transform=ccrs.PlateCarree(),
    )

    return quiver


def format_cbar():
    """Fomart colorbar depending on figure size."""
    # TODO: make colorbar scale with plot size. Example: if plot is wide, make colobar horizontal


def plot_magnitude(
    ds: xr.Dataset,
    extent: tuple,
    streamlines: bool = False,
    density: int = 4,
    quiver: bool = False,
    scalar: int = 4,
) -> None:
    """
    Plot current magnitudes.

    Args:
        ds (xr.Dataset): xarray dataset containing current data.
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        streamlines (bool, optional): Plot streamlines. Defaults to False.
        density (int, optional): Density of streamlines. Defaults to 4.
        quiver (bool, optional): Plot quiver. Defaults to False.
        scalar (int, optional): Scalar for subsampling data. Defaults to 4.

    Returns:
        None
    """
    model = ds.attrs["model"]

    print(f"{model}: Plotting magnitudes...")
    starttime = print_starttime()

    lat_min, lon_min, lat_max, lon_max = extent
    cplt.create(
        [lon_min, lon_max, lat_min, lat_max], gridlines=True, proj=ccrs.PlateCarree()
    )

    if "RTOFS" in model:
        lon2D = ds["lon"]  # RTOFS already has 2D values for lat and lon
        lat2D = ds["lat"]
    else:
        lon2D, lat2D = np.meshgrid(ds.lon, ds.lat)  # create 2D values for lat and lon

    mag = ds["magnitude"]

    levels = np.linspace(0, 0.9, 10)  # hardcoded for now, but should be dynamic?

    contourf = plt.contourf(
        lon2D,
        lat2D,
        mag,
        levels=levels,
        extend="both",
        cmap=cmo.speed,
        transform=ccrs.PlateCarree(),
        vmax=0.9,
    )
    cbar = plt.colorbar(contourf, label="velocity ($\mathregular{ms^{-1}}$)")

    if streamlines:
        plot_streamlines(ds, lon2D, lat2D, density=density)
    elif quiver:
        plot_quiver(ds, lon2D, lat2D, scalar=scalar)
    else:
        print(
            '"streamlines" and "quiver" were not set to True. Plotting magnitude only.'
        )

    date = ds.time.dt.strftime(
        "%Y-%m-%d-%H-%M"
    ).values  # keep until you do multiple times
    plt.title(f"{model} {date} Depth Averaged Current Magnitude")

    plt.tight_layout()
    plt.show()

    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()


def plot_threshold(
    ds: xr.Dataset,
    extent: tuple,
    streamlines: bool = False,
    density: int = 4,
    quiver: bool = False,
    scalar: int = 4,
) -> None:
    """
    Plot current thresholds.

    Args:
        ds (xr.Dataset): xarray dataset containing current data.
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        streamlines (bool, optional): Plot streamlines. Defaults to False.
        density (int, optional): Density of streamlines. Defaults to 4.
        quiver (bool, optional): Plot quiver. Defaults to False.
        scalar (int, optional): Scalar for subsampling data. Defaults to 4.

    Returns:
        None
    """
    model = ds.attrs["model"]

    print(f"{model}: Plotting Thresholds...")
    starttime = print_starttime()

    lat_min, lon_min, lat_max, lon_max = extent
    cplt.create(
        [lon_min, lon_max, lat_min, lat_max], gridlines=True, proj=ccrs.PlateCarree()
    )

    if "RTOFS" in ds.attrs["model"]:
        lon2D = ds["lon"]  # RTOFS already has 2D values for lat and lon
        lat2D = ds["lat"]
    else:
        lon2D, lat2D = np.meshgrid(ds.lon, ds.lat)  # create 2D values for lat and lon

    mag = ds["magnitude"]
    max_mag = np.nanmax(mag)
    max_label = f"{max_mag:.2f}"  # converts to string with 2 decimal places

    levels = [0, 0.2, 0.3, 0.4, 0.5, max_mag]
    colors = ["none", "yellow", "orange", "orangered", "maroon"]
    labels = [
        None,
        "0.2 - 0.3 $\mathregular{ms^{-1}}$",
        "0.3 - 0.4 $\mathregular{ms^{-1}}$",
        "0.4 - 0.5 $\mathregular{ms^{-1}}$",
        "0.5 - " + max_label + " $\mathregular{ms^{-1}}$",
    ]

    patches = []
    for color, label in zip(colors, labels):
        if label:
            patches.append(mpatches.Patch(color=color, label=label))

    contourf = plt.contourf(
        lon2D,
        lat2D,
        mag,
        levels=levels,
        extend="max",
        colors=colors,
        transform=ccrs.PlateCarree(),
    )

    if streamlines:
        plot_streamlines(ds, lon2D, lat2D, density=density)
    elif quiver:
        plot_quiver(ds, lon2D, lat2D, scalar=scalar)
    else:
        print(
            '"streamlines" and "quiver" were not set to True. Plotting magnitude only.'
        )

    legend = plt.legend(handles=patches, loc="best")
    legend.set_zorder(100)

    model = ds.attrs["model"]
    date = ds.time.dt.strftime(
        "%Y-%m-%d-%H-%M"
    ).values  # keep until you do multiple times
    plt.title(f"{model} {date} Depth Averaged Current Magnitude Thresholds")

    plt.tight_layout()
    plt.show()

    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()


def plot_rmse(da: xr.DataArray, extent: tuple) -> None:
    """
    Plot Room Mean Square Error (RMSE).

    Args:
        da (xr.DataArray): xarray dataset containing RMSE data.
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.

    Returns:
        None
    """
    model = da.attrs["model"]

    print(f"{model}: Plotting RMSE...")
    starttime = print_starttime()

    lat_min, lon_min, lat_max, lon_max = extent
    cplt.create(
        [lon_min, lon_max, lat_min, lat_max], gridlines=True, proj=ccrs.PlateCarree()
    )

    if "RTOFS" in model:
        lon2D = da["lon"]  # RTOFS already has 2D values for lat and lon
        lat2D = da["lat"]
    else:
        lon2D, lat2D = np.meshgrid(da.lon, da.lat)  # create 2D values for lat and lon

    levels = 10  # for now

    contourf = plt.contourf(
        lon2D,
        lat2D,
        da.values,
        levels=levels,
        cmap=cmo.deep,
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(contourf, label="RMSE ($\mathregular{ms^{-1}}$)")

    date = da.time.dt.strftime("%Y-%m-%d-%H-%M").values  # for now

    plt.title(f"{model} {date} RMSE")

    plt.tight_layout()
    plt.show()

    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()


def plot_mae(da: xr.DataArray, extent: tuple) -> None:
    """
    Plot Mean Absolute Error (MAE).

    Args:
        da (xr.DataArray): xarray dataset containing MAE data.
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.

    Returns:
        None
    """
    model = da.attrs["model"]

    print(f"{model}: Plotting MAE...")
    starttime = print_starttime()

    lat_min, lon_min, lat_max, lon_max = extent
    cplt.create(
        [lon_min, lon_max, lat_min, lat_max], gridlines=True, proj=ccrs.PlateCarree()
    )

    if "RTOFS" in model:
        lon2D = da["lon"]  # RTOFS already has 2D values for lat and lon
        lat2D = da["lat"]
    else:
        lon2D, lat2D = np.meshgrid(da.lon, da.lat)  # create 2D values for lat and lon

    levels = 10  # for now

    contourf = plt.contourf(
        lon2D,
        lat2D,
        da.values,
        levels=levels,
        cmap=cmo.deep,
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(contourf, label="MAE ($\mathregular{ms^{-1}}$)")  # units????

    date = da.time.dt.strftime("%Y-%m-%d-%H-%M").values  # for now

    plt.title(f"{model} {date} MAE")

    plt.tight_layout()
    plt.show()

    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()
