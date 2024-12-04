# author: matthew learn (matt.learn@marine.rutgers.edu)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import cool_maps.plot as cplt

from functions import *


def save_fig(fig: object, name: str) -> None:
    # TODO: update to match new directory structure
    fname: str = f"plots/{name}.png"
    print(f"Saving figure to {fname}")
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    print("Saved.")


def plot_streamlines(
    ax: object, ds: xr.Dataset, lon: np.ndarray, lat: np.ndarray, density: int = 4
) -> object:
    """
    Plot current streamlines.

    Args:
        ax (object): Axis object.
        ds (xr.Dataset): xarray dataset containing current data.
        lon (np.ndarray): 2D array of longitudes.
        lat (np.ndarray): 2D array of latitudes.
        density (int, optional): Density of streamlines. Defaults to 4.

    Returns:
        streamplot (object): Streamplot object.
    """
    u: xr.DataArray = ds["u"]
    v: xr.DataArray = ds["v"]
    # mag = ds["magnitude"]  # might not keep!

    streamplot: object = ax.streamplot(
        lon,
        lat,
        u,
        v,
        color="black",
        linewidth=0.5,
        arrowsize=0.5,
        density=density,
        transform=ccrs.PlateCarree(),
    )

    return streamplot


def plot_quiver(
    ax: object, ds: xr.Dataset, lon: np.ndarray, lat: np.ndarray, scalar: int = 2
) -> object:
    """
    Plot current quiver.

    Args:
        ax (object): Axis object.
        ds (xr.Dataset): xarray dataset containing current data.
        lon (np.ndarray): 2D array of longitudes.
        lat (np.ndarray): 2D array of latitudes.
        scalar (int, optional): Scalar for subsampling data. Defaults to 2.

    Returns:
        quiver (object): Quiver plot object.
    """
    u: xr.DataArray = ds["u"]
    v: xr.DataArray = ds["v"]

    # subsample data to clean plot up
    u_sub = u[::scalar, ::scalar]
    v_sub = v[::scalar, ::scalar]
    lon_sub = lon[::scalar, ::scalar]
    lat_sub = lat[::scalar, ::scalar]

    quiver = ax.quiver(
        lon_sub,
        lat_sub,
        u_sub,
        v_sub,
        transform=ccrs.PlateCarree(),
    )

    return quiver


def format_plot(
    extent: tuple, figsize: tuple = (12, 8), projection: ccrs = ccrs.PlateCarree()
) -> tuple:
    """
    Format plot.

    Args:
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        projection (ccrs, optional): Projection. Defaults to ccrs.PlateCarree().

    Returns:
        tuple: Tuple containing:
        - fig (object): Figure object.
        - ax (object): Axis object.
    """
    lat_min, lon_min, lat_max, lon_max = extent

    fig = plt.figure(figsize=figsize)
    # leave some space for colorbar
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=projection)

    cplt.create(
        [lon_min, lon_max, lat_min, lat_max],
        gridlines=True,
        ax=ax,
        proj=ccrs.PlateCarree(),
    )

    return fig, ax


def format_cbar(fig: object, ax: object, im: object, label: str) -> object:
    """
    Fomart colorbar depending on figure size.

    Args:
        fig (object): Figure object.
        ax (object): Axis object.
        im (object): Image object.
        label (str): Label for colorbar.

    Returns:
        cax (object): Colorbar object.
    """
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax, label=label)
    return cax


def plot_magnitude(
    ds: xr.Dataset,
    extent: tuple,
    streamlines: bool = False,
    density: int = 4,
    quiver: bool = False,
    scalar: int = 4,
    extend: str = "max",
    figsize: tuple = (12, 8),
    projection=ccrs.PlateCarree(),
    savefig: bool = False,
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
        extend (str, optional): Extend colorbar. Defaults to "max". Takes "min", "max", "both", or "neither".
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        projection (ccrs, optional): Projection. Defaults to ccrs.PlateCarree().
        savefig (bool, optional): Save figure. Defaults to False.

    Returns:
        None
    """
    model = ds.attrs["model"]

    print(f"{model}: Plotting magnitudes...")
    starttime = print_starttime()

    # Initialize plot
    fig, ax = format_plot(extent, figsize=figsize, projection=projection)

    # Create 2D values for lat and lon
    if "RTOFS" in model:
        lon2D = ds["lon"]  # RTOFS already has 2D values for lat and lon
        lat2D = ds["lat"]
    else:
        lon2D, lat2D = np.meshgrid(ds.lon, ds.lat)  # create 2D values for lat and lon

    levels = np.linspace(0, 0.9, 10)  # hardcoded for now, but should be dynamic?

    # Plot
    contourf = ax.contourf(
        lon2D,
        lat2D,
        ds["magnitude"],
        levels=levels,
        extend=extend,
        cmap=cmo.speed,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=0.9,
    )

    # Add colorbar
    label = r"velocity ($\mathregular{ms^{-1}}$)"
    cbar = format_cbar(fig, ax, contourf, label)

    # Plot streamlines or quiver
    if streamlines:
        plot_streamlines(ax, ds, lon2D, lat2D, density=density)
        plot_type = "_str"
    elif quiver:
        plot_quiver(ax, ds, lon2D, lat2D, scalar=scalar)
        plot_type = "_qvr"
    else:
        print(
            '"streamlines" and "quiver" were not set to True. Plotting magnitude only.'
        )
        plot_type = None

    date = ds.time.dt.strftime("%Y-%m-%d-%H-%M").values
    #  ^^^ keep until you do multiple times ^^^
    ax.set_title(f"{model} {date} Depth Averaged Current Magnitude")

    plt.tight_layout()

    if savefig:
        save_fig(fig, f"{model}_magnitude{plot_type}")

    # plt.show()

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
    figsize: tuple = (12, 8),
    projection=ccrs.PlateCarree(),
    savefig: bool = False,
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
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        projection (ccrs, optional): Projection. Defaults to ccrs.PlateCarree().
        savefig (bool, optional): Save figure. Defaults to False.

    Returns:
        None
    """
    model = ds.attrs["model"]

    print(f"{model}: Plotting Thresholds...")
    starttime = print_starttime()

    # Initialize plot
    fig, ax = format_plot(extent, figsize=figsize, projection=projection)

    # Create 2D values for lat and lon
    if "RTOFS" in ds.attrs["model"]:
        lon2D = ds["lon"]  # RTOFS already has 2D values for lat and lon
        lat2D = ds["lat"]
    else:
        lon2D, lat2D = np.meshgrid(ds.lon, ds.lat)  # create 2D values for lat and lon

    # Generate levels and labels. Dynamic with magnitude data.
    mag = ds["magnitude"]
    max_mag = np.nanmax(mag)
    max_label = f"{max_mag:.2f}"  # converts to string with 2 decimal places
    mags = [0, 0.2, 0.3, 0.4, 0.5]  # hardcoded for now
    colors = ["none", "yellow", "orange", "orangered", "maroon", "maroon"]
    levels = []
    labels = []

    for i, mag_value in enumerate(mags):
        if mag_value < max_mag:
            levels.append(mag_value)
        if i > 0 and mag_value <= max_mag:
            labels.append(rf"{mags[i-1]} - {mag_value} $\mathregular{{ms^{{-1}}}}$")
    if max_mag > mags[-1]:
        levels.append(max_mag)
        labels.append(rf"{mags[-1]} - {max_label} $\mathregular{{ms^{{-1}}}}$")  # lol

    colors = colors[: len(levels)]

    patches = []
    for color, label in zip(colors, labels):
        if label:
            patches.append(mpatches.Patch(color=color, label=label))

    labels = labels[1:]
    patches = patches[1:]

    contourf = ax.contourf(
        lon2D,
        lat2D,
        mag,
        levels=levels,
        extend="max",
        colors=colors,
        transform=ccrs.PlateCarree(),
    )

    if streamlines:
        plot_streamlines(ax, ds, lon2D, lat2D, density=density)
        plot_type = "_str"
    elif quiver:
        plot_quiver(ax, ds, lon2D, lat2D, scalar=scalar)
        plot_type = "_qvr"
    else:
        print(
            '"streamlines" and "quiver" were not set to True. Plotting magnitude only.'
        )
        plot_type = None

    legend = plt.legend(handles=patches, loc="best")
    legend.set_zorder(100)

    model = ds.attrs["model"]
    date = ds.time.dt.strftime("%Y-%m-%d-%H-%M").values
    # ^^^ keep until you do multiple times ^^^
    ax.set_title(f"{model} {date} Depth Averaged Current Magnitude Thresholds")

    plt.tight_layout()

    if savefig:
        save_fig(fig, f"{model}_threshold{plot_type}")

    # plt.show()

    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()


def plot_rmsd(
    da: xr.DataArray,
    extent: tuple,
    figsize: tuple = (12, 8),
    projection=ccrs.PlateCarree(),
    savefig: bool = False,
) -> None:
    """
    Plot Room Mean Square Difference (RMSD).

    Args:
        da (xr.DataArray): xarray dataset containing RMSD data.
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        projection (ccrs, optional): Projection. Defaults to ccrs.PlateCarree().
        savefig (bool, optional): Save figure. Defaults to False.

    Returns:
        None
    """
    model = da.attrs["model"]

    print(f"{model}: Plotting RMSD...")
    starttime = print_starttime()

    # Initialize plot
    fig, ax = format_plot(extent, figsize=figsize, projection=projection)

    # Create 2D values for lat and lon
    if "RTOFS (East Coast)" and "RTOFS (Parallel)" in model:
        # RTOFS already has 2D values for lat and lon
        # Only applicable to RTOFS-RTOFS RMSDs.
        lon2D = da["lon"]
        lat2D = da["lat"]
    else:
        lon2D, lat2D = np.meshgrid(da.lon, da.lat)

    levels = 10  # for now

    # Plot
    contourf = ax.contourf(
        lon2D,
        lat2D,
        da.values,
        levels=levels,
        cmap=cmo.deep,
        transform=ccrs.PlateCarree(),
    )

    # Add colorbar
    label = r"RMSD ($\mathregular{ms^{-1}}$)"
    cbar = format_cbar(fig, ax, contourf, label)

    date = da.time.dt.strftime("%Y-%m-%d-%H-%M").values  # for now

    ax.set_title(f"{model} {date} Root Mean Square Difference (RMSD)")

    plt.tight_layout()

    if savefig:
        save_fig(fig, f"{model}_rmsd")

    # plt.show()

    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()


def plot_mad(
    da: xr.DataArray,
    extent: tuple,
    figsize: tuple = (12, 8),
    projection=ccrs.PlateCarree(),
    savefig: bool = False,
) -> None:
    """
    Plot Mean Absolute Difference (MAD).

    Args:
        da (xr.DataArray): xarray dataset containing MAD data.
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        projection (ccrs, optional): Projection. Defaults to ccrs.PlateCarree().
        savefig (bool, optional): Save figure. Defaults to False.

    Returns:
        None
    """
    model = da.attrs["model"]

    print(f"{model}: Plotting MAD...")
    starttime = print_starttime()

    # Initialize plot
    fig, ax = format_plot(extent, figsize=figsize, projection=projection)

    # Create 2D values for lat and lon
    if "RTOFS (East Coast)" and "RTOFS (Parallel)" in model:
        # RTOFS already has 2D values for lat and lon
        # Only applicable to RTOFS-RTOFS RMSDs.
        lon2D = da["lon"]
        lat2D = da["lat"]
    else:
        lon2D, lat2D = np.meshgrid(da.lon, da.lat)

    levels = 10  # for now

    # Plot
    contourf = plt.contourf(
        lon2D,
        lat2D,
        da.values,
        levels=levels,
        cmap=cmo.deep,
        transform=ccrs.PlateCarree(),
    )

    # Add colorbar
    label = r"MAD ($\mathregular{ms^{-1}}$)"
    cbar = format_cbar(fig, ax, contourf, label)

    date = da.time.dt.strftime("%Y-%m-%d-%H-%M").values  # for now

    ax.set_title(f"{model} {date} Mean Absolute Difference (MAD)")

    plt.tight_layout()

    if savefig:
        save_fig(fig, f"{model}_mad")

    # plt.show()

    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()
