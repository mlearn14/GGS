# author: matthew learn (matt.learn@marine.rutgers.edu)

import cartopy.crs as ccrs
import cmocean.cm as cmo
import cool_maps.plot as cplt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import xarray as xr

import datetime as dt
import os

from .util import print_starttime, print_endtime, print_runtime, save_fig

# Set Seaborn style
sns.set_style("whitegrid")
# sns.set_context("poster")


### Saving functions ###
def generate_map_filename(
    mission_name: str,
    ddate: str,
    fdate: str,
    plot_type: str,
    vector_type: str,
    model_name: str,
    comp_plot: bool = False,
    output_dir: str = "products",
) -> str:
    """
    Generate a standardized filename for saving figures.

    Args:
    ----------
        - mission_name (str): The name of the mission.
        - ddate (str): The directory date in YYYY_MM_DD format.
        - fdate (str): The file date in YYYYMMDDHH format.
        - figure_type (str): Type of figure (e.g., 'magnitude', 'threshold', 'rmsd').
        - plot_type (str): Type of plot (e.g., 'streamplot', 'quiverplot', 'none').
        - model_names (list): Model names(s) (e.g., 'RTOFS', 'CMEMS', 'RTOFS+CMEMS').
        - comp_plot (bool, optional): Whether this is a comparison plot. Default is False.
        - output_dir (str, optional): Directory where the file will be saved. Default is 'figures'.

    Returns:
    ----------
        - filename (str): Full path for the output file.
    """
    # Ensure the output directory exists
    if comp_plot:
        output_dir = f"{output_dir}/{ddate}/comparisons"
    else:
        output_dir = f"{output_dir}/{ddate}"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the file name
    filename = f"{mission_name}_{fdate}_{model_name}_{plot_type}_{vector_type}.png"

    # Combine directory and file name
    return os.path.join(output_dir, filename)


### Plotting initialization ###
def initialize_map(
    extent: tuple, figsize: tuple = (12, 8), projection: ccrs = ccrs.Mercator()
) -> tuple:
    """
    Initializes the base map figure with static elements.

    Args:
    -----------
        - extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        - figsize (tuple, optional): Figure size. Defaults to (12, 8).
        - projection (ccrs, optional): Projection. Defaults to ccrs.Mercator().

    Returns:
    -----------
        - fig (object): Figure object.
        - ax (object): Axis object.
    """
    lat_min, lon_min, lat_max, lon_max = extent

    fig = plt.figure(figsize=(12, 8))
    # leave some space for colorbar
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=projection)

    cplt.create(
        [lon_min, lon_max, lat_min, lat_max],
        gridlines=True,
        ax=ax,
        oceancolor="none",
        proj=projection,
    )

    return fig, ax


def clear_map(
    fig: object,
    total_flag: bool = False,
    im: object = None,
    legend: object = None,
    cax=None,
    quiver: object = None,
) -> None:
    """
    Clears the figure of all elements, leavning only the base map.

    Args:
    -----------
        - fig (object): Figure object.
        - ax (object): Axis object.
        - total_flag (bool): Flag for clearing all elements.
        - im (object): Contourf plot object.
        - quiver (object, optional): Quiver plot object. Defaults to None.
        - cax (object, optional): Colorbar object. Defaults to None.
        - legend (object, optional): Legend object. Defaults to None.

    Returns:
    -----------
        - `None`
    """

    if total_flag:
        fig.clear()

        return

    fig.texts = []

    im.remove()

    if quiver is not None:
        quiver.remove()

    if cax is not None:
        cax.remove()

    if legend is not None:
        legend.remove()

    return


### Plotting helper functions ###
def add_text(
    fig: object, ax: object, plot_title: str, date: dt.datetime, text_name: str
) -> None:
    """
    Adds text to figures.

    Args
    ----------
        fig (object): Figure object
        ax (object): Axis object
        plot_title (str): Specific title for the plot.
        date (datetime): Date of data selection in MM-DD-YYYY HH:MM
        text_name (str): Properly formatted model name.

    Returns
    ---------
        - `None`
    """
    # Code adapted from Sal
    ax_pos = ax.get_position()
    top = ax_pos.y1
    bottom = ax_pos.y0

    title_distance_top = 0.15
    title_position = top + title_distance_top

    subtitle_distance_title = 0.05
    subtitle_position = title_position - subtitle_distance_title

    tag_distance_bottom = 0.1
    tag_position = bottom - tag_distance_bottom

    fig.text(
        0.5,
        title_position,
        plot_title,
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    subtitle_text = f"{date} UTC - {text_name}"
    fig.text(
        0.5, subtitle_position, subtitle_text, fontsize=18, ha="center", va="bottom"
    )

    tag_text = f"Generated by the Glider Guidance System 2 (GGS2)"
    fig.text(
        0.5,
        tag_position,
        tag_text,
        fontsize=16,
        ha="center",
        va="top",
        color="gray",
    )

    return


def create_cbar(fig: object, ax: object, im: object, label: str) -> object:
    """
    Creates a colorbar. Colorbar is placed to the right of the plot on its own dedicated ax.

    Args
    -----------
        - fig (object): Figure object.
        - ax (object): Axis object.
        - im (object): Image object.
        - label (str): Label for colorbar.

    Returns
    -----------
        - cax (object): Colorbar object.
    """
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax, format="%.2f", label=label)
    plt.colorbar(im, cax=cax, format="%.2f", label=label)

    # return the cax object instead of the colorbar object because matplotlib is weird and gets mad when I do cbar.remove()
    return cax


def create_thresholds_levels_legend(ax: object, magnitude: xr.DataArray) -> tuple:
    """
    Creates a legend for the threshold contours.

    Args:
    -----------
        - ax (object): Axis object.
        - magnitude (xr.DataArray): Magnitude data.

    Returns:
    -----------
        - levels (np.ndarray): Contour levels.
        - colors (list): List of colors.
        - legend (object): Legend object.
    """
    # code adapted from Sal's code and consolidated with the help of ChatGPT and Codeium
    max_mag = np.nanmax(magnitude)
    max_label = f"{max_mag:.2f}"  # converts to string with 2 decimal places
    mags = [0, 0.2, 0.3, 0.4, 0.5]  # hardcoded for now
    colors = ["none", "yellow", "orange", "orangered", "maroon", "maroon"]
    levels = []
    labels = []

    for i, mag_value in enumerate(mags):
        if mag_value < max_mag:
            levels.append(mag_value)
        if i > 0 and mag_value <= max_mag:
            # this is so hacky but it works ({}{}{{{{how bout 'em?}}}})
            labels.append(rf"{mags[i-1]} - {mag_value} $\mathregular{{ms^{{-1}}}}$")
    if max_mag > mags[-1]:
        levels.append(max_mag)
        labels.append(rf"{mags[-1]} - {max_label} $\mathregular{{ms^{{-1}}}}$")

    colors = colors[: len(levels)]

    patches = []
    for color, label in zip(colors, labels):
        if label:
            patches.append(mpatches.Patch(color=color, label=label))

    labels = labels[1:]
    patches = patches[1:]

    legend = ax.legend(handles=patches)
    legend.set_zorder(100)

    return levels, colors, legend


def create_glider_path(ax: object, path: list, waypoints: list, color="k") -> tuple:
    """
    Creates a scatter plot of the glider path.

    Args
    -----------
        ax (object): Axis object.
        path (list): List of tuples of glider positions.
        waypoints (list): List of tuples of waypoints.

    Returns
    -----------
        path_plot (object)
            Path plot object.
        wp_plot (object)
            Waypoint plot object.
    """

    def format_coords(coords: list) -> tuple:
        """
        Formats coordinates into a tuple of latitudes and longitudes.

        Args:
        -----------
            - coords (list): List of coordinates as tuples: (lat, lon).

        Returns:
        -----------
            - lat (list): List of latitudes.
            - lon (list): List of longitudes.
        """
        lat = [coord[0] for coord in coords]
        lon = [coord[1] for coord in coords]
        return lat, lon

    # separate lons and lats
    wp_lats, wp_lons = format_coords(waypoints)
    path_lats, path_lons = format_coords(path)

    # create plots of glider path, with waypoints plotted over in red
    underlime = ax.plot(
        path_lons,
        path_lats,
        linestyle="-",
        marker="o",
        color="white",
        markersize=8,
        linewidth=6,
        transform=ccrs.Geodetic(),
        zorder=49,
    )

    path_plot = ax.plot(
        path_lons,
        path_lats,
        linestyle="-",
        marker="o",
        color=color,
        markersize=4,
        transform=ccrs.Geodetic(),
        zorder=50,
    )

    start_point = ax.scatter(
        wp_lons[0],
        wp_lats[0],
        color="green",
        edgecolors="black",
        marker="o",
        transform=ccrs.PlateCarree(),
        zorder=52,
    )
    wp_plot = ax.scatter(
        wp_lons,
        wp_lats,
        color="purple",
        edgecolors="black",
        marker="o",
        transform=ccrs.PlateCarree(),
        zorder=51,
    )
    end_point = ax.scatter(
        wp_lons[-1],
        wp_lats[-1],
        color="red",
        edgecolors="black",
        marker="o",
        transform=ccrs.PlateCarree(),
        zorder=52,
    )

    return path_plot, wp_plot


def create_quiverplot(
    ax: object,
    ds: xr.Dataset,
    lon2D: np.ndarray,
    lat2D: np.ndarray,
    scalar: int,
    **kwargs,
) -> object:
    """
    Creates a quiver plot.

    Args:
    -----------
        - ax (object): Axis object.
        - ds (xr.Dataset): xarray dataset containing current data.
        - lon2D (np.ndarray): 2D array of longitudes.
        - lat2D (np.ndarray): 2D array of latitudes.
        - scalar (int): Scalar for subsampling data. A larger scalar results in quivers that are less dense.

    Returns:
    -----------
        - quiver (object): Quiver plot object.

    Other Parameters:
    -----------
        - **kwargs: Additional keyword arguments to pass to the quiver function.
    """
    u: xr.DataArray = ds["u"]
    v: xr.DataArray = ds["v"]

    # subsample data to clean plot up
    u_sub = u[::scalar, ::scalar]
    v_sub = v[::scalar, ::scalar]
    lon_sub = lon2D[::scalar, ::scalar]
    lat_sub = lat2D[::scalar, ::scalar]

    u_sub = u_sub.values
    v_sub = v_sub.values

    # create quiver plot
    quiver = ax.quiver(
        lon_sub,
        lat_sub,
        u_sub,
        v_sub,
        color="black",
        transform=ccrs.PlateCarree(),
        **kwargs,
    )

    return quiver


def create_streamplot(
    ax: object,
    ds: xr.Dataset,
    lon2D: np.ndarray,
    lat2D: np.ndarray,
    density: int = 4,
    **kwargs,
) -> object:
    """
    Creates a streamplot.

    Args:
    -----------
        - ax (object): Axis object.
        - ds (xr.Dataset): xarray dataset containing current data.
        - lon2D (np.ndarray): 2D array of longitudes.
        - lat2D (np.ndarray): 2D array of latitudes.
        - density (int): Density of streamlines. Defaults to 4.
        - linewidth (float): Width of streamlines. Defaults to 0.5.
        - arrowsize (float): Size of arrows. Defaults to 0.5.

    Returns:
    -----------
        - streamplot (object): Streamplot object.

    Other Parameters:
    -----------
        - **kwargs: Additional keyword arguments for `ax.streamplot`. See the documentation for StreamplotSet.
    """
    u: xr.DataArray = ds["u"]
    v: xr.DataArray = ds["v"]

    streamplot = ax.streamplot(
        lon2D,
        lat2D,
        u,
        v,
        density=density,
        color="black",
        transform=ccrs.PlateCarree(),
        **kwargs,
    )

    return streamplot


# Plotting function
def populate_map(
    contour_type: str,
    vector_type: str,
    fig: object,
    ax: object,
    data,
    density: int = 5,
    scalar: int = 4,
    optimal_path: list = None,
    waypoints: list = None,
    **kwargs,
) -> tuple:
    """
    Populates a figure with contours and/or vectors.

    Args
    -----------
        contour_type (str): Type of contour. Options: 'magnitude', 'threshold', 'speed_diff', 'u_diff', 'v_diff', 'mean_diff', 'rmsd', 'mean_magnitude', & 'mean_threshold'
        vector_type (str): Type of vector (e.g., 'quiver', 'streamplot', `None`). NOTE: RTOFS must be regridded prior to being passed to this function for quiver to work.
        fig (object): Figure object.
        ax (object): Axis object.
        data (Dataset): Data to be plotted.
        density (int): Density of streamlines. Defaults to 5.
        scalar (int): Scalar for subsampling data. Defaults to 4.
        optimal_path (list): List of optimal path.
        waypoints (list): List of waypoints.

    Returns
    -----------
        contourf (object): Contourf object.
        legend (object): Legend object.
        cax (object): Colorbar object.
        quiver (object): Quiver object.
        streamplot (object): Streamplot object.
        path_plot (object): Optimized path plot object.
        waypoint_plot (object): Waypoint plot object.

    Other Parameters
    ----------
        **kwargs: Additional keyword arguments to pass to the plotting functions.
    """
    contourf = None
    legend = None
    cax = None
    quiver = None
    streamplot = None

    text_name = data.attrs["text_name"]
    lon2D, lat2D = np.meshgrid(data.lon, data.lat)

    # Create contours.
    if contour_type == "magnitude":
        plot_title = "Depth Averaged Current Magnitudes"
        label = r"Magnitude ($\mathregular{ms^{-1}}$)"
        levels = np.linspace(0, 0.9, 10)

        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.magnitude,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="max",
            cmap=cmo.speed,
        )
        cax = create_cbar(fig, ax, contourf, label)

    elif contour_type == "threshold":
        plot_title = "Depth Averaged Current Magnitude Thresholds"
        levels, colors, legend = create_thresholds_levels_legend(ax, data.magnitude)
        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.magnitude,
            transform=ccrs.PlateCarree(),
            levels=levels,
            colors=colors,
        )

    elif contour_type == "mean_diff":
        plot_title = "Depth Averaged Current Mean Differences"
        label = r"Mean Difference ($\mathregular{ms^{-1}}$)"
        levels = np.linspace(0, 0.9, 10)

        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.magnitude,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="max",
            cmap=cmo.amp,
        )
        cax = create_cbar(fig, ax, contourf, label)

    elif contour_type == "mean_magnitude":
        plot_title = "Depth Averaged Current Means"
        label = r"Simple Mean ($\mathregular{ms^{-1}}$)"
        levels = np.linspace(0, 0.9, 10)

        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.magnitude,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="max",
            cmap=cmo.speed,
        )
        cax = create_cbar(fig, ax, contourf, label)

    elif contour_type == "mean_threshold":
        plot_title = "Depth Averaged Current Mean Thresholds"
        levels, colors, legend = create_thresholds_levels_legend(ax, data.magnitude)
        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.magnitude,
            transform=ccrs.PlateCarree(),
            levels=levels,
            colors=colors,
        )

    elif contour_type == "rmsd_vertical":
        plot_title = (
            "Depth Averaged Root Mean Square Vertical Differences (RMSD Vertical)"
        )
        label = r"RMSD ($\mathregular{ms^{-1}}$)"
        levels = np.linspace(0, 0.9, 10)

        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.magnitude,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="max",
            cmap=cmo.deep,
        )
        cax = create_cbar(fig, ax, contourf, label)

    elif contour_type == "speed_diff":
        plot_title = "Depth Averaged Current Speed Differences"
        label = r"Difference ($\mathregular{ms^{-1}}$)"

        min_mag = np.nanmin(data.magnitude.values)
        max_mag = np.nanmax(data.magnitude.values)

        # Case 1: If all values are negative, set vmin and vmax to match the negative range
        if max_mag <= 0:
            vmin, vmax = min_mag, 1e-6

        # Case 2: If all values are positive, set vmin and vmax to match the positive range
        elif min_mag >= 0:
            vmin, vmax = -1e-6, max_mag

        # Case 3: Mixed positive and negative values (default behavior)
        else:
            max_abs_diff = max(abs(min_mag), abs(max_mag))
            vmin, vmax = -max_abs_diff, max_abs_diff

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        levels = np.linspace(vmin, vmax, 11)

        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.magnitude,
            transform=ccrs.PlateCarree(),
            norm=norm,
            levels=levels,
            extend="both",
            cmap=cmo.balance,
        )

        cax = create_cbar(fig, ax, contourf, label)
        ax_pos = ax.get_position()
        top = ax_pos.y1
        eq_position = top + 0.05
        fig.text(
            0.5,
            eq_position,
            f"Difference = {data.attrs['model1_name']} - {data.attrs['model2_name']}",
            fontsize=15,
            ha="center",
            va="center",
        )

    elif contour_type == "u_diff":
        plot_title = "Depth Averaged Current Eastward Velocity Differences"
        label = r"u Difference ($\mathregular{ms^{-1}}$)"

        min_u = np.nanmin(data.u.values)
        max_u = np.nanmax(data.u.values)

        # Case 1: If all values are negative, set vmin and vmax to match the negative range
        if max_u <= 0:
            vmin, vmax = min_u, 1e-6

        # Case 2: If all values are positive, set vmin and vmax to match the positive range
        elif min_u >= 0:
            vmin, vmax = -1e-6, max_u

        # Case 3: Mixed positive and negative values (default behavior)
        else:
            max_abs_diff = max(abs(min_u), abs(max_u))
            vmin, vmax = -max_abs_diff, max_abs_diff

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        levels = np.linspace(vmin, vmax, 11)

        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.u,
            transform=ccrs.PlateCarree(),
            norm=norm,
            levels=levels,
            extend="both",
            cmap=cmo.balance,
        )

        cax = create_cbar(fig, ax, contourf, label)
        ax_pos = ax.get_position()
        top = ax_pos.y1
        eq_position = top + 0.05
        fig.text(
            0.5,
            eq_position,
            f"Eastward Difference = {data.attrs['model1_name']} - {data.attrs['model2_name']}",
            fontsize=15,
            ha="center",
            va="center",
        )

    elif contour_type == "v_diff":
        plot_title = "Depth Averaged Current Northward Velocity Differences"
        label = r"v Difference ($\mathregular{ms^{-1}}$)"

        min_v = np.nanmin(data.v.values)
        max_v = np.nanmax(data.v.values)

        # Case 1: If all values are negative, set vmin and vmax to match the negative range
        if max_v <= 0:
            vmin, vmax = min_v, 1e-6

        # Case 2: If all values are positive, set vmin and vmax to match the positive range
        elif min_v >= 0:
            vmin, vmax = -1e-6, max_v

        # Case 3: Mixed positive and negative values (default behavior)
        else:
            max_abs_diff = max(abs(min_v), abs(max_v))
            vmin, vmax = -max_abs_diff, max_abs_diff

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        levels = np.linspace(vmin, vmax, 11)

        contourf = ax.contourf(
            lon2D,
            lat2D,
            data.v,
            transform=ccrs.PlateCarree(),
            norm=norm,
            levels=levels,
            extend="both",
            cmap=cmo.balance,
        )

        cax = create_cbar(fig, ax, contourf, label)
        ax_pos = ax.get_position()
        top = ax_pos.y1
        eq_position = top + 0.05
        fig.text(
            0.5,
            eq_position,
            f"Northward Difference = {data.attrs['model1_name']} - {data.attrs['model2_name']}",
            fontsize=15,
            ha="center",
            va="center",
        )

    # Create vectors
    if vector_type == "quiver":
        quiver = create_quiverplot(ax, data, lon2D, lat2D, scalar=scalar, **kwargs)
    elif vector_type == "streamplot":
        streamplot = create_streamplot(
            ax, data, lon2D, lat2D, density=density, **kwargs
        )
    elif vector_type == None:
        pass
    else:
        raise ValueError(
            f"Invalid vector type: {vector_type}. Please choose 'quiver', 'streamplot', or None."
        )

    # Add optimized glider path
    if optimal_path and waypoints is not None:
        path_plot, wp_plot = create_glider_path(ax, optimal_path, waypoints)
    else:
        path_plot = None
        wp_plot = None

    # Add plot title and text
    date = data.time.dt.strftime("%Y-%m-%d %H:%M").values
    add_text(fig, ax, plot_title, date, text_name)

    return contourf, legend, cax, quiver, streamplot, path_plot, wp_plot


# Smart plotting functions
def create_map(
    data: xr.Dataset,
    extent: tuple,
    contour_type: str,
    vector_type: str,
    density: int = 4,
    scalar: int = 4,
    optimal_path: list = None,
    waypoints: list = None,
    initialize: bool = True,
    mission_fname: str = None,
    comp_plot: bool = False,
    save: bool = False,
    diag_text: bool = True,
    **kwargs,
) -> tuple:
    """
    Creates a map.

    Args
    ----------
        data (xr.Dataset): Data to be plotted.
        extent (tuple): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        contour_type (str): Type of contour. Options are: 'magnitude', 'threshold', 'speed_diff', 'u_diff', 'v_diff', 'mean_diff', 'mean_magnitude', 'mean_threshold', & 'rmsd_profile'
        vector_type (str): Type of vector (e.g., 'quiver', 'streamplot', `None`).
        density (int): Density of streamlines. Defaults to 5.
        scalar (int): Scalar for subsampling data. Defaults to 4.
        optimal_path (list): List of optimized glider paths (e.g., [path1, path2, ...]).
        waypoints (list): List of waypoints (e.g., [waypoint1, waypoint2, ...]).
        initialize (bool): Determines if a new figure will be initialized. Defaults to False.
        mission_fname (str): Name of the mission formatted for the file download.
        comp_plot (bool): Determines if a comparison plot will be created. Defaults to False.
        save (bool): Determines if the figure will be saved to a file. Defaults to False. If set to True, figures will be saved to the /figures directory.
        diag_text (bool): Determines if diagnostic text will be printed. Defaults to True.

    Returns
    ----------
        fig (object): Figure object.
        ax (object): Axis object.
        contourf (object): Contourf object.
        legend (object): Legend object.
        cax (object): Colorbar object.
        quiver (object): Quiver object.
        streamplot (object): Streamplot object.
        path_plot (object): Optimized path plot object.
        waypoint_plot (object): Waypoint plot object.

    Other Parameters
    ----------
        **kwargs: Additional keyword arguments to pass to the plotting functions.
    """
    text_name = data.attrs["text_name"]

    if vector_type == "quiver":
        vector_text = "quiver"
    elif vector_type == "streamplot":
        vector_text = "streamline"
    elif vector_type == None:
        vector_text = "vectorless"

    if contour_type == "threshold":
        contour_text = "magnitude threshold"
    elif contour_type == "rmsd_verticle":
        contour_text = "RMS Difference"
    elif contour_type == "mean_diff":
        contour_text = "mean difference"
    elif contour_type == "mean_magnitude":
        contour_text = "simple mean magnitude"
    elif contour_type == "mean_threshold":
        contour_text = "simple mean magnitude threshold"
    elif contour_type == "speed_diff":
        contour_text = "speed difference"
    elif contour_type == "u_diff":
        contour_text = "eastward velocity difference"
    elif contour_type == "v_diff":
        contour_text = "northward velocity difference"
    else:
        contour_text = contour_type

    if diag_text:
        print(
            f"{text_name}: Creating {vector_text} plot of depth averaged current {contour_text}s..."
        )
        starttime = print_starttime()

    if initialize:
        fig, ax = initialize_map(extent)
    else:
        fig = plt.gcf()
        ax = fig.get_axes()[0]

    contourf, legend, cax, quiver, streamplot, path_plot, wp_plot = populate_map(
        contour_type,
        vector_type,
        fig,
        ax,
        data,
        density=density,
        scalar=scalar,
        optimal_path=optimal_path,
        waypoints=waypoints,
        linewidth=0.5,
        **kwargs,
    )

    if save:
        model_name = data.attrs["model_name"]
        ddate = data.time.dt.strftime("%Y_%m_%d").values
        fdate = data.time.dt.strftime("%Y%m%d%H").values
        filename = generate_map_filename(
            mission_fname,
            ddate,
            fdate,
            contour_type,
            vector_type,
            model_name,
            comp_plot,
        )
        save_fig(fig, filename)

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return fig, ax, contourf, legend, cax, quiver, streamplot, path_plot, wp_plot
