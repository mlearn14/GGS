# Using the Glider Guidance System 2

Last updated for v1.1.0

## Overview

This document will cover how to properly run the Glider Guidance System 2 (GGS2).

## General Data Flow

1. Initialize parameters and models
2. Initialize and load the common grid
3. Load models from respective API's
4. Individual model processing
    - Subset model to desired spatiotemporal bounds
    - Regrid dataset to the common grid for model comparison
    - Interpolate data from the surface to desired maximum depth
    - Calculate the depth average current velocity
    - Calculate magnitude of currents from u and v components
    - Compute the optimal path using the A* algorithm
    - Plot depth averaged current velocity data
5. Model comparison processing
    - Generate all non-repeating combinations of selected model datasets
    - Calculate and plot simple differences
    - Calculate and plot mean of simple differences
    - Calculate and plot simple mean
    - Calculate and plot Root Mean Square (RMS) Profile Difference

## Running the GGS2

As of the current version, GGS2 utilizes .JSON configuration files:

1. `0_dataviz.ipynb`: Jupyter Notebook file. Parameters are set in the first code cell. The notebook can be ran by pressing the "double triangle"/"Run All" button located in the ribbon at the top of the screen.
2. `0_main.py`: Standard python file. Parameters are set in a config .JSON file before being parsed as an argument. Running the python script can be done by running `python 0_main.py "config_fname"` in a Anaconda Powershell terminal when in the same working directory. `"config_name"` should be replaced with the name of the config file you wish to use sans file suffix.

## Parameter Selection

Here is a list of all parameters and what they do:

| Variable | Data Type | Valid Inputs/Format | Notes |
|---|---|---|---|
| MISSION_NAME | str | "mission" | Name of the mission configuration. |
| START_DATE | str | "YYYY-MM-DD", "today", "tomorrow", "yesterday", or `None` | Sample start date |
| END_DATE | str | "YYYY-MM-DD", "today", "tomorrow", "yesterday", or `None` | Sample end date |
| SW_COORD | tuple(float, float) | (-90 - 90, -180 - 180) | Southwest boundary coordinate |
| NE_COORD | tuple(float, float) | (-90 - 90, -180 - 180) | Northeast boundary coordinate |
| MAX_DEPTH | int | 0 - 1000 (recommended) | Maximum working depth (m) of glider model |
| CMEMS | bool | True - False | European Model |
| ESPC | bool | True - False | Navy Model |
| RTOFS_EAST | bool | True - False | NOAA model for US east coast |
| RTOFS_WEST | bool | True - False | NOAA model for US west coast |
| RTOFS_PARALLEL | bool | True - False | Experimental NOAA model for east coast |
| COMPUTE_OPTIMAL_PATH | bool | True - False | Compute the optimal path using an A* algorithm |
| WAYPOINTS | list(tuple) | [(lat1, lon1), ..., (latx, lonx)] | List of coordinates to pass into the A* algorithm. Minimum of 2 points are required |
| GLIDER_RAW_SPEED | float | 0.5 (recommended) | Raw speed of glider model |
| INDIVIDUAL_PLOTS | bool | True - False | Make plots of individual model products |
| SIMPLE_DIFFERENCE | bool | True - False | Make plots of simple differences of each non-repeating pair of models |
| MEAN_DIFFERENCES | bool | True - False | Plot the mean of the differences of each non-repeating pair of models |
| SIMPLE_MEAN | bool | True - False | Plot the simple mean of all selected model combinations |
| RMS_PROFILE_DIFFERENCE | bool | True - False | Plot the root mean square profile difference between all selected model combinations |
| PLOT_MAGNITUDES | bool | True - False | Make a plot of depth averaged current magnitude contours |
| PLOT_MAGNITUDE_THRESHOLDS | bool | True - False | Make a plot of depth averaged current magnitude threshold contours |
| VECTOR_TYPE | string | "quiver", "streamplot", None | Determines the vector type of the plot to show current direction |
| STREAMLINE_DENSITY | int | --- | Density of streamlines. Higher number = denser streamlines |
| QUIVER_DOWNSCALING | int | --- | Downsampling scalar for quiver plots. Higher number = Less quivers |
| SAVE_FIGURES | bool | True - False | Save figures locally in the GGS/figures directory |
| save_config | bool | True - False | Save current options as a JSON file |
| config_directory | str | "relative_path" | Directory that the config file will save to |
| load_config | bool | True - False | Load locally stored config file |
| config_name | str | "filename" | File name of the locally stored config file |

__WARNING__: Failure to adhere to variable formatting will result in unexpected errors.
