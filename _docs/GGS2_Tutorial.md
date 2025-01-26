# Using the Glider Guidance System 2

Last updated for v1.0.0.

## Overview

This document will cover how to properly run the Glider Guidance System 2 (GGS2).

## General Data Flow

1. Initialize parameters and models
2. Initialize and load the common grid
3. Individual model processing
    - Load model from its repsective API
    - Subset model to desire parameters
    - Interpolate data from the surface to desired maximum depth
    - Calculate magnitude of currents from u and v components
    - Calculate the depth average current velocity
    - Regrid dataset to the common grid for model comparison
    - Compute the optimal path using the A* algorithm
    - Plot depth averaged current velocity data
4. Model comparison processing
    - Root Mean Square Difference (RMSD) Processing
        - Generate all not repeating combinations of selected model datasets
        - Calculate RMSD
        - Regrid RMSD data to common grid
        - Plot RMSD data

## Running the GGS2

As of the current version there are two ways to run the GGS2:

1. `0_main.py`: Standard python file. Parameters may be edited at the start of the script. Running the python script can be done by pressing the "play" button (VS Code) or by running `python 0_main.py` in a Anaconda Powershell terminal.
2. `0_dataviz.ipynb`: Jupyter Notebook file. Parameters are set in the first code cell. The notebook can be ran by pressing the "double triangle"/"Run All" button located in the ribbon at the top of the screen.

## Parameter Selection

Here is a list of all parameters and what they do:

| variable | data type | valid inputs/format | Notes |
|---|---|---|---|
| date | string | "YYYY-MM-DD" | Sample date |
| depth | int | 0 - 1100 (recommended) | Maximum working depth of glider model |
| lat_min | float | -85 - 85 | Southern latitude bound |
| lon_min | float | -180 - 180 | Western longitude bound |
| lat_max | float | -85 - 85 | Northern latitude bound |
| lon_max | float | -180 - 180 | Eastern longitude bound |
| CMEMS_ | boolean | True - False | European Model |
| ESPC_ | boolean | True - False | Navy Model |
| RTOFS_EAST_ | boolean | True - False | NOAA model for US east coast |
| RTOFS_WEST_ | boolean | True - False | NOAA model for US west coast |
| RTOFS_PARALLEL_ | boolean | True - False | Experimental NOAA model for east coast |
| COMPUTE_OPTIMAL_PATH | boolean | True - False | Compute the optimal path using an A* algorithm |
| waypoints | list(tuple) | [(lat1, lon1), ..., (latx, lonx)] | List of coordinates to pass into the A* algorithm. Minimum of 2 points are required |
| glider_raw_speed | float | --- | Raw speed of glider model |
| RMSD | boolean | True - False | Calculate the root mean square difference between all selected model combinations |
| make_magnitude_plot | boolean | True - False | Make a plot of depth averaged current magnitude contours |
| make_threshold_plot | boolean | True - False | Make a plot of depth averaged current magnitude threshold contours |
| make_rmsd_plot | boolean | True - False | Make a plot of the root mean squared difference between all model combinations |
| vector_type | string | "quiver", "streamplot", None | Determines the vector type of the plot to show current direction |
| density | int | --- | Density of streamlines. Higher number = denser streamlines |
| scalar | int | --- | Downsampling scalar for quiver plots. Higher number = Less quivers |
| save | boolean | True - False | Save figures locally in the GGS/figures directory |

__WARNING__: Failure to adhere to variable formatting will result in unexpected errors.
