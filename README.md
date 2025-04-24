# Glider Guidance System 2

      /\\\\\\\\\\\\      /\\\\\\\\\\\\      /\\\\\\\\\\\       /\\\\\\\\\         
     /\\\//////////     /\\\//////////     /\\\/////////\\\   /\\\///////\\\
     /\\\               /\\\               \//\\\      \///   \///      \//\\\
     \/\\\    /\\\\\\\  \/\\\    /\\\\\\\    \////\\\                    /\\\/
      \/\\\   \/////\\\  \/\\\   \/////\\\       \////\\\              /\\\//
       \/\\\       \/\\\  \/\\\       \/\\\          \////\\\        /\\\//
        \/\\\       \/\\\  \/\\\       \/\\\   /\\\      \//\\\     /\\\/
         \//\\\\\\\\\\\\/   \//\\\\\\\\\\\\/   \///\\\\\\\\\\\/     /\\\\\\\\\\\\\\\
           \////////////      \////////////       \///////////      \///////////////

Last updated for v1.2.2

## Overview

This README will cover the installation of the Glider Guidance System 2 (GGS2) and all of its dependencies. For documentation on the use of the GGS2, refer to `_docs/GGS2_Tutorial.md`.

### Program Description

GGS2 loads ocean current model data into memory, subsets, regrid to a common grid, interpolated over depth to uniform one meter resolution intervals, and averaged over depth. The depth averaged data is fed into an A* search algorithm along with a set of waypoints to compute the most time-optimal path between waypoints, taking into accoun the impact of the depth averaged currents. Results are visualized as figures.

### Data Description

GGS2 utilizes ocean current forecast data from three ocean current models:

- __Copernicus Marine Environmental Service (CMEMS)__ - Funded by the European Union and Mercator Ocean International. Uses the Global Ocean Physics Reanalysis (GLORYS) product.
- __Earth System Prediction Capability (ESPC)__ - Based on 1/12° HYbrid Coordinates Ocean Model (HYCOM) unded by the United States Navy.
- __Real-Time Ocean Forecast System (RTOFS)__ - Based on a 1/12° HYCOM and funded by the National Oceanic and Atmospheric Administration and the National Weather Service.

## Installation Requirements

- __Python:__ Install from the [official Python website](https://www.python.org/downloads/). Select `Add to PATH` in the installation wizard.
- __Conda:__ Conda can be installed using either Miniconda (recommended) or Anaconda. Install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html). Select `Add to PATH` in the installation wizard.
- __Microsoft Visual Studio Code (recommended):__ Any IDE can be used, but VS Code is recommended. Install from the [official website](https://code.visualstudio.com/). Once installed ensure the following extensions are installed:
  - Python
  - Jupyter

### Python Requirements

An environment that includes all requirements to run the GGS2 can be found by running `ggs2.yml` in any Anaconda powershell terminal through the following line: `conda env create -f ggs2.yml`.

If instead the user wishes to create their own environment manually, run `conda create -n env_name` and `conda activate env_name`, replacing `env_name` with a name of their choice. Then, using the command `conda install -c conda-forge package`, replacing `package` with the name of the module/package/library, must be run to install the following:

- cartopy
- cmocean
- copernicusmarine
- cool_maps
- esmpy
- jupyterlab
- matplotlib
- numpy
- pandas
- python
- seaborn
- simplekml
- xarray
- xesmf
