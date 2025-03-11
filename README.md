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

Last updated for v1.2.0

## Overview

This README will cover the installation of the Glider Guidance System 2 (GGS2) and all of its dependencies. For documentation on the use of the GGS2, refer to `_docs/GGS2_Tutorial.md`.

## Installation Requirements

- __Python:__ Install from the [official Python website](https://www.python.org/downloads/). Select `Add to PATH` in the installation wizard.
- __Conda:__ Conda can be installed using either Miniconda (recommended) or Anaconda. Install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html). Select `Add to PATH` in the installation wizard.
- __Microsoft Visual Studio Code (recommended):__ Any IDE can be used, but VS Code is recommended. Install from the [official website](https://code.visualstudio.com/). Once installed ensure the following extensions are installed:
  - Python
  - Jupyter

## Python Requirements

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
- xarray
- xesmf
