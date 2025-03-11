# author: matthew learn (matt.learn@marine.rutgers.edu)
# This script contains general helper functions and individual model processing functions.

import datetime as dt
from datetime import datetime
import json
import math
import os


def read_config(config_name: str) -> dict:
    """
    Reads the config file and returns the contents as a dictionary.

    Args
    ----------
        config_name (str)
            The name of the config file to read.
    Returns
    ----------
        config (dict)
            The contents of the config file as a dictionary.
    """
    try:
        with open(f"config/{config_name}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file '{config_name}.json' not found.")
        return


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

    Args
    ----------
        `None`

    Returns
    ----------
        start_time (datetime): The start time of the script.
    """
    starttime = dt.datetime.now(dt.timezone.utc)
    print(f"Start time (UTC): {starttime}")

    return starttime


def print_endtime() -> datetime:
    """
    Prints the end time of the script.

    Args
    ----------
        `None`

    Returns
    ----------
        end_time (datetime): The end time of the script.
    """
    endtime = dt.datetime.now(dt.timezone.utc)
    print(f"End time (UTC): {endtime}")

    return endtime


def print_runtime(starttime: datetime, endtime: datetime) -> None:
    """
    Prints the runtime of the script.

    Args
    ----------
        starttime (datetime): The start time of the script.
        endtime (datetime): The end time of the script.

    Returns
    ----------
        `None`
    """
    runtime = endtime - starttime
    print(f"Runtime: {runtime}\n")


# will be useful for parallel processing for multiple timestamps
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


def save_fig(fig: object, filename: str, date: str) -> None:
    """
    Save a figure to a file.

    Args:
    ----------
        - fig (object): The figure object to be saved.
        - filename (str): The name of the file to save the figure to.

    Returns:
    ----------
        - `None`
    """
    print(f"Saving figure to {filename}")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    print("Saved.")
