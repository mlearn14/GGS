# author: matthew learn (matthewalearn@gmail.com)
# This script contains functions for handling parameter inputs and reporting their values to the terminal.

import datetime as dt
from itertools import combinations

import ggs2.models as model


def format_dates(dates: tuple) -> tuple:
    """
    Formats the dates into two datetime objects.

    Args
    ----------
        dates (tuple): A tuple of (date_min, date_max).

    Returns
    ----------
        dates_formatted (tuple): A tuple of (date_min, date_max) in datetime format.
    """

    def parse_date(date):
        today = dt.date.today()
        if date is None or date.lower() == "today":
            return today
        elif date.lower() == "tomorrow":
            tomorrow = today + dt.timedelta(days=1)
            tomorrow = tomorrow.strftime("%Y-%m-%d %H:%M:%S")
            return tomorrow
        elif date.lower() == "yesterday":
            yesterday = today - dt.timedelta(days=1)
            yesterday = yesterday.strftime("%Y-%m-%d %H:%M:%S")
            return yesterday
        else:
            return date

    return tuple(parse_date(date) for date in (dates[0], dates[1] or dates[0]))


def parse_models(model_selection: dict):
    """
    Parses the model selection dictionary.

    Args
    ----------
        model_selection (dict)
            A dictionary of model selections.

    Returns
    ----------
        models (list)
            A list of selected models.
    """
    model_classes = {
        "CMEMS": model.CMEMS,
        "ESPC": model.ESPC,
        "RTOFS_EAST": lambda: model.RTOFS("east"),
        "RTOFS_WEST": lambda: model.RTOFS("west"),
        "RTOFS_PARALLEL": lambda: model.RTOFS("parallel"),
    }

    return [model_classes[name]() for name in model_classes if model_selection[name]]


def get_model_combos(models: list):
    """
    Returns a list of all possible combinations of models.

    Args
    ----------
        models (list)
            A list of models.

    Returns
    ----------
        model_combos (list)
            A list of all possible non-repeating combinations of models.
    """
    if len(models) == 1:
        return None
    else:
        return list(combinations(models, r=2))


def parse_contours(contour_selection: dict):
    """
    Parses the contour selection dictionary.

    Args
    ----------
        contour_selection (dict)
            A dictionary of contour selections.

    Returns
    ----------
        contours (list)
            A list of selected contours.
    """
    contour_options = {
        "magnitude": "PLOT_MAGNITUDES",
        "threshold": "PLOT_MAGNITUDE_THRESHOLDS",
    }

    return [
        contour for contour, key in contour_options.items() if contour_selection[key]
    ]


def parse_comps(comp_selection: dict):
    """
    Parses the comparison selection dictionary.

    Args
    ----------
        comp_selection (dict)
            A dictionary of comparison selections.

    Returns
    ----------
        comps (list)
            A list of selected comparisons.
    """
    comparison_options = {
        "speed_diff": "SIMPLE_DIFFERENCE",
        "u_diff": "U_DIFFERENCE",
        "v_diff": "V_DIFFERENCE",
        "mean_diff": "MEAN_DIFFERENCE",
        "simple_mean": "SIMPLE_MEAN",
        "rmsd_vertical": "RMS_VERTICAL_DIFFERENCE",
    }

    return [comp for comp, key in comparison_options.items() if comp_selection[key]]


def initialize_parameters(config: dict) -> dict:
    """
    Initializes the parameters dictionary.

    Args
    ----------
        config (dict)
            The configuration dictionary.

    Returns
    ----------
        params (dict)
            The parameters dictionary.
    """
    params = {}  # initialize dictionary

    # mission name
    params["mission_name"] = config["MISSION_NAME"]
    params["mission_fname"] = config["MISSION_NAME"].replace(" ", "_").lower()

    # dates
    dates = config["SUBSET"]["TIME"]["START_DATE"], config["SUBSET"]["TIME"]["END_DATE"]
    params["dates"] = format_dates(dates)
    params["single_date"] = len(set(params["dates"])) == 1

    # extent
    params["extent"] = (
        config["SUBSET"]["EXTENT"]["SW_POINT"][0],
        config["SUBSET"]["EXTENT"]["SW_POINT"][1],
        config["SUBSET"]["EXTENT"]["NE_POINT"][0],
        config["SUBSET"]["EXTENT"]["NE_POINT"][1],
    )

    # depth
    params["depth"] = config["SUBSET"]["MAX_DEPTH"]

    # models
    params["models"] = parse_models(config["MODELS"])
    params["model_combos"] = get_model_combos(params["models"])

    # pathfinding
    pathfinding_params = config["PATHFINDING"]
    params["pathfinding"] = pathfinding_params["ENABLE"]
    params["algorithm"] = pathfinding_params["ALGORITHM"]
    params["heuristic"] = pathfinding_params["HEURISTIC"]
    params["waypoints"] = [(wp[0], wp[1]) for wp in pathfinding_params["WAYPOINTS"]]
    params["glider_speed"] = pathfinding_params["GLIDER_RAW_SPEED"]

    params["save_data"] = config["SAVE_DATA"]

    # plotting
    params["individual_plots"] = config["PLOTTING"]["INDIVIDUAL_PLOTS"]
    params["contours"] = parse_contours(config["PLOTTING"])
    params["vector_type"] = config["PLOTTING"]["VECTORS"]["TYPE"]
    params["density"] = config["PLOTTING"]["VECTORS"]["STREAMLINE_DENSITY"]
    params["scalar"] = config["PLOTTING"]["VECTORS"]["QUIVER_DOWNSCALING"]
    params["comparison_plots"] = parse_comps(config["PLOTTING"]["COMPARISON_PLOTS"])
    params["plot_optimal_path"] = config["PLOTTING"]["PLOT_OPTIMAL_PATH"]
    params["save_figures"] = config["PLOTTING"]["SAVE_FIGURES"]

    return params


def ticket_report(params: dict) -> None:
    """Prints the ticket report for the selected parameters."""

    waypoints = params["waypoints"]
    if waypoints is not None:
        waypoints = ",\n\t   ".join([f"({y}°, {x}°)" for y, x in params["waypoints"]])

    contour_dict = {"magnitude": "Magnitude", "threshold": "Magnitude Threshold"}
    comp_dict = {
        "speed_diff": "Speed Difference",
        "u_diff": "Eastward Difference (u)",
        "v_diff": "Northward Difference (v)",
        "mean_diff": "Mean Difference",
        "simple_mean": "Simple Mean",
        "rmsd_vertical": "RMS Vertical Difference",
    }
    ticket = {
        "Mission Name": params["mission_name"],
        "Start Date": params["dates"][0],
        "End Date": params["dates"][1],
        "Southwest Point": f"({params['extent'][0]}°, {params['extent'][1]}°)",
        "Northeast Point": f"({params['extent'][2]}°, {params['extent'][3]}°)",
        "Depth": params["depth"],
        "Models": ", ".join([model.name for model in params["models"]]),
        "Pathfinding": params["pathfinding"],
        "Algorithm": params["algorithm"],
        "Heuristic": params["heuristic"],
        "Waypoints": waypoints,
        "Glider Raw Speed": f"{params["glider_speed"]} m/s",
        "Save Data": params["save_data"],
        "Individual Plots": params["individual_plots"],
        "Contours": ", ".join(contour_dict[plot] for plot in params["contours"]),
        "Vectors": params["vector_type"],
        "Streamline Density": params["density"],
        "Quiver Downscaling Value": params["scalar"],
        "Comparison Plots": ", ".join(
            [comp_dict[plot] for plot in params["comparison_plots"]]
        ),
        "Plot Optimal Path": params["plot_optimal_path"],
        "Save Figures": params["save_figures"],
    }

    print(
        "----------------------------\nTicket Information:\n----------------------------"
    )
    for key, value in ticket.items():
        print(f"{key}: {value}")
    print()


def model_raw_report(model_list: list[object]) -> None:
    """Prints the raw report for the selected models."""
    min_date_list = [model.raw_data.time.min().values for model in model_list]
    max_date_list = [model.raw_data.time.max().values for model in model_list]
    print(
        "----------------------------\nModel Raw Report:\n----------------------------"
    )
    # prints the range of valid dates that can be used.
    print(f"Minimum date: {max(min_date_list).astype(str).split('T')[0]}")
    print(f"Maximum date: {min(max_date_list).astype(str).split('T')[0]}")
