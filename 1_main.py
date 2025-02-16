# author: matthew learn (matt.learn@marine.rutgers.edu)
from functions import *
from models import *
from pathfinding import *
from plotting import *
import itertools
import json


def process_individual_model(
    model_name: str, dates: tuple, extent: tuple, depth: int, source: str = None
) -> object:
    # load model
    if model_name == "CMEMS":
        model = CMEMS()
    elif model_name == "RTOFS":
        model = RTOFS()
    elif model_name == "ESPC":
        model = ESPC()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # load data
    model.load(source=source)

    # regrid data

    # raw data summary report

    # subset
    # regrid to common grid
    # interpolate
    #
    pass


def main(config_name: str) -> None:
    try:
        with open(f"config/{config_name}.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file '{config_name}.json' not found.")
        return

    # primary parameter initialization
    MISSION_NAME: str = config["MISSION_NAME"]
    DATES: tuple[str, str] = (
        config["SUBSET"]["TIME"]["START_DATE"],
        config["SUBSET"]["TIME"]["END_DATE"],
    )
    EXTENT: dict = config["SUBSET"]["EXTENT"]
    DEPTH: int = config["SUBSET"]["MAX_DEPTH"]

    MODEL_SELECTION: dict = config["MODELS"]
    IS_PATHFINDING: bool = config["PATHFINDING"]["ENABLE"]
    ALGORITHM: str = config["PATHFINDING"]["ALGORITHM"]
    HEURISTIC: str = config["PATHFINDING"]["HEURISTIC"]
    WAYPOINTS: list[list] = config["PATHFINDING"]["WAYPOINTS"]
    GLIDER_RAW_SPEED: float = config["PATHFINDING"]["GLIDER_RAW_SPEED"]

    INDIVIDUAL_PLOTS: bool = config["PLOTTING"]["INDIVIDUAL_PLOTS"]
    COMPARISON_PLOTS: dict = config["PLOTTING"]["COMPARISON_PLOTS"]

    MAGNITUDE_PLOTS: bool = config["PLOTTING"]["PLOT_MAGNITUDES"]
    THRESHOLD_PLOTS: bool = config["PLOTTING"]["PLOT_MAGNITUDE_THRESHOLDS"]
    PLOT_OPTIMAL_PATH: bool = config["PLOTTING"]["PLOT_OPTIMAL_PATH"]
    VECTORS: dict = config["PLOTTING"]["VECTORS"]
    SAVE_FIGURES: bool = config["PLOTTING"]["SAVE_FIGURES"]

    # logo text
    logo_text()

    # initialize models
    cmems: object = CMEMS()
    espc: object = ESPC()
    rtofs_e: object = RTOFS("east")
    rtofs_w: object = RTOFS("west")
    rtofs_p: object = RTOFS("parallel")

    # secondary parameter initialization
    if DATES[1] is None:
        DATES = DATES[0], DATES[0]
        single_date = True

    extent = (
        EXTENT["SW_POINT"][0],
        EXTENT["SW_POINT"][1],
        EXTENT["NE_POINT"][0],
        EXTENT["NE_POINT"][1],
    )

    model_selection_dict = {
        cmems: MODEL_SELECTION["CMEMS"],
        espc: MODEL_SELECTION["ESPC"],
        rtofs_e: MODEL_SELECTION["RTOFS_EAST"],
        rtofs_w: MODEL_SELECTION["RTOFS_WEST"],
        rtofs_p: MODEL_SELECTION["RTOFS_PARALLEL"],
    }
    model_list = [model for model, selected in model_selection_dict.items() if selected]

    waypoints = [(coord[0], coord[1]) for coord in WAYPOINTS]
    optimal_paths = {}

    comparison_selection_dict = {
        "simple_diff": COMPARISON_PLOTS["SIMPLE_DIFFERENCE"],
        "mean_diff": COMPARISON_PLOTS["MEAN_DIFFERENCE"],
        "simple_mean": COMPARISON_PLOTS["SIMPLE_MEAN"],
        "rmsd": COMPARISON_PLOTS["RMS_PROFILE_DIFFERENCE"],
    }
    comparison_list = [
        comp for comp, selected in comparison_selection_dict.items() if selected
    ]

    contour_select_dict = {
        "magnitude": MAGNITUDE_PLOTS,
        "threshold": THRESHOLD_PLOTS,
    }
    contour_type = [cntr for cntr, selected in contour_select_dict.items() if selected]

    parameters = {
        "mission_name": MISSION_NAME,
        "start_date": DATES[0],
        "end_date": DATES[1],
        "extent": extent,
        "depth": DEPTH,
        "models": model_list,
        "pathfinding": IS_PATHFINDING,
        "algorithm": ALGORITHM,
        "heuristic": HEURISTIC,
        "waypoints": waypoints,
        "glider_raw_speed": GLIDER_RAW_SPEED,
        "indv_plots": INDIVIDUAL_PLOTS,
        "contours": contour_type,
        "vectors": VECTORS,
        "comp_plots": comparison_list,
        "plot_opt_path": PLOT_OPTIMAL_PATH,
        "save_figs": SAVE_FIGURES,
    }

    # read back ticket dict
    ticket_report(parameters)

    # load & ubset common grid

    # process individual models

    # process pathfinding

    # process model comparisons

    # plot individual models

    # plot model comparisons


if __name__ == "__main__":
    config_name = "TEST_MISSION"
    main(config_name)
