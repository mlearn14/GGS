# author: matthew learn (matt.learn@marine.rutgers.edu)
import argparse
import datetime as dt
from ggs2 import *
import itertools
import json


def main() -> None:
    s_starttime = print_starttime()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    args = parser.parse_args()
    config_name = args.config_name

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
    PATHFINDING: bool = config["PATHFINDING"]

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

    mission_fname = MISSION_NAME.replace(" ", "_").lower()

    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    yesterday = today - dt.timedelta(days=1)

    dates = tuple(
        (
            today
            if date is None or date.lower() == "today"
            else (
                tomorrow
                if date.lower() == "tomorrow"
                else yesterday if date.lower() == "yesterday" else date
            )
        )
        for date in (DATES[0], DATES[0] if DATES[1] is None else DATES[1])
    )

    single_date = len(set(dates)) == 1

    # set extent as a tuple from config
    extent = (
        EXTENT["SW_POINT"][0],
        EXTENT["SW_POINT"][1],
        EXTENT["NE_POINT"][0],
        EXTENT["NE_POINT"][1],
    )

    # initialize model list
    model_selection_dict = {
        cmems: MODEL_SELECTION["CMEMS"],
        espc: MODEL_SELECTION["ESPC"],
        rtofs_e: MODEL_SELECTION["RTOFS_EAST"],
        rtofs_w: MODEL_SELECTION["RTOFS_WEST"],
        rtofs_p: MODEL_SELECTION["RTOFS_PARALLEL"],
    }
    model_list: list[object] = [
        model for model, selected in model_selection_dict.items() if selected
    ]

    # convert waypoints from list[list[float, float]] to list[tuple[float, float]]
    PATHFINDING["WAYPOINTS"] = [
        (waypoint[0], waypoint[1]) for waypoint in PATHFINDING["WAYPOINTS"]
    ]

    # initialize contour list
    contour_select_dict = {
        "magnitude": MAGNITUDE_PLOTS,
        "threshold": THRESHOLD_PLOTS,
    }
    contour_type = [cntr for cntr, selected in contour_select_dict.items() if selected]

    # fix for backwards compatibility
    if COMPARISON_PLOTS["SIMPLE_DIFFERENCE"]:
        COMPARISON_PLOTS["SPEED_DIFFERENCE"] = COMPARISON_PLOTS["SIMPLE_DIFFERENCE"]

    # initialize comparison list
    comparison_selection_dict = {
        "speed_diff": COMPARISON_PLOTS["SPEED_DIFFERENCE"],
        "mean_diff": COMPARISON_PLOTS["MEAN_DIFFERENCE"],
        "simple_mean": COMPARISON_PLOTS["SIMPLE_MEAN"],
        "rmsd_profile": COMPARISON_PLOTS["RMS_PROFILE_DIFFERENCE"],
    }
    comparison_list = [
        comp for comp, selected in comparison_selection_dict.items() if selected
    ]

    # initialize non repleating model combinations list
    model_combos = list(itertools.combinations(model_list, r=2))

    parameters = {
        "mission_name": MISSION_NAME,
        "start_date": dates[0],
        "end_date": dates[1],
        "extent": extent,
        "depth": DEPTH,
        "models": model_list,
        "pathfinding": PATHFINDING["ENABLE"],
        "algorithm": PATHFINDING["ALGORITHM"],
        "heuristic": PATHFINDING["HEURISTIC"],
        "waypoints": PATHFINDING["WAYPOINTS"],
        "glider_raw_speed": PATHFINDING["GLIDER_RAW_SPEED"],
        "indv_plots": INDIVIDUAL_PLOTS,
        "contours": contour_type,
        "vectors": VECTORS,
        "comp_plots": comparison_list,
        "plot_opt_path": PLOT_OPTIMAL_PATH,
        "save_figs": SAVE_FIGURES,
    }

    # read back ticket dict
    ticket_report(parameters)

    print("\n----------------------------\nLoading Data:\n----------------------------")
    # load & subset common grid
    COMMON_GRID = process_common_grid(extent, DEPTH)

    # load all selectred models
    for model in model_list:
        model.load()

    # read model reports
    model_raw_report(model_list)

    # process individual models
    print(
        "\n----------------------------\nProcessing Individual Model Data:\n----------------------------"
    )
    for model in model_list:
        process_individual_model(
            model,
            COMMON_GRID,
            dates,
            extent,
            DEPTH,
            single_date,
            PATHFINDING,
            mission_fname,
        )

    # plot individual models
    print(
        "\n----------------------------\nPlotting Individual Model Data:\n----------------------------"
    )
    if INDIVIDUAL_PLOTS:
        for model in model_list:
            if PLOT_OPTIMAL_PATH == False:
                model.optimal_path = None
                model.waypoints = None
            for cntr in contour_type:
                create_map(
                    data=model.da_data,
                    extent=extent,
                    contour_type=cntr,
                    vector_type=VECTORS["TYPE"],
                    density=VECTORS["STREAMLINE_DENSITY"],
                    scalar=VECTORS["QUIVER_DOWNSCALING"],
                    optimized_path=model.optimal_path,
                    waypoints=model.waypoints,
                    mission_name=mission_fname,
                    save=SAVE_FIGURES,
                )

    # process and plot model comparisons FIXME: add error handling for if only one model is selected!!
    print(
        "\n----------------------------\nProcessing and Plotting Model Comparisons:\n----------------------------"
    )
    if comparison_list is not None and comparison_list:
        for comparison in comparison_list:
            if comparison == "speed_diff":
                plot_data = [
                    calculate_speed_diff(model1, model2)
                    for model1, model2 in model_combos
                ]
            elif comparison == "rmsd_profile":
                plot_data = [
                    calculate_rmsd_profile(model1, model2)
                    for model1, model2 in model_combos
                ]
            elif comparison == "mean_diff":
                plot_data = calculate_mean_diff(model_list)
                create_map(
                    data=plot_data,
                    extent=extent,
                    contour_type=comparison,
                    vector_type=None,
                    mission_name=mission_fname,
                    comp_plot=True,
                    save=SAVE_FIGURES,
                )
            elif comparison == "simple_mean":
                simple_mean_cntrs = ["mean_magnitude", "mean_threshold"]
                plot_data = calculate_simple_mean(model_list)
                for cntr in simple_mean_cntrs:
                    create_map(
                        data=plot_data,
                        extent=extent,
                        contour_type=cntr,
                        vector_type=VECTORS["TYPE"],
                        density=VECTORS["STREAMLINE_DENSITY"],
                        scalar=VECTORS["QUIVER_DOWNSCALING"],
                        mission_name=mission_fname,
                        comp_plot=True,
                        save=SAVE_FIGURES,
                    )

            if type(plot_data) == list:
                for data in plot_data:
                    create_map(
                        data=data,
                        extent=extent,
                        contour_type=comparison,
                        vector_type=None,
                        mission_name=mission_fname,
                        comp_plot=True,
                        save=SAVE_FIGURES,
                    )

    s_endtime = print_endtime()
    print("Script complete.")
    print_runtime(s_starttime, s_endtime)
    print("Thank you for using Glider Guidance System 2.")


if __name__ == "__main__":
    main()
