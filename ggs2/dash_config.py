import streamlit as st

from .dash_util import (
    process_config_dates,
    get_model_str_list,
    get_model_dict,
    get_wp_list,
    get_vector_index,
)


def update_config_ss(config: dict):
    waypoints = config["PATHFINDING"]["WAYPOINTS"]
    if waypoints is not None:
        waypoints = ", ".join(
            f"({lat}, {lon})" for lat, lon in config["PATHFINDING"]["WAYPOINTS"]
        )
    start_date, end_date = process_config_dates(config)
    st.session_state["mission_name"] = config["MISSION_NAME"]
    st.session_state["start_date"] = start_date
    st.session_state["end_date"] = end_date
    st.session_state["min_lat"] = config["SUBSET"]["EXTENT"]["SW_POINT"][0]
    st.session_state["min_lon"] = config["SUBSET"]["EXTENT"]["SW_POINT"][1]
    st.session_state["max_lat"] = config["SUBSET"]["EXTENT"]["NE_POINT"][0]
    st.session_state["max_lon"] = config["SUBSET"]["EXTENT"]["NE_POINT"][1]
    st.session_state["depth"] = config["SUBSET"]["MAX_DEPTH"]
    st.session_state["models"] = get_model_str_list(config["MODELS"])
    st.session_state["is_pathfinding"] = config["PATHFINDING"]["ENABLE"]
    st.session_state["pathfinding_algorithm"] = config["PATHFINDING"]["ALGORITHM"]
    st.session_state["pathfinding_heuristic"] = config["PATHFINDING"]["HEURISTIC"]
    st.session_state["waypoints"] = waypoints
    st.session_state["glider_raw_speed"] = config["PATHFINDING"]["GLIDER_RAW_SPEED"]
    st.session_state["individual_plots"] = config["PLOTTING"]["INDIVIDUAL_PLOTS"]
    st.session_state["simple_diff"] = config["PLOTTING"]["COMPARISON_PLOTS"][
        "SIMPLE_DIFFERENCE"
    ]
    st.session_state["mean_diff"] = config["PLOTTING"]["COMPARISON_PLOTS"][
        "MEAN_DIFFERENCE"
    ]
    st.session_state["simple_mean"] = config["PLOTTING"]["COMPARISON_PLOTS"][
        "SIMPLE_MEAN"
    ]
    st.session_state["rmsd_profile"] = config["PLOTTING"]["COMPARISON_PLOTS"][
        "RMS_PROFILE_DIFFERENCE"
    ]
    st.session_state["mag_contours"] = config["PLOTTING"]["PLOT_MAGNITUDES"]
    st.session_state["threshold_contours"] = config["PLOTTING"][
        "PLOT_MAGNITUDE_THRESHOLDS"
    ]
    st.session_state["vector_type"] = get_vector_index(
        config["PLOTTING"]["VECTORS"]["TYPE"]
    )
    st.session_state["density"] = config["PLOTTING"]["VECTORS"]["STREAMLINE_DENSITY"]
    st.session_state["scalar"] = config["PLOTTING"]["VECTORS"]["QUIVER_DOWNSCALING"]
    st.session_state["save_plots"] = config["PLOTTING"]["SAVE_FIGURES"]


def reset_config_ss():
    st.session_state["mission_name"] = None
    st.session_state["start_date"] = None
    st.session_state["end_date"] = None
    st.session_state["min_lat"] = 35
    st.session_state["min_lon"] = -75
    st.session_state["max_lat"] = 45
    st.session_state["max_lon"] = -50
    st.session_state["depth"] = 1000
    st.session_state["models"] = None
    st.session_state["is_pathfinding"] = None
    st.session_state["pathfinding_algorithm"] = None
    st.session_state["pathfinding_heuristic"] = None
    st.session_state["waypoints"] = None
    st.session_state["glider_raw_speed"] = 0.5
    st.session_state["individual_plots"] = None
    st.session_state["simple_diff"] = None
    st.session_state["mean_diff"] = None
    st.session_state["simple_mean"] = None
    st.session_state["rmsd_profile"] = None
    st.session_state["mag_contours"] = None
    st.session_state["threshold_contours"] = None
    st.session_state["vector_type"] = None
    st.session_state["density"] = 5
    st.session_state["scalar"] = 4
    st.session_state["save_plots"] = None


def get_current_config() -> dict:
    if st.session_state["start_date"] is None:
        start_date = None
    else:
        start_date = str(st.session_state["start_date"])

    if st.session_state["end_date"] is None:
        end_date = None
    else:
        end_date = str(st.session_state["end_date"])

    return {
        "MISSION_NAME": st.session_state["mission_name"],
        "SUBSET": {
            "TIME": {
                "START_DATE": start_date,
                "END_DATE": end_date,
            },
            "EXTENT": {
                "SW_POINT": [st.session_state["min_lat"], st.session_state["min_lon"]],
                "NE_POINT": [st.session_state["max_lat"], st.session_state["max_lon"]],
            },
            "MAX_DEPTH": st.session_state["depth"],
        },
        "MODELS": get_model_dict(st.session_state["models"]),
        "PATHFINDING": {
            "ENABLE": st.session_state["is_pathfinding"],
            "ALGORITHM": st.session_state["pathfinding_algorithm"],
            "HEURISTIC": st.session_state["pathfinding_heuristic"],
            "WAYPOINTS": get_wp_list(st.session_state["waypoints"]),
            "GLIDER_RAW_SPEED": st.session_state["glider_raw_speed"],
        },
        "PLOTTING": {
            "INDIVIDUAL_PLOTS": st.session_state["individual_plots"],
            "COMPARISON_PLOTS": {
                "SIMPLE_DIFFERENCE": st.session_state["simple_diff"],
                "MEAN_DIFFERENCE": st.session_state["mean_diff"],
                "SIMPLE_MEAN": st.session_state["simple_mean"],
                "RMS_PROFILE_DIFFERENCE": st.session_state["rmsd_profile"],
            },
            "PLOT_MAGNITUDES": st.session_state["mag_contours"],
            "PLOT_MAGNITUDE_THRESHOLDS": st.session_state["threshold_contours"],
            "PLOT_OPTIMAL_PATH": st.session_state["is_pathfinding"],
            "VECTORS": {
                "TYPE": st.session_state["vector_type"],
                "STREAMLINE_DENSITY": st.session_state["density"],
                "QUIVER_DOWNSCALING": st.session_state["scalar"],
            },
            "SAVE_FIGURES": st.session_state["save_plots"],
        },
    }
