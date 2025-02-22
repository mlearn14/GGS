# author: matthew learn (matt.learn@marine.rutgers.edu)
# main streamlit dashboard script

import numpy as np
import streamlit as st
import pandas as pd

import json
import datetime as dt

from ggs2.models import *
from ggs2.dash_funcs import *


## Page Configuration ##
st.set_page_config(
    page_title="Glider Guidance System", page_icon=":ship:", layout="wide"
)

cmems = CMEMS()
espc = ESPC()
rtofs_east = RTOFS("east")
rtofs_west = RTOFS("west")
rtofs_parallel = RTOFS("parallel")

(
    cmems.raw_data,
    espc.raw_data,
    rtofs_east.raw_data,
    rtofs_west.raw_data,
    rtofs_parallel.raw_data,
) = load_models(cmems, espc, rtofs_east, rtofs_west, rtofs_parallel)

## Streamlit UI ##
st.title("Glider Guidance System 2")
st.write("Written by Matthew Learn (matt.learn@marine.rutgers.edu)")
st.write(
    "This dashboard allows for the visualization and analysis of ocean currents to help gliders navigate their paths through the ocean."
)
st.divider()

col1, col2 = st.columns([4, 6])

with col1:
    st.write("## Configuration")

    config_file = st.file_uploader("Upload config file", type=["json"])

    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    yesterday = today - dt.timedelta(days=1)

    if config_file is not None:
        # TODO: check if session state can be updated in a separate python file. if so then this can be moved to dash_funcs
        config = json.loads(config_file.getvalue())
        st.session_state["mission_name"] = config["MISSION_NAME"]

        start_date = config["SUBSET"]["TIME"]["START_DATE"]
        end_date = config["SUBSET"]["TIME"]["END_DATE"]
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
            for date in (start_date, start_date if end_date is None else end_date)
        )
        st.session_state["start_date"] = dates[0]
        st.session_state["end_date"] = dates[1]
        st.session_state["min_lat"] = config["SUBSET"]["EXTENT"]["SW_POINT"][0]
        st.session_state["min_lon"] = config["SUBSET"]["EXTENT"]["SW_POINT"][1]
        st.session_state["max_lat"] = config["SUBSET"]["EXTENT"]["NE_POINT"][0]
        st.session_state["max_lon"] = config["SUBSET"]["EXTENT"]["NE_POINT"][1]
        st.session_state["extent"] = (
            st.session_state["min_lat"],
            st.session_state["min_lon"],
            st.session_state["max_lat"],
            st.session_state["max_lon"],
        )
        st.session_state["depth"] = config["SUBSET"]["MAX_DEPTH"]

        models = config["MODELS"]
        model_selection_dict = {
            cmems: models["CMEMS"],
            espc: models["ESPC"],
            rtofs_east: models["RTOFS_EAST"],
            rtofs_west: models["RTOFS_WEST"],
            rtofs_parallel: models["RTOFS_PARALLEL"],
        }
        model_name_dict = {
            cmems: "CMEMS",
            espc: "ESPC",
            rtofs_east: "RTOFS (East Coast)",
            rtofs_west: "RTOFS (West Coast)",
            rtofs_parallel: "RTOFS (Parallel)",
        }
        model_list: list[object] = [
            model for model, selected in model_selection_dict.items() if selected
        ]
        model_str_list = [model_name_dict[model] for model in model_list]
        st.session_state["models"] = model_str_list

        st.session_state["is_pathfinding"] = config["PATHFINDING"]["ENABLE"]
        st.session_state["pathfinding_algorithm"] = config["PATHFINDING"]["ALGORITHM"]
        st.session_state["pathfinding_heuristic"] = config["PATHFINDING"]["HEURISTIC"]

        waypoints = config["PATHFINDING"]["WAYPOINTS"]
        waypoints_str = ", ".join(f"({lat}, {lon})" for lat, lon in waypoints)
        st.session_state["waypoints"] = waypoints_str

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
        vector_selection = config["PLOTTING"]["VECTORS"]["TYPE"]
        if vector_selection == "streamplot":
            st.session_state["vector_type"] = 0
        else:
            st.session_state["vector_type"] = 1
        st.session_state["density"] = config["PLOTTING"]["VECTORS"][
            "STREAMLINE_DENSITY"
        ]
        st.session_state["scalar"] = config["PLOTTING"]["VECTORS"]["QUIVER_DOWNSCALING"]

        st.session_state["save_plots"] = config["PLOTTING"]["SAVE_FIGURES"]
    else:
        st.session_state["mission_name"] = None
        st.session_state["start_date"] = None
        st.session_state["end_date"] = None
        st.session_state["min_lat"] = 35
        st.session_state["min_lon"] = -75
        st.session_state["max_lat"] = 45
        st.session_state["max_lon"] = -50
        st.session_state["extent"] = None
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

    with st.form("ggs2_form"):
        mission_name = st.text_input(
            "Mission name:",
            st.session_state.get("mission_name"),
        )

        st.write("Subset Options:")
        start_date = st.date_input(
            "Start date:", value=st.session_state.get("start_date")
        )
        end_date = st.date_input("End date:", value=st.session_state.get("end_date"))

        min_lat = st.number_input(
            "Minimum latitude:", -90, 90, value=st.session_state.get("min_lat")
        )
        min_lon = st.number_input(
            "Minimum longitude:", -180, 180, value=st.session_state.get("min_lon")
        )
        max_lat = st.number_input(
            "Maximum latitude:", -90, 90, value=st.session_state.get("max_lat")
        )
        max_lon = st.number_input(
            "Maximum longitude:", -180, 180, value=st.session_state.get("max_lon")
        )

        depth = st.slider(
            "Maximum glider working depth in meters:",
            0,
            1000,
            value=st.session_state.get("depth"),
        )

        st.write("Model Options:")
        model_selection = st.multiselect(
            "Current model(s):",
            [
                "CMEMS",
                "ESPC",
                "RTOFS (East Coast)",
                "RTOFS (West Coast)",
                "RTOFS (Parallel)",
            ],
            default=st.session_state.get("models"),
        )

        st.write("Pathfinding Options:")
        is_optimal_path = st.checkbox(
            "Calculate Optimal Path", value=st.session_state.get("is_pathfinding")
        )
        waypoints = st.text_input("Waypoints:", value=st.session_state.get("waypoints"))

        st.write("Plotting Options:")
        is_individual_plots = st.checkbox(
            "Individual plots", value=st.session_state.get("individual_plots")
        )
        is_simple_diff = st.checkbox(
            "Simple Difference plot", value=st.session_state.get("simple_diff")
        )
        is_mean_diff = st.checkbox(
            "Mean of Differences plot", value=st.session_state.get("mean_diff")
        )
        is_simple_mean = st.checkbox(
            "Simple Mean plot", value=st.session_state.get("simple_mean")
        )
        is_rmsd_profile = st.checkbox(
            "RMS Profile Difference plot", value=st.session_state.get("rmsd_profile")
        )

        st.write("Contour Options:")
        is_mag_contours = st.checkbox(
            "Magnitude Contours", value=st.session_state.get("mag_contours")
        )
        is_threshold_contours = st.checkbox(
            "Threshold Contours", value=st.session_state.get("threshold_contours")
        )

        vector_type = st.radio(
            "Vector type:",
            ["streamplot", "quiver"],
            index=st.session_state.get("vector_type"),
        )
        density = st.number_input(
            "Streamline density:",
            min_value=1,
            max_value=7,
            value=st.session_state.get("density"),
        )
        scalar = st.number_input(
            "Quiver downscaling value:",
            min_value=1,
            max_value=7,
            value=st.session_state.get("scalar"),
        )

        is_save = st.toggle("Save Plots", value=st.session_state.get("save_plots"))

        st.form_submit_button("Submit")


with col2:
    coords = pd.DataFrame(
        {
            "lat": [min_lat, max_lat, min_lat, max_lat],
            "lon": [min_lon, max_lon, max_lon, min_lon],
        }
    )
    map = st.empty()
    map.map(coords)
