# author: matthew learn (matt.learn@marine.rutgers.edu)
# functions creating the dashboard form

import streamlit as st


def update_mission_name():
    mission_name = st.text_input(
        "Mission name:", value=st.session_state.get("mission_name")
    )
    st.session_state["mission_name"] = mission_name


def update_subset_options():
    start_date = st.date_input("Start date:", value=st.session_state.get("start_date"))
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
    st.session_state["start_date"] = start_date
    st.session_state["end_date"] = end_date
    st.session_state["min_lat"] = min_lat
    st.session_state["min_lon"] = min_lon
    st.session_state["max_lat"] = max_lat
    st.session_state["max_lon"] = max_lon
    st.session_state["depth"] = depth


def update_model_options():
    model_selection = st.multiselect(
        "Current model(s):",
        [
            "CMEMS",
            "ESPC",
            "RTOFS (East Coast)",
            "RTOFS (West Coast)",
            "RTOFS (Parallel)",
        ],
        st.session_state.get("models"),
    )
    st.session_state["models"] = model_selection


def update_pathfinding_options():
    is_pathfinding = st.toggle(
        "Calculate Optimal Path:", value=st.session_state.get("is_pathfinding")
    )
    waypoints = st.text_input("Waypoints:", value=st.session_state.get("waypoints"))
    st.session_state["is_pathfinding"] = is_pathfinding
    st.session_state["waypoints"] = waypoints


def update_plotting_options():
    is_individual_plots = st.checkbox(
        "Individual plots:", value=st.session_state.get("individual_plots")
    )
    is_simple_diff = st.checkbox(
        "Simple Difference plot:", value=st.session_state.get("simple_diff")
    )
    is_mean_diff = st.checkbox(
        "Mean of Differences plot:", value=st.session_state.get("mean_diff")
    )
    is_simple_mean = st.checkbox(
        "Simple Mean plot:", value=st.session_state.get("simple_mean")
    )
    is_rmsd_profile = st.checkbox(
        "RMS Profile Difference plot:", value=st.session_state.get("rmsd_profile")
    )

    st.write("Contour Options")
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

    st.session_state["individual_plots"] = is_individual_plots
    st.session_state["simple_diff"] = is_simple_diff
    st.session_state["mean_diff"] = is_mean_diff
    st.session_state["simple_mean"] = is_simple_mean
    st.session_state["rmsd_profile"] = is_rmsd_profile
    st.session_state["mag_contours"] = is_mag_contours
    st.session_state["threshold_contours"] = is_threshold_contours
    st.session_state["vector_type"] = vector_type
    st.session_state["density"] = density
    st.session_state["scalar"] = scalar
    st.session_state["save_plots"] = is_save


def render_form():
    with st.form("ggs2_form"):
        update_mission_name()
        st.divider()
        st.write("### Subset Options")
        update_subset_options()
        st.divider()
        st.write("### Model Options")
        update_model_options()
        st.divider()
        st.write("### Pathfinding Options")
        update_pathfinding_options()
        st.divider()
        st.write("### Plotting Options")
        update_plotting_options()
        st.form_submit_button("Submit")
