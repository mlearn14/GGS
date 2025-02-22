# author: matthew learn (matt.learn@marine.rutgers.edu)
# main streamlit dashboard script

import numpy as np
import streamlit as st
import pandas as pd

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

    # TODO: test file reading
    config_file = st.file_uploader("Upload config file", type=["json"])

    with st.form("ggs2_form"):
        mission_name = st.text_input("Mission name:", "GGS2 Mission")
        start_date = st.date_input("Start date:")
        end_date = st.date_input("End date:")
        min_lat = st.number_input("Minimum latitude:", -90, 90, -90)
        min_lon = st.number_input("Minimum longitude:", -180, 180, -180)
        max_lat = st.number_input("Maximum latitude:", -90, 90, 90)
        max_lon = st.number_input("Maximum longitude:", -180, 180, 180)
        depth = st.slider("Maximum glider working depth in meters:", 0, 1000, 1000)
        model_selection = st.multiselect(
            "Current model(s):",
            [
                "CMEMS",
                "ESPC",
                "RTOFS (East Coast)",
                "RTOFS (West Coast)",
                "RTOFS (Parallel)",
            ],
        )
        st.form_submit_button("Submit")

with col2:
    st.write("column 2")
