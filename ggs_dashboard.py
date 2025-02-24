# author: matthew learn (matt.learn@marine.rutgers.edu)
# main streamlit dashboard script

import numpy as np
import streamlit as st
import pandas as pd

import io
import json
import datetime as dt
import os
import subprocess
import time

from ggs2.models import *
from ggs2.dash_config import *
from ggs2.dash_util import load_models
from ggs2.dash_form import *


## Page Configuration ##
st.set_page_config(
    page_title="Glider Guidance System", page_icon=":ship:", layout="wide"
)

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

    if config_file is not None:
        config: dict = json.loads(config_file.getvalue())
        update_config_ss(config)
    else:
        reset_config_ss()

    render_form()

    st.write(f"{st.session_state['mission_name']} Extent:")
    min_lat = st.session_state["min_lat"]
    min_lon = st.session_state["min_lon"]
    max_lat = st.session_state["max_lat"]
    max_lon = st.session_state["max_lon"]
    coords = pd.DataFrame(
        {
            "lat": [min_lat, max_lat, min_lat, max_lat],
            "lon": [min_lon, max_lon, max_lon, min_lon],
        }
    )
    map = st.empty()
    map.map(coords)
    st.divider()

    st.write("### Config Report")
    mission_name = st.session_state["mission_name"]
    if mission_name is None:
        mission_name = "mission"

    config = get_current_config()
    f = io.StringIO()
    json.dump(config, f, indent=4)
    config_json = f.getvalue()

    dir = "config"
    config_path = f"{dir}/dash_temp.json"

    os.makedirs("config", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    st.download_button(
        f"Download {mission_name} config file",
        data=config_json,
        file_name=f"{(mission_name).replace(" ", "_").lower()}.json",
    )


with col2:
    st.write("## Results")  # TODO: come up with better name

    if "output" not in st.session_state:
        st.session_state.output = ""

    if st.button("Run GGS2"):
        with st.expander("Console Output", expanded=True):
            output_box = st.empty()  # Placeholder for the output

            # run main.py
            process = subprocess.Popen(
                ["python", "main.py", "--config_name", "dash_temp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # display console output
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
                st.session_state.output += line  # Append new output
                output_box.text(st.session_state.output)  # Update Streamlit UI
                time.sleep(0.1)  # Give time for UI refresh

            process.stdout.close()
            process.wait()
