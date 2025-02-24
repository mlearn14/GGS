# author: matthew learn (matt.learn@marine.rutgers.edu)
# functions for dash.py

import streamlit as st

import datetime as dt
import re

from .models import *
from .model_processing import *


@st.cache_resource
def load_models(
    _model1: object, _model2: object, _model3: object, _model4: object, _model5: object
) -> tuple:
    _model1.load(diag_text=False)
    _model2.load(diag_text=False)
    _model3.load(diag_text=False)
    _model4.load(diag_text=False)
    _model5.load(diag_text=False)
    return (
        _model1.raw_data,
        _model2.raw_data,
        _model3.raw_data,
        _model4.raw_data,
        _model5.raw_data,
    )


def process_config_dates(config: dict) -> tuple:
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    yesterday = today - dt.timedelta(days=1)

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

    return dates


def get_model_str_list(model_dict: dict) -> list:
    model_selection_dict = {
        "CMEMS": model_dict["CMEMS"],
        "ESPC": model_dict["ESPC"],
        "RTOFS (East Coast)": model_dict["RTOFS_EAST"],
        "RTOFS (West Coast)": model_dict["RTOFS_WEST"],
        "RTOFS (Parallel)": model_dict["RTOFS_PARALLEL"],
    }
    return [model for model, selected in model_selection_dict.items() if selected]


def get_model_dict(model_list: list) -> dict:
    full_model_list = [
        "CMEMS",
        "ESPC",
        "RTOFS (East Coast)",
        "RTOFS (West Coast)",
        "RTOFS (Parallel)",
    ]
    model_mapping = {
        "RTOFS (East Coast)": "RTOFS_EAST",
        "RTOFS (West Coast)": "RTOFS_WEST",
        "RTOFS (Parallel)": "RTOFS_PARALLEL",
    }
    return {model_mapping.get(model, model): model in model_list for model in full_model_list}


def get_wp_list(waypoint_str: str) -> list:
    if waypoint_str is None:
        return None
    else:
        return [
            list(map(float, pair.split(", ")))
            for pair in re.findall(r"\(([^)]+)\)", waypoint_str)
        ]


def get_vector_index(vector_type: str) -> int:
    if vector_type == "streamplot":
        return 0
    elif vector_type == "quiver":
        return 1
    elif vector_type is None:
        return None
    else:
        ValueError(
            f"Invalid vector type: {vector_type}. Must be 'streamplot', 'quiver', or None."
        )
