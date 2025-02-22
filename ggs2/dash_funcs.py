# author: matthew learn (matt.learn@marine.rutgers.edu)
# functions for dash.py

import streamlit as st

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


def load_cg(extent: tuple, depth: int) -> xr.Dataset:
    return process_common_grid(extent, depth)
