# author: matthew learn (matt.learn@marine.rutgers.edu)

from models import *
from functions import *

import streamlit as st


@st.cache_resource
def load_data() -> None:
    cmems = CMEMS()
    espc = ESPC()
    rtofs_east = RTOFS()
    rtofs_west = RTOFS()
    rtofs_parallel = RTOFS()

    cmems.load()
    espc.load()
    rtofs_east.load("east")
    rtofs_west.load("west")
    rtofs_parallel.load("parallel")

    return cmems, espc, rtofs_east, rtofs_west, rtofs_parallel


@st.cache_resource
def subset(
    _model_tuple: tuple, date: tuple, extent: tuple, depth: float = 1000
) -> tuple:
    _cmems, _espc, _rtofs_east, _rtofs_west, _rtofs_parallel = _model_tuple
    _cmems.subset(date, extent, depth)
    _espc.subset(date, extent, depth)
    _rtofs_east.subset(date, extent, depth)
    _rtofs_west.subset(date, extent, depth)
    _rtofs_parallel.subset(date, extent, depth)

    _cmems.subset_data = _cmems.subset_data.isel(time=0)
    _espc.subset_data = _espc.subset_data.isel(time=0)
    _rtofs_east.subset_data = _rtofs_east.subset_data.isel(time=0)
    _rtofs_west.subset_data = _rtofs_west.subset_data.isel(time=0)
    _rtofs_parallel.subset_data = _rtofs_parallel.subset_data.isel(time=0)

    return (
        _cmems.subset_data,
        _espc.subset_data,
        _rtofs_east.subset_data,
        _rtofs_west.subset_data,
        _rtofs_parallel.subset_data,
    )


def process_data(_model: object) -> tuple:
    _model.z_interpolated_data = interpolate_depth(_model)
    _model.z_interpolated_data = calculate_magnitude(_model)
    _model.da_data = depth_average(_model)

    return _model.z_interpolated_data, _model.da_data
