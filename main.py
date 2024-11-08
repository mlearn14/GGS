from models import *
from functions import *
from plotting import *


def main() -> None:
    """
    Main entry point for the script.

    Still a work in progress.
    """
    date_min = "2024-10-24"
    date_max = "2024-10-25"
    depth = 1000
    lon_min = -78
    lon_max = -68
    lat_min = 36.5
    lat_max = 42.5

    dates = (date_min, date_max)
    extent = (lat_min, lon_min, lat_max, lon_max)

    # rtofs_east = RTOFS()
    # rtofs_east.load_data('east')
    # print(rtofs_east.data) # should print the data from rtofs_east.data

    # cmems = CMEMS()
    # cmems.load_data()
    # print(cmems.raw_data)
    # cmems_interp = interpolate_depth(cmems.data)
    # print(cmems_interp)

    # espc = ESPC()
    # espc.load_data()
    # # print(espc.raw_data)
    # espc.subset_data(dates, extent, depth)
    # # print(espc.data)
    # # u_interp, v_interp, depth = interpolate_depth(espc.data)
    # # print(u_interp)
    # # print("u_interp.max() ", u_interp.max())
    # # print("u_interp.min() ", u_interp.min())

    # # print(v_interp)
    # # print("v_interp.max() ", v_interp.max())
    # # print("v_interp.min() ", v_interp.min())

    # # test_plot(extent, None, "test.png")

    rtofs = RTOFS()
    rtofs.load("east")
    rtofs.subset(dates, extent, depth)
    rtofs.interpolated_data = interpolate_depth(rtofs.data)
    rtofs.da_data = depth_average(rtofs.interpolated_data)
    print(rtofs.da_data)

    cmems = CMEMS()
    cmems.load()
    cmems.subset(dates, extent, depth)
    cmems.interpolated_data = interpolate_depth(cmems.data)
    cmems.da_data = depth_average(cmems.interpolated_data)
    print(cmems.da_data)


if __name__ == "__main__":
    main()
