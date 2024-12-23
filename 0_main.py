from models import *
from functions import *
from plotting import *

from concurrent.futures import ProcessPoolExecutor


def main() -> None:
    """
    Main entry point for the script.
    """
    print("Starting script...")
    starttime = print_starttime()

    start_date = "2024-12-23"
    end_date = "2024-12-23"
    depth = 1000
    lon_min = -79
    lon_max = -50
    lat_min = 34
    lat_max = 45
    density = 5
    scalar = 4

    dates = (start_date, end_date)
    extent = (lat_min, lon_min, lat_max, lon_max)

    # Process CMEMS data
    cmems = CMEMS()
    cmems.load()
    cmems.subset(dates, extent, depth)
    subset = cmems.subset_data.isel(time=0)
    interp = interpolate_depth(subset)
    cmems_ds = calculate_magnitude(interp)
    cmems_dac = depth_average(cmems_ds)

    # Process ESPC data
    espc = ESPC()
    espc.load()
    espc.subset(dates, extent, depth)
    subset = espc.subset_data.isel(time=0)
    interp = interpolate_depth(subset)
    espc_ds = calculate_magnitude(interp)
    espc_dac = depth_average(espc_ds)

    # Process east coast RTOFS data
    rtofs_e = RTOFS()
    rtofs_e.load("east")
    rtofs_e.subset(dates, extent, depth)
    subset = rtofs_e.subset_data.isel(time=0)
    interp = interpolate_depth(subset)
    rtofs_e_ds = calculate_magnitude(interp)
    rtofs_e_dac = depth_average(rtofs_e_ds)

    # Process parallel RTOFS data
    rtofs_p = RTOFS()
    rtofs_p.load("parallel")
    rtofs_p.subset(dates, extent, depth)
    subset = rtofs_p.subset_data.isel(time=0)
    interp = interpolate_depth(subset)
    rtofs_p_ds = calculate_magnitude(interp)
    rtofs_p_dac = depth_average(rtofs_p_ds)

    # Take RMSD
    rmsd_c_e = calculate_rmsd(cmems_ds, espc_ds)
    rmsd_c_re = calculate_rmsd(rtofs_e_ds, cmems_ds)
    rmsd_c_rp = calculate_rmsd(rtofs_p_ds, cmems_ds)
    rmsd_e_re = calculate_rmsd(rtofs_e_ds, espc_ds)
    rmsd_e_rp = calculate_rmsd(rtofs_p_ds, espc_ds)
    rmsd_re_rp = calculate_rmsd(rtofs_e_ds, rtofs_p_ds)

    # # Take MAD
    # mad_c_e = calculate_mad(cmems_ds, espc_ds)
    # mad_c_re = calculate_mad(cmems_ds, rtofs_e_ds)
    # mad_c_rp = calculate_mad(cmems_ds, rtofs_p_ds)
    # mad_e_re = calculate_mad(espc_ds, rtofs_e_ds)
    # mad_e_rp = calculate_mad(espc_ds, rtofs_p_ds)
    # mad_re_rp = calculate_mad(rtofs_e_ds, rtofs_p_ds)

    # Plot Magnitudes
    plot_magnitude(cmems_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_magnitude(cmems_dac, extent, quiver=True, scalar=scalar, savefig=True)

    plot_magnitude(espc_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_magnitude(espc_dac, extent, quiver=True, scalar=scalar, savefig=True)

    plot_magnitude(rtofs_e_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_magnitude(rtofs_e_dac, extent, quiver=True, scalar=scalar, savefig=True)

    plot_magnitude(rtofs_p_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_magnitude(rtofs_p_dac, extent, quiver=True, scalar=scalar, savefig=True)

    # Plot Thresholds
    plot_threshold(cmems_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_threshold(cmems_dac, extent, quiver=True, scalar=scalar, savefig=True)

    plot_threshold(espc_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_threshold(espc_dac, extent, quiver=True, scalar=scalar, savefig=True)

    plot_threshold(rtofs_e_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_threshold(rtofs_e_dac, extent, quiver=True, scalar=scalar, savefig=True)

    plot_threshold(rtofs_p_dac, extent, streamlines=True, density=density, savefig=True)
    # plot_threshold(rtofs_p_dac, extent, quiver=True, scalar=scalar, savefig=True)

    # # Plot RMSD
    plot_rmsd(rmsd_c_e, extent, savefig=True)
    plot_rmsd(rmsd_c_re, extent, savefig=True)
    plot_rmsd(rmsd_c_rp, extent, savefig=True)
    plot_rmsd(rmsd_e_re, extent, savefig=True)
    plot_rmsd(rmsd_e_rp, extent, savefig=True)
    plot_rmsd(rmsd_re_rp, extent, savefig=True)

    # # Plot MAD
    # plot_mad(mad_c_e, extent, savefig=True)
    # plot_mad(mad_c_re, extent, savefig=True)
    # plot_mad(mad_c_rp, extent, savefig=True)
    # plot_mad(mad_e_re, extent, savefig=True)
    # plot_mad(mad_e_rp, extent, savefig=True)
    # plot_mad(mad_re_rp, extent, savefig=True)

    print("Script Completed.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)


if __name__ == "__main__":
    # TODO: add parallel processing
    main()
