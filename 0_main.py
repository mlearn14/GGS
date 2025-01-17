from models import *
from functions import *
from pathfinding import *
from plotting import *

from concurrent.futures import ProcessPoolExecutor


def main() -> None:
    """
    Main entry point for the script.
    """
    ############################## PARAMETERS ##############################

    # Dataset parameters
    date = "2025-01-18"
    depth = 1100
    lon_min = -79
    lon_max = -50
    lat_min = 34
    lat_max = 45

    # Pathfinding parameters
    conpute_optimal_path = False
    waypoints = [
        (41.240, -70.958),
        (37.992, -71.248),
        (36.943, -66.862),
        (38.666, -62.978),
        (39.801, -60.653),
        (39.618, -55.87),
    ]

    # Plotting parameters
    contour_type = "threshold"  # "magnitude", "threshold", "rmsd"
    vector_type = "streamplot"  # "quiver", "streamplot", None
    density = 6  # Used for streamlines. Higer number = more streamlines
    scalar = 4  # Used for quiver plots. Lower number = more dense quivers
    save = True

    ############################## INITIALIZE ##################################
    print(
        rf"""
                  _____  _____  ______
                 / ___/ / ___/ /  ___/
                | | __ | | __ (  (__
                | | \ || | \ | \__  \
                | |_| || |_| | ___)  )
                 \____| \____|/_____/

                Glider Guidance System
                        v 2.0
                Created by Matthew Learn 
        
Need help? Send an email to matt.learn@marine.rutgers.edu
        """
    )
    print("Starting script...")
    starttime = print_starttime()

    ############################## LOAD DATA ##################################
    print("\n######################## LOADING DATA ##############################\n")
    dates = (date, date)
    extent = (lat_min, lon_min, lat_max, lon_max)

    cmems = CMEMS()
    cmems.load()
    cmems.subset(dates, extent, depth)
    cmems.subset_data = cmems.subset_data.isel(time=0)

    espc = ESPC()
    espc.load()
    espc.subset(dates, extent, depth)
    espc.subset_data = espc.subset_data.isel(time=0)

    rtofs_e = RTOFS()
    rtofs_e.load("east")
    rtofs_e.subset(dates, extent, depth)
    rtofs_e.subset_data = rtofs_e.subset_data.isel(time=0)

    rtofs_p = RTOFS()
    rtofs_p.load("parallel")
    rtofs_p.subset(dates, extent, depth)
    rtofs_p.subset_data = rtofs_p.subset_data.isel(time=0)

    ############################## PROCESS DATA #################################
    print("######################## PROCESSING DATA ##############################\n")
    cmems.z_interpolated_data = interpolate_depth(cmems)
    cmems.z_interpolated_data = calculate_magnitude(cmems)
    cmems.da_data = depth_average(cmems)

    espc.z_interpolated_data = interpolate_depth(espc)
    espc.z_interpolated_data = calculate_magnitude(espc)
    espc.da_data = depth_average(espc)

    rtofs_e.z_interpolated_data = interpolate_depth(rtofs_e)
    rtofs_e.z_interpolated_data = calculate_magnitude(rtofs_e)
    rtofs_e.da_data = depth_average(rtofs_e)
    rtofs_e_dac_regridded = regrid_ds(rtofs_e.da_data, cmems.da_data)

    rtofs_p.z_interpolated_data = interpolate_depth(rtofs_p)
    rtofs_p.z_interpolated_data = calculate_magnitude(rtofs_p)
    rtofs_p.da_data = depth_average(rtofs_p)
    rtofs_p_dac_regridded = regrid_ds(rtofs_p.da_data, cmems.da_data)

    # Compare models
    if contour_type == "rmsd":
        rmsd_re_rp = calculate_rmsd(
            rtofs_e.z_interpolated_data, rtofs_p.z_interpolated_data, regrid=False
        )
        rmsd_re_c = calculate_rmsd(
            rtofs_e.z_interpolated_data, cmems.z_interpolated_data
        )
        rmsd_re_e = calculate_rmsd(
            rtofs_e.z_interpolated_data, espc.z_interpolated_data
        )
        rmsd_rp_c = calculate_rmsd(
            rtofs_p.z_interpolated_data, cmems.z_interpolated_data
        )
        rmsd_rp_e = calculate_rmsd(
            rtofs_p.z_interpolated_data, espc.z_interpolated_data
        )
        rmsd_e_c = calculate_rmsd(espc.z_interpolated_data, cmems.z_interpolated_data)

    # Get optimized paths
    if conpute_optimal_path:
        cmems_path = compute_a_star_path(waypoints, cmems.da_data)
        espc_path = compute_a_star_path(waypoints, espc.da_data)
        rtofs_e_path = compute_a_star_path(waypoints, rtofs_e_dac_regridded)
        rtofs_p_path = compute_a_star_path(waypoints, rtofs_p_dac_regridded)
    else:
        cmems_path = None
        espc_path = None
        rtofs_e_path = None
        rtofs_p_path = None
        waypoints = None

    ############################## PLOT DATA ##############################
    print("######################## PLOTTING DATA ##############################\n")
    if contour_type == "magnitude" or contour_type == "threshold":
        try:
            (
                fig_c,
                contourf_c,
                legend_c,
                cax_c,
                quiver_c,
                streamplot_c,
                path_plot_c,
                wp_plot_c,
            ) = create_map(
                data=cmems.da_data,
                extent=extent,
                contour_type=contour_type,
                vector_type=vector_type,
                density=density,
                scalar=scalar,
                optimized_path=cmems_path,
                waypoints=waypoints,
                initialize=True,
                save=save,
            )
        except:
            print("[ERROR] CMEMS: Failed to create map.\n")

        try:
            (
                fig_e,
                contourf_e,
                legend_e,
                cax_e,
                quiver_e,
                streamplot_e,
                path_plot_e,
                wp_plot_e,
            ) = create_map(
                data=espc.da_data,
                extent=extent,
                contour_type=contour_type,
                vector_type=vector_type,
                density=density,
                scalar=scalar,
                optimized_path=espc_path,
                waypoints=waypoints,
                initialize=True,
                save=save,
            )
        except:
            print("[ERROR] ESPC: Failed to create map.\n")

        try:
            (
                fig_re,
                contourf_re,
                legend_re,
                cax_re,
                quiver_re,
                streamplot_re,
                path_plot_re,
                wp_plot_re,
            ) = create_map(
                data=rtofs_e_dac_regridded,
                extent=extent,
                contour_type=contour_type,
                vector_type=vector_type,
                density=density,
                scalar=scalar,
                optimized_path=rtofs_e_path,
                waypoints=waypoints,
                initialize=True,
                save=save,
            )
        except:
            print("[ERROR] RTOFS (East Coast): Failed to create map.\n")

        try:
            (
                fig_rp,
                contourf_rp,
                legend_rp,
                cax_rp,
                quiver_rp,
                streamplot_rp,
                path_plot_rp,
                wp_plot_rp,
            ) = create_map(
                data=rtofs_p_dac_regridded,
                extent=extent,
                contour_type=contour_type,
                vector_type=vector_type,
                density=density,
                scalar=scalar,
                optimized_path=rtofs_p_path,
                waypoints=waypoints,
                initialize=True,
                save=save,
            )
        except:
            print("[ERROR] RTOFS (Parallel): Failed to create map.\n")

    elif contour_type == "rmsd":
        vector_type = None
        (
            fig_re_rp,
            contourf_re_rp,
            legend_re_rp,
            cax_re_rp,
            quiver_re_rp,
            streamplot_re_rp,
            path_plot_re_rp,
            wp_plot_re_rp,
        ) = create_map(
            data=rmsd_re_rp,
            extent=extent,
            contour_type=contour_type,
            vector_type=vector_type,
            density=density,
            scalar=scalar,
            optimized_path=None,
            waypoints=None,
            initialize=True,
            save=save,
        )

        (
            fig_re_c,
            contourf_re_c,
            legend_re_c,
            cax_re_c,
            quiver_re_c,
            streamplot_re_c,
            path_plot_re_c,
            wp_plot_re_c,
        ) = create_map(
            data=rmsd_re_c,
            extent=extent,
            contour_type=contour_type,
            vector_type=vector_type,
            density=density,
            scalar=scalar,
            optimized_path=None,
            waypoints=None,
            initialize=True,
            save=save,
        )

        (
            fig_re_e,
            contourf_re_e,
            legend_re_e,
            cax_re_e,
            quiver_re_e,
            streamplot_re_e,
            path_plot_re_e,
            wp_plot_re_e,
        ) = create_map(
            data=rmsd_re_e,
            extent=extent,
            contour_type=contour_type,
            vector_type=vector_type,
            density=density,
            scalar=scalar,
            optimized_path=None,
            waypoints=None,
            initialize=True,
            save=save,
        )

        (
            fig_rp_c,
            contourf_rp_c,
            legend_rp_c,
            cax_rp_c,
            quiver_rp_c,
            streamplot_rp_c,
            path_plot_rp_c,
            wp_plot_rp_c,
        ) = create_map(
            data=rmsd_rp_c,
            extent=extent,
            contour_type=contour_type,
            vector_type=vector_type,
            density=density,
            scalar=scalar,
            optimized_path=None,
            waypoints=None,
            initialize=True,
            save=save,
        )

        (
            fig_rp_e,
            contourf_rp_e,
            legend_rp_e,
            cax_rp_e,
            quiver_rp_e,
            streamplot_rp_e,
            path_plot_rp_e,
            wp_plot_rp_e,
        ) = create_map(
            data=rmsd_rp_e,
            extent=extent,
            contour_type=contour_type,
            vector_type=vector_type,
            density=density,
            scalar=scalar,
            optimized_path=None,
            waypoints=None,
            initialize=True,
            save=save,
        )

        (
            fig_e_c,
            contourf_e_c,
            legend_e_c,
            cax_e_c,
            quiver_e_c,
            streamplot_e_c,
            path_plot_e_c,
            wp_plot_e_c,
        ) = create_map(
            data=rmsd_e_c,
            extent=extent,
            contour_type=contour_type,
            vector_type=vector_type,
            density=density,
            scalar=scalar,
            optimized_path=None,
            waypoints=None,
            initialize=True,
            save=save,
        )

    else:
        raise ValueError(f"Invalid contour type: {contour_type}.")

    print("Script Completed.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)


if __name__ == "__main__":
    # TODO: add parallel processing
    main()
