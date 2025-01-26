from models import *
from functions import *
from pathfinding import *
from plotting import *

import itertools


def main() -> None:
    """
    Main entry point for the script.
    """
    ############################## PARAMETERS ##############################

    # Dataset parameters
    date = "2025-01-26"  # format: "YYYY-MM-DD"
    depth = 1100  # keep 1100 to make sure we get the next deepest layer. CMEMS data does not have a layer at 1000 meters
    lat_min = 34
    lon_min = -79
    lat_max = 45
    lon_max = -50

    # Model selection
    CMEMS_ = True  # European model
    ESPC_ = True  # Navy model
    RTOFS_EAST_ = True  # NOAA model for US east coast
    RTOFS_WEST_ = False  # NOAA model for US west coast
    RTOFS_PARALLEL_ = True  # experimental NOAA model for US east coast

    # Pathfinding parameters
    compute_optimal_path = False
    waypoints = [
        (41.240, -70.958),
        (37.992, -71.248),
        (36.943, -66.862),
        (38.666, -62.978),
        (39.801, -60.653),
        (39.618, -55.87),
    ]
    glider_raw_speed = 0.5  # m/s. This is the base speed of the glider

    # Model combarison parameters
    RMSD = True  # Root mean squared difference. Set to True or False (no quotations)

    # Plotting parameters
    make_magnitude_plot = False  # set to True or False (no quotations)
    make_threshold_plot = False  # set to True or False (no quotations)
    make_rmsd_plot = True  # set to True or False (no quotations)

    # "quiver", "streamplot", or None (no quotations arond None)
    vector_type = "quiver"
    density = 5  # Used for streamlines. Higer number = more streamlines
    scalar = 4  # Used for quiver plots. Lower number = more dense quivers

    save = True  # set to True or False (no quotations)

    ############################## INITIALIZE SCRIPT ##################################
    # test
    print(
        rf"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~          
 ~~~~~/\\\\\\\\\\\\~~~~~~/\\\\\\\\\\\\~~~~~~/\\\\\\\\\\\~~~~~~~/\\\\\\\\\~~~~~         
  ~~~/\\\//////////~~~~~/\\\//////////~~~~~/\\\/////////\\\~~~/\\\///////\\\~~~        
   ~~/\\\~~~~~~~~~~~~~~~/\\\~~~~~~~~~~~~~~~\//\\\~~~~~~\///~~~\///~~~~~~\//\\\~~       
    ~\/\\\~~~~/\\\\\\\~~\/\\\~~~~/\\\\\\\~~~~\////\\\~~~~~~~~~~~~~~~~~~~~/\\\/~~~      
     ~\/\\\~~~\/////\\\~~\/\\\~~~\/////\\\~~~~~~~\////\\\~~~~~~~~~~~~~~/\\\//~~~~~     
      ~\/\\\~~~~~~~\/\\\~~\/\\\~~~~~~~\/\\\~~~~~~~~~~\////\\\~~~~~~~~/\\\//~~~~~~~~    
       ~\/\\\~~~~~~~\/\\\~~\/\\\~~~~~~~\/\\\~~~/\\\~~~~~~\//\\\~~~~~/\\\/~~~~~~~~~~~   
        ~\//\\\\\\\\\\\\/~~~\//\\\\\\\\\\\\/~~~\///\\\\\\\\\\\/~~~~~/\\\\\\\\\\\\\\\~  
         ~~\////////////~~~~~~\////////////~~~~~~~\///////////~~~~~~\///////////////~~ 
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                                    Glider Guidance System 2
                                          Version 1.0.0
                                    Created by Matthew Learn

                      Need help? Send an email to matt.learn@marine.rutgers.edu
        """
    )
    # print parameters
    print(
        "----------------------------\nTicket Information:\n----------------------------"
    )
    print(f"Date: {date}")
    print(f"Depth: {depth} meters")
    print(f"Lat bounds: {lat_min} to {lat_max}")
    print(f"Lon bounds: {lon_min} to {lon_max}")
    print(f"CMEMS: {CMEMS_}")
    print(f"ESPC: {ESPC_}")
    print(f"RTOFS (East Coast): {RTOFS_EAST_}")
    print(f"RTOFS (West Coast): {RTOFS_WEST_}")
    print(f"RTOFS (Parallel): {RTOFS_PARALLEL_}")
    print(f"Compute optimal path: {compute_optimal_path}")
    print(f"Waypoints: {waypoints}")
    print(f"Raw glider speed: {glider_raw_speed} m/s")
    print(f"RMSD: {RMSD}")
    print(f"Make magnitude plot: {make_magnitude_plot}")
    print(f"Make threshold plot: {make_threshold_plot}")
    print(f"Make RMSD plot: {make_rmsd_plot}")
    print(f"Vector type: {vector_type}")
    print(f"Density: {density}")
    print(f"Scalar: {scalar}")
    print(f"Save: {save}")
    print("----------------------------\n")

    print("Starting script...")
    starttime = print_starttime()

    # initialize secondary subset parameters
    dates = (date, date)
    extent = (lat_min, lon_min, lat_max, lon_max)

    # initialize models
    cmems = CMEMS()
    espc = ESPC()
    rtofs_e = RTOFS()
    rtofs_w = RTOFS()
    rtofs_p = RTOFS()

    # get selected models
    model_select_dict = {
        cmems: CMEMS_,
        espc: ESPC_,
        rtofs_e: RTOFS_EAST_,
        rtofs_w: RTOFS_WEST_,
        rtofs_p: RTOFS_PARALLEL_,
    }
    model_list = [model for model, selected in model_select_dict.items() if selected]

    # initialize pathfinding dictionary
    optimal_paths = {}

    # get selected plotting paramters
    contour_select_dict = {
        "magnitude": make_magnitude_plot,
        "threshold": make_threshold_plot,
        "rmsd": make_rmsd_plot,
    }
    contour_type = [
        contour for contour, selected in contour_select_dict.items() if selected
    ]

    ############################## PROCESSING INDIVIDUAL MODELS ##############################
    print("\n############## PROCESSING INDIVIDUAL MODELS ##############\n")

    # make an instance of ESPC so that it can be regridded to the RTOFS dataset
    try:
        temp = ESPC()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        today = datetime.today().strftime("%Y-%m-%d")
        temp.subset((today, today), extent, depth, diag_text=False)
        temp.subset_data = temp.subset_data.isel(time=0)
        temp.z_interpolated_data = interpolate_depth(temp, depth, diag_text=False)
        temp.z_interpolated_data = calculate_magnitude(temp, diag_text=False)
        temp.da_data = depth_average(temp, diag_text=False)
    except Exception as e:
        print(
            f"ERROR: Failed to process ESPC {temp.raw_data.attrs['text_name']} data due to: {e}\n"
        )
        print("Processing CMEMS COMMON GRID instead...")

        temp = CMEMS()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        today = datetime.today().strftime("%Y-%m-%d")
        temp.subset((today, today), extent, depth, diag_text=False)
        temp.subset_data = temp.subset_data.isel(time=0)
        temp.z_interpolated_data = interpolate_depth(temp, depth, diag_text=False)
        temp.z_interpolated_data = calculate_magnitude(temp, diag_text=False)
        temp.da_data = depth_average(temp, diag_text=False)

    for model in model_list:
        # load data
        try:
            if isinstance(model, RTOFS):
                rtofs_sources = {rtofs_e: "east", rtofs_w: "west", rtofs_p: "parallel"}
                source = rtofs_sources[model]
                model.load(source)
            else:
                model.load()

            # process data
            model.subset(dates, extent, depth)  # subset
            model.subset_data = model.subset_data.isel(time=0)
            model.z_interpolated_data = interpolate_depth(model, depth)
            model.z_interpolated_data = calculate_magnitude(model)
            model.da_data = depth_average(model)
            model.xy_interpolated_data = regrid_ds(model.da_data, temp.da_data)

            # pathfinding
            if compute_optimal_path:
                optimal_paths[model] = compute_a_star_path(
                    waypoints, model.xy_interpolated_data, glider_raw_speed
                )
            else:
                optimal_paths[model] = None
                waypoints = None

            # plot da_data
            if "magnitude" in contour_type:
                create_map(
                    model.xy_interpolated_data,
                    extent,
                    "magnitude",
                    vector_type,
                    density,
                    scalar,
                    optimal_paths[model],
                    waypoints,
                    save=save,
                )
            if "threshold" in contour_type:
                create_map(
                    model.xy_interpolated_data,
                    extent,
                    "threshold",
                    vector_type,
                    density,
                    scalar,
                    optimal_paths[model],
                    waypoints,
                    save=save,
                )
        except Exception as e:
            print(
                f"ERROR: Failed to process {model.raw_data.attrs['text_name']} data due to: {e}\n"
            )
            continue

    ############################## PROCESSING MODEL COMPARISONS ##############################

    # Calculate simple means for all selected models

    # Calculate mean difference for all selected models

    # Calculate the RMSD for every non-repeating model combination
    if RMSD == True:
        print("############## PROCESSING MODEL COMPARISONS ##############\n")
        try:
            model_combos = list(itertools.combinations(model_list, r=2))
            # not going to do list comprehension on this because of readability
            rmsd_list = []
            for model1, model2 in model_combos:
                rmsd = calculate_rmsd(
                    model1.z_interpolated_data, model2.z_interpolated_data
                )
                rmsd_regrid = regrid_ds(rmsd, temp.da_data)
                rmsd_list.append(rmsd_regrid)
            if "rmsd" in contour_type:  # this doesnt need to be its own thing?
                [
                    create_map(rmsd, extent, "rmsd", None, density, scalar, save=save)
                    for rmsd in rmsd_list
                ]
        except Exception as e:
            print(f"ERROR: Failed to calculate RMSD due to: {e}\n")

    print("############## PROCESSING COMPLETE ##############\n")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return


if __name__ == "__main__":
    # TODO: add parallel processing
    main()
