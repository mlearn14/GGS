from models import *
from functions import *
from pathfinding import *
from plotting import *
import itertools

####################################################
#                                                  #
#                    PARAMETERS                    #
#                                                  #
####################################################

# Subset parameters
DATE = "2025-01-26"  # format: "YYYY-MM-DD"
DEPTH = 1100  # keep 1100 to make sure we get the next deepest layer. CMEMS data does not have a layer at 1000 meters
LAT_MIN = 34  # southern latitude
LON_MIN = -79  # western longitude
LAT_MAX = 45  # northern latitude
LON_MAX = -50  # eastern longitude

# Model selection
CMEMS_ = True  # European model
ESPC_ = True  # Navy model
RTOFS_EAST_ = True  # NOAA model for US east coast
RTOFS_WEST_ = False  # NOAA model for US west coast
RTOFS_PARALLEL_ = True  # experimental NOAA model for US east coast

# Pathfinding parameters
COMPUTE_OPIMTAL_PATH = False  # set to True or False (no quotations)
WAYPOINTS = [
    (41.240, -70.958),
    (37.992, -71.248),
    (36.943, -66.862),
    (38.666, -62.978),
    (39.801, -60.653),
    (39.618, -55.87),
]  # [(lat1, lon1), (lat2, lon2), ..., (latx, lonx)]
GLIDER_RAW_SPEED = 0.5  # m/s. This is the base speed of the glider

# Model combarison parameters
RMSD = True  # Root mean squared difference. Set to True or False (no quotations)

# Plotting parameters
MAKE_MAGNITUDE_PLOT = False  # set to True or False (no quotations)
MAKE_THRESHOLD_PLOT = False  # set to True or False (no quotations)
MAKE_RMSD_PLOT = True  # set to True or False (no quotations)

VECTOR_TYPE = "streamplot"  # "quiver", "streamplot", or None (no quotations arond None)
STREAMLINE_DENSITY = 5  # Higer number = more streamlines
QUIVER_DOWNSCALING = 4  # Lower number = more dense quivers

SAVE_FIGURES = True  # Set to True or False (no quotations)


#######################################################################################
def main(
    date: str,
    depth: int,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    CMEMS_: bool,
    ESPC_: bool,
    RTOFS_EAST_: bool,
    RTOFS_WEST_: bool,
    RTOFS_PARALLEL_: bool,
    compute_optimal_path: bool,
    waypoints: list[tuple[float, float]],
    glider_raw_speed: float,
    RMSD: bool,
    make_magnitude_plot: bool,
    make_threshold_plot: bool,
    make_rmsd_plot: bool,
    vector_type: str,
    density: int,
    scalar: int,
    save: bool,
) -> None:
    """
    Main entry point for the script.

    Args:
    ----------
        - date (str): The date of the glider data in the format "YYYY-MM-DD".
        - depth (int): The depth of the glider data in meters.
        - lat_min (float): The minimum latitude of the glider data.
        - lon_min (float): The minimum longitude of the glider data.
        - lat_max (float): The maximum latitude of the glider data.
        - lon_max (float): The maximum longitude of the glider data.
        - CMEMS_ (bool): Whether to use the CMEMS model.
        - ESPC_ (bool): Whether to use the ESPC model.
        - RTOFS_EAST_ (bool): Whether to use the RTOFS model for the US east coast.
        - RTOFS_WEST_ (bool): Whether to use the RTOFS model for the US west coast.
        - RTOFS_PARALLEL_ (bool): Whether to use the experimental RTOFS model for the US east coast.
        - compute_optimal_path (bool): Whether to compute the optimal path.
        - waypoints (list[tuple[float, float]]): The waypoints to pass into the A* algorithm.
        - glider_raw_speed (float): The raw speed of the glider in meters per second.
        - RMSD (bool): Whether to calculate the root mean squared difference.
        - make_magnitude_plot (bool): Whether to make a magnitude plot.
        - make_threshold_plot (bool): Whether to make a threshold plot.
        - make_rmsd_plot (bool): Whether to make an RMSD plot.
        - vector_type (str): The type of vector to plot. Can be "quiver", "streamplot", or None.
        - density (int): The density of the streamlines.
        - scalar (int): The scalar for the quiver plots.
        - save (bool): Whether to save the plots.
    """
    ############################## INITIALIZE SCRIPT ##################################
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
    print(f"Southwest Bound: ({lat_min}°,{lon_min}°)")
    print(f"Northeast Bound: ({lat_max}°,{lon_max}°)")
    print(f"CMEMS: {CMEMS_}")
    print(f"ESPC: {ESPC_}")
    print(f"RTOFS (East Coast): {RTOFS_EAST_}")
    print(f"RTOFS (West Coast): {RTOFS_WEST_}")
    print(f"RTOFS (Parallel): {RTOFS_PARALLEL_}")
    print(f"Compute Optimal Path: {compute_optimal_path}")
    print(f"Waypoints: {waypoints}")
    print(f"Raw Glider Speed: {glider_raw_speed} m/s")
    print(f"RMSD: {RMSD}")
    print(f"Make Magnitude Plot: {make_magnitude_plot}")
    print(f"Make Mangitude Threshold Plot: {make_threshold_plot}")
    print(f"Make RMSD Plot: {make_rmsd_plot}")
    print(f"Vector Type: {vector_type}")
    print(f"Streamline Density: {density}")
    print(f"Quiver Downscale Value: {scalar}")
    print(f"Save Figures: {save}")
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
    main(
        date=DATE,
        depth=DEPTH,
        lat_min=LAT_MIN,
        lon_min=LON_MIN,
        lat_max=LAT_MAX,
        lon_max=LON_MAX,
        CMEMS_=CMEMS_,
        ESPC_=ESPC_,
        RTOFS_EAST_=RTOFS_EAST_,
        RTOFS_WEST_=RTOFS_WEST_,
        RTOFS_PARALLEL_=RTOFS_PARALLEL_,
        compute_optimal_path=COMPUTE_OPIMTAL_PATH,
        waypoints=WAYPOINTS,
        glider_raw_speed=GLIDER_RAW_SPEED,
        RMSD=RMSD,
        make_magnitude_plot=MAKE_MAGNITUDE_PLOT,
        make_threshold_plot=MAKE_THRESHOLD_PLOT,
        make_rmsd_plot=MAKE_RMSD_PLOT,
        vector_type=VECTOR_TYPE,
        density=STREAMLINE_DENSITY,
        scalar=QUIVER_DOWNSCALING,
        save=SAVE_FIGURES,
    )
