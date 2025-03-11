# author: matthew learn (matt.learn@marine.rutgers.edu)
import argparse

import ggs2.util as util
import ggs2.maps as maps
import ggs2.model_processing as mp
import ggs2.parameters as prm


def main():
    starttime = util.print_starttime()

    # parse config file name from program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    args = parser.parse_args()
    config_name = args.config_name

    # read config file
    config = util.read_config(config_name)

    # logo text
    util.logo_text()

    # initialize paramters
    parameters = prm.initialize_parameters(config)

    # read back ticket dict
    prm.ticket_report(parameters)

    # load common grid
    common_grid = mp.process_common_grid(parameters["extent"], parameters["depth"])

    # process and plot individual model data
    plot_params = {
        "extent": parameters["extent"],
        "density": parameters["density"],
        "scalar": parameters["scalar"],
        "mission_fname": parameters["mission_fname"],
        "save": parameters["save_figures"],
    }

    for obj in parameters["models"]:
        obj.load()
        # calculates the depth interpoalated and depth averaged current
        mp.process_individual_model(
            obj,
            common_grid,
            parameters["dates"],
            parameters["extent"],
            parameters["depth"],
            parameters["single_date"],
            parameters["pathfinding"],
            parameters["waypoints"],
            parameters["glider_speed"],
            parameters["mission_fname"],
        )

        # create map of depth averaged current data
        if parameters["individual_plots"]:
            plot_params["data"] = obj.da_data
            plot_params["vector_type"] = parameters["vector_type"]
            plot_params["optimal_path"] = obj.optimal_path
            plot_params["waypoints"] = obj.waypoints
            plot_params["comp_plot"] = False

            for contour in parameters["contours"]:
                plot_params["contour_type"] = contour
                maps.create_map(**plot_params)

    if parameters["comparison_plots"] and parameters["model_combos"]:
        for comp in parameters["comparison_plots"]:
            # initialize constant comparison plot parameters
            plot_params["contour_type"] = comp
            plot_params["optimal_path"] = None
            plot_params["waypoints"] = None
            plot_params["comp_plot"] = True

            # get data and vector_type for each comparison
            if comp in ["speed_diff", "u_diff", "v_diff"]:
                plot_data = [
                    mp.calculate_simple_diff(model1, model2, diag_text=False)
                    for model1, model2 in parameters["model_combos"]
                ]
                plot_params["vector_type"] = None
            elif comp == "rmsd_profile":
                plot_data = [
                    mp.calculate_rms_vertical_diff(model1, model2, diag_text=False)
                    for model1, model2 in parameters["model_combos"]
                ]
                plot_params["vector_type"] = None
            elif comp in ["mean_diff", "simple_mean"]:
                plot_data = [
                    (
                        mp.calculate_mean_diff(parameters["models"], diag_text=False)
                        if comp == "mean_diff"
                        else mp.calculate_simple_mean(
                            parameters["models"], diag_text=False
                        )
                    )
                ]
                plot_params["vector_type"] = parameters["vector_type"]

            if comp == "simple_mean":
                for contour in parameters["contours"]:
                    plot_params["data"] = plot_data[0]
                    plot_params["contour"] = (
                        "mean_magnitude"
                        if contour == "magnitude"
                        else "mean_threshold" if contour == "threshold" else contour
                    )
                    maps.create_map(**plot_params)
            else:
                for data in plot_data:
                    plot_params["data"] = data
                    maps.create_map(**plot_params)

    endtime = util.print_endtime()
    print("Script complete.")
    util.print_runtime(starttime, endtime)
    print("Thank you for using Glider Guidance System 2.")


if __name__ == "__main__":
    main()
