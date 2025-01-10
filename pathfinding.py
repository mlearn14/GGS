# author: Salvatore Fricano
# edited and implemented by Matthew Learn (matt.learn@marine.rutgers.edu)

import csv
import heapq
from math import radians, cos, sin, asin, sqrt
import numpy as np

from functions import *


def compute_a_star_path(
    waypoints_list: list, model: object, glider_raw_speed: float = 0.5
) -> list:
    """
    Calculates the optimal path between waypoints for a mission, considering the impact of ocean currents and distance.

    Args:
    ----------
    waypoints_list (list[(lat, lon)]): A list of latitude and longitude tuples representing the waypoints.
    model (object): An object representing the ocean currents model.
    glider_raw_speed (float, optional): The glider's base speed in meters per second. Defaults to 0.5.

    Returns:
    ----------
    optimal_mission_path (list): A list of latitude and longitude tuples representing the optimal route.
    """
    # Code adapted from Salvatore Fricano. Updated to match current code structure of GGS.

    ### HELPER FUNCTIONS ###

    # Coordinate conversion functions
    def coord_to_grid(
        lat: float, lon: float, lat_array: np.ndarray, lon_array: np.ndarray
    ) -> tuple:
        """
        Converts geographical latitude and longitude to the nearest index on the dataset grid.

        Args:
        -----------
            - lat (float): Latitude in degrees.
            - lon (float): Longitude in degrees.
            - lat_array (np.ndarray): 1D array of latitude values.
            - lon_array (np.ndarray): 1D array of longitude values.

        Returns:
        -----------
            - lat_index (int): Index of the nearest latitude value.
            - lon_index (int): Index of the nearest longitude value.
        """
        lat_index = np.argmin(np.abs(lat_array - lat))
        lon_index = np.argmin(np.abs(lon_array - lon))

        return lat_index, lon_index

    def grid_to_coord(
        lat_index: float, lon_index: float, lat_array: np.ndarray, lon_array: np.ndarray
    ) -> tuple:
        """
        Converts dataset grid indices back to geographical latitude and longitude coordinates.

        Args:
        -----------
            - lat_index (int): Index of the nearest latitude value.
            - lon_index (int): Index of the nearest longitude value.
            - lat_array (np.ndarray): 1D array of latitude values.
            - lon_array (np.ndarray): 1D array of longitude values.

        Returns:
        -----------
            - lat (float): Latitude in degrees.
            - lon (float): Longitude in degrees.
        """
        lat = lat_array[lat_index]
        lon = lon_array[lon_index]

        return lat, lon

    # Neighbor node generation function
    def generate_neighbors(index: tuple, lat_array: np.ndarray, lon_array: np.ndarray):
        """
        Generates neighboring index nodes for exploration based on the current index's position.

        Args:
        -----------
            - index (tuple): Current index node.
            - lat_array (np.ndarray): 1D array of latitude values.
            - lon_array (np.ndarray): 1D array of longitude values.

        Yields:
        -----------
            - neighbor (tuple): Neighboring index as a tuple (lat_idx2, lon_idx2).
        """
        # Unpack the current index into latitude and longitude indices
        lat_idx, lon_idx = index

        # Iterate over the possible changes in latitude index (-1, 0, 1)
        for delta_lat in [-1, 0, 1]:
            # Iterate over the possible changes in longitude index (-1, 0, 1)
            for delta_lon in [-1, 0, 1]:
                # Skip the current index itself (no change in both lat and lon)
                if delta_lat == 0 and delta_lon == 0:
                    continue

                # Calculate the new indices by applying the changes
                lat_idx2, lon_idx2 = (
                    lat_idx + delta_lat,
                    lon_idx + delta_lon,
                )

                # Ensure the new indices are within the bounds of the arrays
                if 0 <= lat_idx2 < len(lat_array) and 0 <= lon_idx2 < len(lon_array):
                    # Yield the valid neighboring index as a tuple
                    yield (lat_idx2, lon_idx2)

    # Distance calculation functions
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates the great circle distance between two points on the earth using the Haversine formula.

        Args:
        -----------
            - lat1 (float): Latitude of the first point in degrees.
            - lon1 (float): Longitude of the first point in degrees.
            - lat2 (float): Latitude of the second point in degrees.
            - lon2 (float): Longitude of the second point in degrees.

        Returns:
        -----------
            - distance (float): The great circle distance between the two points in meters.
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1

        # Haversine fomula
        a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371000

        return distance

    def direct_distance(
        start_index: tuple,
        end_index: tuple,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        glider_raw_speed: float,
    ):
        """
        Calculates the direct distance and time cost between two grid points. Fallback if no optimal path is found.

        Args:
        -----------
            - start_index (tuple): Index of the starting grid point.
            - end_index (tuple): Index of the ending grid point.
            - lat_array (np.ndarray): 1D array of latitude values.
            - lon_array (np.ndarray): 1D array of longitude values.
            - glider_raw_speed (float): The glider's base speed in meters per second.

        Returns:
        -----------
            - path (list): List of latitude and longitude tuples representing the direct path.
            - time (float): The time cost of the direct path in seconds.
            - distance (float): The distance cost of the direct path in meters.
        """
        start_lat, start_lon = grid_to_coord(*start_index, lat_array, lon_array)
        end_lat, end_lon = grid_to_coord(*end_index, lat_array, lon_array)
        path = [(start_lat, start_lon), (end_lat, end_lon)]

        distance = haversine_distance(start_lon, start_lat, end_lon, end_lat)
        time = distance / glider_raw_speed

        return path, time, distance

    def calculate_movement(
        ds: xr.Dataset,
        start_index: tuple,
        end_index: tuple,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        glider_raw_speed: float,
    ):
        """
        Calculates the time and distance cost of moving from one grid point to the next, considering ocean currents.

        This function takes the following parameters:

        - ds (xr.Dataset): The dataset containing ocean current data.
        - start_index (tuple): The starting grid point as a tuple of (latitude index, longitude index).
        - end_index (tuple): The ending grid point as a tuple of (latitude index, longitude index).
        - lat_array (np.ndarray): 1D array of latitude values.
        - lon_array (np.ndarray): 1D array of longitude values.
        - glider_raw_speed (float): The glider's base speed in meters per second.

        The function returns a tuple containing the time cost and distance cost of moving from the start index to the end index.
        """
        start_lat, start_lon = grid_to_coord(*start_index, lat_array, lon_array)
        end_lat, end_lon = grid_to_coord(*end_index, lat_array, lon_array)

        # If the start and end indices are the same, return 0 time and distance cost
        if start_lat == end_lat and start_lon == end_lon:
            return 0, 0

        # Calculate the heading vector from the start to the end point
        heading_vector = np.array([end_lon - start_lon, end_lat - start_lat])
        norm = np.linalg.norm(heading_vector)

        # If the start and end points are the same, return 0 time and distance cost
        if norm == 0:
            return 0, 0

        # Normalize the heading vector
        heading_vector = heading_vector / norm

        # Get the current velocity at the start point
        u_inst = ds.u.isel(lat=start_index[0], lon=start_index[1]).values.item()
        v_inst = ds.v.isel(lat=start_index[0], lon=start_index[1]).values.item()
        inst_vector = np.array([u_inst, v_inst])

        # Calculate the current velocity along the heading vector
        current_along_heading = np.dot(inst_vector, heading_vector)

        # Calculate the net speed by adding the glider's raw speed to the current velocity
        net_speed = glider_raw_speed + current_along_heading

        # Ensure the net speed is at least 0.1 m/s
        net_speed = max(net_speed, 0.1)

        # Calculate the distance from the start to the end point
        distance = haversine_distance(start_lon, start_lat, end_lon, end_lat)

        # Calculate the time cost by dividing the distance by the net speed
        time = distance / net_speed

        return time, distance

    # Movement cost function
    def calculate_heuristic_cost(
        inst_index: tuple,
        goal_index: tuple,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
    ) -> float:
        """Estimates the cost from the current index to the goal using the Haversine formula as a heuristic.

        Args:
        -----------
            - inst_index (tuple): Tuple containing the instance latitude and longitude.
            - goal_index (tuple): Tuple containing the goal latitude and longitude.
            - lat_array (np.ndarray): 1D array of latitude values.
            - lon_array (np.ndarray): 1D array of longitude values.

        Returns:
        -----------
            - heuristic_cost (float): Estimated cost from the current index to the goal.
        """
        inst_lat, inst_lon = grid_to_coord(*inst_index, lat_array, lon_array)
        goal_lat, goal_lon = grid_to_coord(*goal_index, lat_array, lon_array)

        heuristic_cost = haversine_distance(inst_lon, inst_lat, goal_lon, goal_lat)

        return heuristic_cost

    # Path reconstruction function
    def reconstruct_path(
        came_from_dict: dict,
        start_idx: tuple,
        goal_idx: tuple,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
    ):
        """
        Reconstructs the path from the start index to the goal index using the came_from dictionary populated by the A* algorithm.

        The came_from dictionary is a mapping of each index to the index that it came from during the A* search. The path is reconstructed by starting from the goal index and tracing back through the came_from dictionary until the start index is reached.

        Args:
        -----------
            - came_from_dict (dict): Dictionary containing the came_from information for each index.
            - start_idx (tuple): Tuple containing the start latitude and longitude.
            - goal_idx (tuple): Tuple containing the goal latitude and longitude.
            - lat_array (np.ndarray): 1D array of latitude values.
            - lon_array (np.ndarray): 1D array of longitude values.

        Returns:
        -----------
            - optimal_path_coords (list): List of latitude and longitude tuples representing the optimal path from the start to the goal.
        """
        optimal_path = [goal_idx]

        # Start from the goal index and trace back through the came_from dictionary
        # until the start index is reached.
        while goal_idx != start_idx:
            # Get the index that the current goal index came from.
            goal_idx = came_from_dict[goal_idx]
            # Add the new index to the beginning of the optimal path.
            optimal_path.append(goal_idx)

        # Reverse the optimal path so that it goes from start to goal.
        optimal_path.reverse()
        # Convert the optimal path from a list of indices to a list of latitude and longitude coordinates.
        optimal_path_coords = [
            grid_to_coord(*idx, lat_array, lon_array) for idx in optimal_path
        ]

        return optimal_path_coords

    # A* algorithm
    def algorithm_a_star(
        ds: xr.Dataset,
        start_idx: tuple,
        end_idx: tuple,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        glider_raw_speed: float,
    ) -> tuple:
        """
        Applies the A* algorithm to find the optimal path between waypoints, considering ocean currents.

        Args:
        -----------
            - ds (xr.Dataset): Dataset containing ocean current data.
            - start_idx (tuple): Tuple containing the start latitude and longitude.
            - end_idx (tuple): Tuple containing the goal latitude and longitude.
            - lat_array (np.ndarray): 1D array of latitude values.
            - lon_array (np.ndarray): 1D array of longitude values.
            - glider_raw_speed (float): Raw speed of the glider.

        Returns:
        -----------
            - path (list): List of latitude and longitude tuples representing the optimal path from the start to the goal.
            - time (float): Time cost of the optimal path in seconds.
            - distance (float): Distance cost of the optimal path in meters.
        """

        # Initialize the A* algorithm
        open_set = [
            (
                # Calculate the heuristic cost from the start index to the end index
                calculate_heuristic_cost(start_idx, end_idx, lat_array, lon_array),
                start_idx,
            )
        ]

        # Create a dictionary to store the came_from information for each index
        came_from = {start_idx: None}

        # Create a dictionary to store the g_score (cost from start to index) for each index
        g_score = {start_idx: 0}

        # Create a dictionary to store the f_score (cost from start to index + heuristic cost from index to goal) for each index
        f_score = {start_idx: calculate_heuristic_cost(start_idx, end_idx)}

        # Initialize a flag to indicate if a path has been found
        path_found = False

        # Loop through the open set until it is empty
        while open_set:
            # Get the index with the lowest f_score from the open set
            _, current = heapq.heappop(open_set)

            # If the current index is the end index, then the path has been found
            if current == end_idx:
                path_found = True
                break

            # Loop through all the neighbors of the current index
            for neighbor in generate_neighbors(current, lat_array, lon_array):
                # Calculate the tentative g_score for the neighbor
                tent_g_score = (
                    g_score[current]
                    + calculate_movement(ds, current, neighbor, glider_raw_speed)[1]
                )

                # If the tentative g_score is less than the current g_score for the neighbor,
                # then update the g_score and came_from information for the neighbor
                if tent_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tent_g_score
                    f_score[neighbor] = tent_g_score + calculate_heuristic_cost(
                        neighbor, end_idx
                    )

                    # If the neighbor is not already in the open set, add it
                    if neighbor not in [n for _, n in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If a path has been found, reconstruct the path from the came_from information
        if path_found:
            path = reconstruct_path(came_from, start_idx, end_idx)

            # Calculate the time and distance cost of the optimal path
            time, distance = calculate_movement(
                ds, start_idx, end_idx, glider_raw_speed
            )

        # If no path has been found, use the direct distance from start to end
        # and calculate the time and distance cost of the direct path
        else:
            print(
                f"Direct path used from {grid_to_coord(*start_idx, lat_array, lon_array)} to {grid_to_coord(*end_idx, lat_array, lon_array)}."
            )
            path, time, distance = direct_distance(start_idx, end_idx, glider_raw_speed)

        # Return the optimal path, time cost, and distance cost
        return path, time, distance

    ### MAIN FUNCTION CODE ###
    # Define variables
    ds = model.da_data
    text_name = ds.attrs["text_name"]
    model_name = ds.attrs["model_name"]

    # Initialize a list to store the CSV data
    csv_data = [
        ("Segment Start", "Segment End", "Segment Time (s)", "Segment Distance (m)")
    ]

    print(f"{text_name}: Calculating A* optimal path...")
    starttime = print_starttime()

    # Get the latitude and longitude arrays from the data
    lat_array = ds.lon.values
    lon_array = ds.lon.values

    # Initialize an empty list to store the optimal mission path
    optimal_mission_path = []
    # Initialize variables to store the total time and distance
    total_time = 0
    total_distance = 0

    # Loop through each pair of waypoints
    for i in range(len(waypoints_list) - 1):
        # Get the start and end indices from the current waypoint pair
        start_idx = coord_to_grid(*waypoints_list[i], lat_array, lon_array)
        end_idx = coord_to_grid(*waypoints_list[i + 1], lat_array, lon_array)

        # Run the A* algorithm to get the optimal path, time, and distance for the current segment
        segment_path, segment_time, segment_distance = algorithm_a_star(
            ds, start_idx, end_idx, glider_raw_speed
        )
        # Extend the optimal mission path with the current segment path (excluding the last point)
        optimal_mission_path.extend(segment_path[:-1])

        # Add the current segment's time and distance to the total
        total_time += segment_time
        total_distance += segment_distance

        # Append the current segment's data to the CSV list
        csv_data.append(
            (
                waypoints_list[i],
                waypoints_list[i + 1],
                segment_time,
                segment_distance,
            )
        )
        # Print a message to show the current segment's data
        print(
            f"Segment {i+1}: Start {waypoints_list[i]} End {waypoints_list[i+1]} Time {segment_time} seconds Distance {segment_distance} meters"
        )

    # Add the last waypoint to the optimal mission path
    optimal_mission_path.append(waypoints_list[-1])

    # Print the total mission time (adjusted) and distance
    print(f"Total mission time (adjusted): {total_time} seconds")
    print(f"Total mission distance: {total_distance} meters")

    # Define the path for the CSV file
    csv_file_path = os.path.join(
        "data/mission_statistics", f"{model_name}_mission_statistics.csv"
    )
    # Open the CSV file and write the data to it
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)
    print()

    return optimal_mission_path

# TODO: Add a new algorithm for finding the optimal path. 
# Might need to rework A* function to have helper functions be on the outside. We shall see.