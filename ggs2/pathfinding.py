# author: Salvatore Fricano
# edited and implemented by Matthew Learn (matt.learn@marine.rutgers.edu)

import csv
import heapq
from math import radians, cos, sin, asin, sqrt
import os

import numpy as np
import simplekml
import xarray as xr

from .util import print_starttime, print_endtime, print_runtime


def ensure_land_mask(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensures the land mask is present in the dataset.

    Args
    -----------
        ds (xr.Dataset): The dataset to ensure the land mask is present in.

    Returns
    -----------
        xr.Dataset: The dataset with the land mask present.
    """
    if "land_mask" not in ds:
        ds["land_mask"] = xr.ufuncs.isnan(ds.u) | xr.ufuncs.isnan(ds.v)
    return ds


def compute_a_star_path(
    waypoints_list: list[tuple],
    model: object,
    heuristic: str,
    glider_raw_speed: float,
    mission_name: str = None,
    heuristic_weight: float = 1.2,
) -> list:
    """
    Calculates the optimal path between waypoints for a mission, considering the impact of ocean currents and distance.

    Args
    ----------
        waypoints_list (list)
            A list of latitude and longitude tuples representing the waypoints.
            - List format: [(lat1, lon1), (lat2, lon2), ...]
        model (object)
            Model object.
        heuristic (str)
            Heuristic to use for the A* algorithm. Options: "drift_aware", "haversine".
        glider_raw_speed (float, optional)
            The glider's base horizontal speed in meters per second.
        mission_name (str, optional)
            Name of the mission (default is None).
        heuristic_weight (float, optional)
            Weight to apply to the heuristic cost in the A* algorithm. Default is 1.2.

    Returns
    ----------
        optimal_mission_path (list): A list of latitude and longitude tuples representing the optimal route.
    """
    # Code adapted from Salvatore Fricano. Updated to match current code structure of GGS.

    ### HELPER FUNCTIONS ###

    # Coordinate conversion functions
    def coord_to_grid(
        lat: float, lon: float, lat_array: np.ndarray, lon_array: np.ndarray
    ) -> tuple[float, float]:
        """
        Converts geographical latitude and longitude to the nearest index on the dataset grid.

        Args
        -----------
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            lat_array (np.ndarray): 1D array of latitude values.
            lon_array (np.ndarray): 1D array of longitude values.

        Returns
        -----------
            lat_index (int): Index of the nearest latitude value.
            lon_index (int): Index of the nearest longitude value.
        """
        lat_index = np.argmin(np.abs(lat_array - lat))
        lon_index = np.argmin(np.abs(lon_array - lon))

        return lat_index, lon_index

    def grid_to_coord(
        lat_index: float, lon_index: float, lat_array: np.ndarray, lon_array: np.ndarray
    ) -> tuple[float, float]:
        """
        Converts dataset grid indices back to geographical latitude and longitude coordinates.

        Args
        -----------
            lat_index (int): Index of the nearest latitude value.
            lon_index (int): Index of the nearest longitude value.
            lat_array (np.ndarray): 1D array of latitude values.
            lon_array (np.ndarray): 1D array of longitude values.

        Returns
        -----------
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
        """
        lat = lat_array[lat_index]
        lon = lon_array[lon_index]

        return lat, lon

    # Neighbor node generation function
    def generate_neighbors(
        index: tuple,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        ds: xr.Dataset,
        max_leg_distance_m: float = 55550,
    ):
        """
        Generates neighboring index nodes for exploration based on the current index's position, filtering by land mask and maximum segment length.

        Args
        -----------
            index (tuple): Current index node.
            lat_array (np.ndarray): 1D array of latitude values.
            lon_array (np.ndarray): 1D array of longitude values.
            ds (xr.Dataset): xarray dataset containing current data.
            max_leg_distance_m (float): Maximum leg distance in meters. Default is 55550 meters (0.5 degree of latitude).

        Yields
        -----------
            neighbor (tuple)
                Neighboring index as a tuple (lat_idx2, lon_idx2).
        """
        # Unpack the current index into latitude and longitude indices
        lat_idx, lon_idx = index

        # Iterate over all possible neighboring indices
        for d_lat in [-1, 0, 1]:
            for d_lon in [-1, 0, 1]:
                if d_lat == 0 and d_lon == 0:
                    continue

                n_lat, n_lon = lat_idx + d_lat, lon_idx + d_lon

                if 0 <= n_lat < len(lat_array) and 0 <= n_lon < len(lon_array):
                    try:
                        if ds.land_mask.isel(lat=n_lat, lon=n_lon).values.item():
                            continue
                        dist = haversine_distance(
                            lat_array[lat_idx],
                            lon_array[lon_idx],
                            lat_array[n_lat],
                            lon_array[n_lon],
                        )
                        if dist <= max_leg_distance_m:
                            yield (n_lat, n_lon)
                        else:
                            print(
                                f"Skipping neighbor {n_lat}, {n_lon} due to distance {(dist / 1000):.2f} km."
                            )
                    except Exception as e:
                        print(f"Skipping neighbor {n_lat}, {n_lon} due to {e}.")
                        continue

    # Distance calculation functions
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates the great circle distance between two points on the earth using the Haversine formula.

        Args
        -----------
            lat1 (float): Latitude of the first point in degrees.
            lon1 (float): Longitude of the first point in degrees.
            lat2 (float): Latitude of the second point in degrees.
            lon2 (float): Longitude of the second point in degrees.

        Returns
        -----------
            distance (float): The great circle distance between the two points in meters.
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1

        # Haversine fomula
        a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371000

        return distance

    def calculate_drift_aware_heuristic_cost(
        inst_index: tuple,
        goal_index: tuple,
        u_array: np.ndarray,
        v_array: np.ndarray,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        glider_raw_speed: float,
    ) -> float:
        """
        Estimates the cost from the current index to the goal, considering the benefit of ocean currents and accounting for drift.
        Falls back to a straight-line distance if ocean current data is unavailable.

        Args
        -----------
            inst_index (tuple)
                Tuple containing the instance latitude and longitude.
            goal_index (tuple)
                Tuple containing the goal latitude and longitude.
            u_array (np.ndarray)
                2D array of ocean current velocities in the x direction.
            v_array (np.ndarray)
                2D array of ocean current velocities in the y direction.
            lat_array (np.ndarray)
                1D array of latitude values.
            lon_array (np.ndarray)
                1D array of longitude values.
            ds (xr.Dataset)
                xarray dataset containing current data.
            glider_raw_speed (float)
                The glider's base speed in meters per second.

        Returns
        -----------
            heuristic_cost (float)
                Estimated cost from the current index to the goal.
        """
        # Convert grid indices to coordinates
        inst_lat, inst_lon = grid_to_coord(*inst_index, lat_array, lon_array)
        goal_lat, goal_lon = grid_to_coord(*goal_index, lat_array, lon_array)

        # Compute base heuristic using Haversine distance
        base_heuristic = haversine_distance(inst_lat, inst_lon, goal_lat, goal_lon)
        net_speed = compute_effective_speed(
            u_array,
            v_array,
            lat_array,
            lon_array,
            inst_lat,
            inst_lon,
            goal_lat,
            goal_lon,
            glider_raw_speed,
        )

        if net_speed is None or net_speed <= 0:
            return base_heuristic / glider_raw_speed

        return base_heuristic / net_speed

    def direct_distance(
        start_index: tuple,
        end_index: tuple,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        glider_raw_speed: float,
    ) -> tuple[list[tuple[float, float]], float, float]:
        """
        Calculates the direct distance and time cost between two grid points. Fallback if no optimal path is found.

        Args
        -----------
            start_index (tuple): Index of the starting grid point.
            end_index (tuple): Index of the ending grid point.
            lat_array (np.ndarray): 1D array of latitude values.
            lon_array (np.ndarray): 1D array of longitude values.
            glider_raw_speed (float): The glider's base speed in meters per second.

        Returns
        -----------
            path (list[tuple[float, float]]): List of latitude and longitude tuples representing the direct path.
            time (float): The time cost of the direct path in seconds.
            distance (float): The distance cost of the direct path in meters.
        """
        start_lat, start_lon = grid_to_coord(*start_index, lat_array, lon_array)
        end_lat, end_lon = grid_to_coord(*end_index, lat_array, lon_array)
        path = [(start_lat, start_lon), (end_lat, end_lon)]

        distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
        time = distance / glider_raw_speed

        return path, time, distance

    def compute_effective_speed(
        u_array: np.ndarray,
        v_array: np.ndarray,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        glider_raw_speed: float,
        clamp: bool = True,
    ) -> float | None:
        """
        Calculates the effective speed of a glider toward a target considering ocean currents.
        Returns None if current data is invalid or missing.

        Args
        -----------
            u_array (np.ndarray): 2D array of u wind components.
            v_array (np.ndarray): 2D array of v wind components.
            lat_array (np.ndarray): 1D array of latitude values.
            lon_array (np.ndarray): 1D array of longitude values.
            start_lat (float): Starting latitude.
            start_lon (float): Starting longitude.
            end_lat (float): Target latitude.
            end_lon (float): Target longitude.
            glider_raw_speed (float): The glider's base speed in meters per second.
            clamp (bool, optional): Whether to clamp the effective speed between 0.05 and 1.5 * glider_raw_speed. Defaults to True.

        Returns
        -----------
            effective_speed (float | None)
                Effective speed in meters per second, or None if data is invalid or missing.
        """
        start_idx = coord_to_grid(start_lat, start_lon, lat_array, lon_array)
        end_idx = coord_to_grid(end_lat, end_lon, lat_array, lon_array)

        # Calculate the vector from the current location to the target
        heading_vector = np.array([end_lon - start_lon, end_lat - start_lat])
        norm = np.linalg.norm(heading_vector)

        # If the target is the same as the current location, the heading is undefined
        if norm == 0:
            return 0

        # Normalize the heading vector
        heading_vector /= norm

        # Get the current vector from the current location
        try:
            u_inst = u_array[start_idx[0], start_idx[1]]
            v_inst = v_array[start_idx[0], start_idx[1]]
        except Exception:
            # If the current data is invalid or missing, return None
            return None

        # If the current data is invalid or missing, return None
        if np.isnan(u_inst) or np.isnan(v_inst):
            return None

        # Calculate the current vector
        inst_vector = np.array([u_inst, v_inst])

        # Calculate the projection of the current vector onto the heading vector
        current_along_heading = np.dot(inst_vector, heading_vector)

        # Calculate the net speed by adding the glider's base speed and the current
        net_speed = glider_raw_speed + current_along_heading

        # If clamping is enabled, clamp the net speed between 0.05 and 1.5 * glider_raw_speed
        if clamp:
            net_speed = np.clip(net_speed, 0.05, 1.5 * glider_raw_speed)

        # Return the net speed
        return net_speed

    def calculate_movement(
        start_index: tuple,
        end_index: tuple,
        u_array: np.ndarray,
        v_array: np.ndarray,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        glider_raw_speed: float,
    ) -> tuple:
        """
        Calculates the time and distance of moving from one grid point to the next, considering ocean currents.

        Parameters
        ----------
        start_index : tuple
            The starting grid point as a tuple of (latitude index, longitude index).
        end_index : tuple
            The ending grid point as a tuple of (latitude index, longitude index).
        u_array : np.ndarray
            2D array of u wind components.
        v_array : np.ndarray
            2D array of v wind components.
        lat_array : np.ndarray
            1D array of latitude values.
        lon_array : np.ndarray
            1D array of longitude values.
        glider_raw_speed : float
            The glider's base speed in meters per second.

        Returns
        -------
        tuple
            A tuple containing the time and distance of moving from the start index to the end index.
        """

        # Get the coordinates of the start and end points
        start_lat, start_lon = grid_to_coord(*start_index, lat_array, lon_array)
        end_lat, end_lon = grid_to_coord(*end_index, lat_array, lon_array)

        # If the start and end points are the same, return a time of 0 and a distance of 0
        if start_lat == end_lat and start_lon == end_lon:
            return 0, 0

        # Calculate the distance between the start and end points
        distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)

        # Calculate the effective speed from the start point to the end point
        net_speed = compute_effective_speed(
            u_array,
            v_array,
            lat_array,
            lon_array,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            glider_raw_speed,
        )

        # If the effective speed is invalid or 0, use the glider's base speed
        if net_speed is None or net_speed <= 0:
            time = distance / glider_raw_speed
        else:
            time = distance / net_speed

        return time, distance

    # Movement cost function
    def calculate_heuristic_cost(
        inst_index: tuple,
        goal_index: tuple,
        u_array: np.ndarray,
        v_array: np.ndarray,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        heuristic: str,
    ) -> float:
        """Estimates the cost from the current index to the goal using the Haversine formula as a heuristic.

        Args
        -----------
            inst_index (tuple)
                Tuple containing the instance latitude and longitude.
            goal_index (tuple)
                Tuple containing the goal latitude and longitude.
            u_array (np.ndarray)
                2D array of ocean current velocities in the x direction.
            v_array (np.ndarray)
                2D array of ocean current velocities in the y direction.
            lat_array (np.ndarray)
                1D array of latitude values.
            lon_array (np.ndarray)
                1D array of longitude values.
            heuristic (str)
                Heuristic to use for the A* algorithm. Options: "drift_aware", "haversine".

        Returns
        -----------
            heuristic_cost (float)
                Estimated cost from the current index to the goal.
        """
        inst_lat, inst_lon = grid_to_coord(*inst_index, lat_array, lon_array)
        goal_lat, goal_lon = grid_to_coord(*goal_index, lat_array, lon_array)

        if heuristic == "drift_aware":
            return calculate_drift_aware_heuristic_cost(
                inst_index,
                goal_index,
                u_array,
                v_array,
                lat_array,
                lon_array,
                glider_raw_speed,
            )
        elif heuristic == "haversine":
            return haversine_distance(inst_lat, inst_lon, goal_lat, goal_lon)
        else:
            raise ValueError(
                f"Unknown heuristic: {heuristic}. Supported heuristics: 'drift_aware', 'haversine'"
            )

    # A* algorithm
    def algorithm_a_star(
        start_idx: tuple[int, int],
        end_idx: tuple[int, int],
        ds: xr.Dataset,
        u_array: np.ndarray,
        v_array: np.ndarray,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
        glider_raw_speed: float,
        heuristic: str,
        heuristic_weight: float = 1.2,
    ) -> tuple[list[tuple[float, float]], float, float, list[float], list[float]]:
        """
        A* pathfinding algorithm optimized for glider missions with ocean current-aware cost and heuristics.
        Returns the optimal path, total time, total distance, and per-segment time and distance lists.

        Args
        -----------
            start_idx (tuple[int, int]): Tuple containing the start latitude and longitude.
            end_idx (tuple[int, int]): Tuple containing the end latitude and longitude.
            ds (xr.Dataset): xarray dataset containing current data.
            u_array (np.ndarray): 2D array of ocean current velocities in the x direction.
            v_array (np.ndarray): 2D array of ocean current velocities in the y direction.
            lat_array (np.ndarray): 1D array of latitude values.
            lon_array (np.ndarray): 1D array of longitude values.
            glider_raw_speed (float): The glider's base speed in meters per second.
            heuristic (str): Heuristic to use for the A* algorithm. Options: "drift_aware", "haversine".
            heuristic_weight (float): Weight to apply to the heuristic cost in the A* algorithm. Default is 1.2.

        Returns
        -----------
            optimal_path_coords (list[tuple[float, float]]): List of latitude and longitude tuples representing the optimal path from the start to the goal.
            total_time (float): Total time required to traverse the optimal path.
            total_distance (float): Total distance required to traverse the optimal path.
            segment_times (list[float]): List of times required to traverse each segment of the optimal path.
            segment_distances (list[float]): List of distances required to traverse each segment of the optimal path.
        """
        g_score = {start_idx: 0}
        f_score = {
            start_idx: calculate_heuristic_cost(
                start_idx, end_idx, u_array, v_array, lat_array, lon_array, heuristic
            )
        }

        open_set = [(f_score[start_idx], start_idx)]
        open_set_hash = {start_idx}
        came_from = {}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            if current in visited:
                continue
            visited.add(current)

            if current == end_idx:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                path.reverse()

                optimal_path = [
                    grid_to_coord(i[0], i[1], lat_array, lon_array) for i in path
                ]

                segment_times = []
                segment_distances = []
                total_time = 0
                total_distance = 0

                for i in range(1, len(path)):
                    seg_time, seg_dist = calculate_movement(
                        path[i - 1],
                        path[i],
                        u_array,
                        v_array,
                        lat_array,
                        lon_array,
                        glider_raw_speed,
                    )
                    segment_times.append(seg_time)
                    segment_distances.append(seg_dist)
                    total_time += seg_time
                    total_distance += seg_dist

                return (
                    optimal_path,
                    total_time,
                    total_distance,
                    segment_times,
                    segment_distances,
                )

            for neighbor in generate_neighbors(current, lat_array, lon_array, ds):
                if neighbor in visited:
                    continue

                movement_cost, _ = calculate_movement(
                    current,
                    neighbor,
                    u_array,
                    v_array,
                    lat_array,
                    lon_array,
                    glider_raw_speed,
                )
                tentative_g_score = g_score[current] + movement_cost

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_cost = calculate_heuristic_cost(
                        neighbor,
                        end_idx,
                        u_array,
                        v_array,
                        lat_array,
                        lon_array,
                        heuristic,
                    )
                    f_score[neighbor] = tentative_g_score + heuristic_weight * h_cost

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        print("A* failed to find a path. Using direct distance fallback.")
        fallback_path, fallback_time, fallback_distance = direct_distance(
            start_idx, end_idx, lat_array, lon_array, glider_raw_speed
        )
        return (
            fallback_path,
            fallback_time,
            fallback_distance,
            [fallback_time],
            [fallback_distance],
        )

    ### MAIN FUNCTION CODE ###

    # Define variables
    ds = model.da_data
    text_name = ds.attrs["text_name"]
    model_name = ds.attrs["model_name"]
    ddate = ds.time.dt.strftime("%m-%d-%Y %H:%M").values
    fdate = ds.time.dt.strftime("%Y%m%d%H").values

    # Ensure the presence of the land mask
    ds = ensure_land_mask(ds)

    # Initialize a list to store the CSV data
    path_csv_data = [("lat", "lon", "time (s)", "distance (m)")]
    csv_data = [
        (
            "Date",
            "Segment Start",
            "Segment End",
            "Segment Time (s)",
            "Segment Distance (m)",
        )
    ]

    print(f"{text_name}: Calculating A* optimal path...")
    starttime = print_starttime()

    # Ensure the waypoints are float tuples
    waypoints_list = [(float(lat), float(lon)) for lat, lon in waypoints_list]

    # Subset the dataset to only inlcude the area around the waypoints. Decreases search time.
    lats, lons = zip(*waypoints_list)
    lat_min = min(lats) - 2
    lat_max = max(lats) + 2
    lon_min = min(lons) - 2
    lon_max = max(lons) + 2
    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Get the latitude and longitude arrays from the data
    lat_array = ds.lat.values
    lon_array = ds.lon.values

    # Get the u and v wind arrays from the data
    u_array = ds.u.values
    v_array = ds.v.values

    # Initialize an empty list to store the optimal mission path
    optimal_mission_path = []
    segment_times_list = []
    segment_distances_list = []
    # Initialize variables to store the total time and distance
    total_time = 0
    total_distance = 0

    # Loop through each pair of waypoints
    for i in range(len(waypoints_list) - 1):
        # Get the start and end indices from the current waypoint pair
        start_idx = coord_to_grid(*waypoints_list[i], lat_array, lon_array)
        end_idx = coord_to_grid(*waypoints_list[i + 1], lat_array, lon_array)

        # Run the A* algorithm to get the optimal path, time, and distance for the current segment
        (
            segment_path,
            segment_total_time,
            segment_total_distance,
            segment_times,
            segment_distances,
        ) = algorithm_a_star(
            start_idx,
            end_idx,
            ds,
            u_array,
            v_array,
            lat_array,
            lon_array,
            glider_raw_speed,
            heuristic,
            heuristic_weight,
        )
        # Extend the optimal mission path with the current segment path (excluding the last point)
        optimal_mission_path.extend(segment_path[:-1])

        # Add the current segment's time and distance to the total
        total_time += segment_total_time
        total_distance += segment_total_distance

        # Append the current segment's data to the CSV list
        csv_data.append(
            (
                ddate,
                waypoints_list[i],
                waypoints_list[i + 1],
                segment_total_time,
                segment_total_distance,
            )
        )

        segment_times_list.extend(segment_times)
        segment_distances_list.extend(segment_distances)

        # Print a message to show the current segment's data
        print(
            f"Segment {i+1}: Start {waypoints_list[i]} End {waypoints_list[i+1]} Time {segment_total_time/86400} days Distance {segment_total_distance/1000} kilometers"
        )

    # Add the last waypoint to the optimal mission path
    optimal_mission_path.append(waypoints_list[-1])

    for i, (lat, lon) in enumerate(optimal_mission_path):
        if i == 0:
            path_csv_data.append((lat, lon, 0, 0))
        else:
            path_csv_data.append(
                (lat, lon, segment_times_list[i - 1], segment_distances_list[i - 1])
            )

    # Print the total mission time (adjusted) and distance
    print(f"Total mission time (adjusted): {total_time/86400} days")
    print(f"Total mission distance: {total_distance/1000} kilometers")

    # Ensure output directory exists
    dir = f"products/{ds.time.dt.strftime('%Y_%m_%d').values}/data"
    os.makedirs(dir, exist_ok=True)

    # Define the path for the path CSV file
    path_csv_file_path = os.path.join(
        dir,
        f"{mission_name}_{fdate}_{model_name}_mission_path.csv",
    )
    # Open the CSV file and write the data to it
    with open(path_csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(path_csv_data)

    # Save path again but as a KML file
    kml_path = os.path.join(
        dir, f"{mission_name}_{fdate}_{model_name}_mission_path.kml"
    )
    kml = simplekml.Kml()
    optimal_path_formatted = [(lon, lat) for lat, lon in optimal_mission_path]
    kml.newlinestring(name="Path", coords=optimal_path_formatted)
    kml.save(kml_path)

    # Define the path for the CSV file
    csv_file_path = os.path.join(
        dir,
        f"{mission_name}_{fdate}_{model_name}_mission_statistics.csv",
    )
    # Open the CSV file and write the data to it
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)

    return optimal_mission_path


# TODO: Add a new algorithm for finding the optimal path.
# Might need to rework A* function to have helper functions be on the outside. We shall see.
