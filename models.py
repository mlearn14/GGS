# author: matthew learn (matt.learn@marine.rutgers.edu)

import numpy as np
import xarray as xr
import copernicusmarine as cm

from functions import *

# TODO: for dubugging, check if dates given are within the datasets!


class CMEMS:
    """
    Class for handling Copernicus Marine Environment Monitoring Service (CMEMS) data.

    Attributes:
    ----------
        - raw_data (xr.Dataset): Raw data from CMEMS.
        - subset_data (xr.Dataset): Subset of raw data.
        - z_interpolated_data (xr.Dataset): Interpolated data to 1 meter depth intervals.
        - da_data (xr.Dataset): Depth averaged data.
    """

    def __init__(self) -> None:
        """Initialize the CMEMS instance."""
        self.raw_data: xr.Dataset = None
        self.subset_data: xr.Dataset = None
        self.z_interpolated_data: xr.Dataset = None
        self.da_data: xr.Dataset = None

    def load(
        self, username: str = "maristizabalvar", password: str = "MariaCMEMS2018"
    ) -> None:
        """
        Loads and subsets Eastward and Northward current velocities from the CMEMS model. Saves data to self.raw_data attribute.

        Args:
        ----------
            - username (str, optional): CMEMS username.
            - password (str, optional): CMEMS password.

        Returns:
        ----------
            `None`
        """
        print("Loading CMEMS data...")
        starttime = print_starttime()
        ds_id = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"  # dataset id for CMEMS current model
        ds = cm.open_dataset(dataset_id=ds_id, username=username, password=password)

        # rename variables for consistency across all datasets
        ds = ds.rename({"longitude": "lon", "latitude": "lat", "uo": "u", "vo": "v"})

        ds.attrs["text_name"] = "CMEMS"
        ds.attrs["model_name"] = "CMEMS"

        self.raw_data = ds  # keeps the raw data just in case

        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    def subset(self, dates: tuple, extent: tuple, depth: int = 1100) -> None:
        """
        Subsets the CMEMS dataset to the specified date, lon, lat, and depth bounds. Saves data to self.data attribute.

        Args:
        ----------
            - dates (tuple): A tuple of (date_min, date_max) in datetime format.
            - extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees.
            - depth (int, optional): The maximum depth in meters. Defaults to 1100. It is set to 1100 because CMEMS data does not have a layer at 1000 meters, so for interpolation to work, it has to have the next deepest layer.

        Returns:
            `None`
        """
        text_name = self.raw_data.attrs["text_name"]

        # unpack the dates and extent tuples
        date_min, date_max = dates
        lat_min, lon_min, lat_max, lon_max = extent

        # subset the data using the xarray .sel selector
        self.subset_data = self.raw_data.sel(
            time=slice(date_min, date_max),
            depth=slice(0, depth),
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )

        self.subset_data = self.subset_data.chunk("auto")

        print(f"{text_name}: Subsetted data.\n")


class ESPC:
    """
    Class for handling Earth System Prediciton Capability (ESPC) data.

    Attributes:
    ----------
        - raw_data (xr.Dataset): Raw data from CMEMS.
        - subset_data (xr.Dataset): Subset of raw data.
        - z_interpolated_data (xr.Dataset): Interpolated data to 1 meter depth intervals.
        - da_data (xr.Dataset): Depth averaged data.
    """

    def __init__(self) -> None:
        """Initialize the ESPC instance."""
        self.raw_data: xr.Dataset = None
        self.subset_data: xr.Dataset = None
        self.z_interpolated_data: xr.Dataset = None
        self.da_data: xr.Dataset = None

    def load(self) -> None:
        """Loads Eastward and Northward current velocities from the ESPC model. Saves data to self.raw_data attribute."""
        print("Loading ESPC data...")
        starttime = print_starttime()

        url = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"
        ds = xr.open_dataset(url, drop_variables="tau")  # , chunks="auto"

        # rename variables for consistency across all datasets
        ds = ds.rename({"water_u": "u", "water_v": "v"})

        # convert lon to -180 to 180 and reindex. the sort by lon for consistent indexing
        ds = ds.assign_coords(lon=(ds.lon - 180) % 360 - 180)
        ds = ds.sortby("lon")

        ds.attrs["text_name"] = "ESPC"
        ds.attrs["model_name"] = "ESPC"

        self.raw_data = ds

        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    def subset(self, dates: tuple, extent: tuple, depth: int = 1000) -> None:
        """
        Subsets the ESPC dataset to the specified date, lon, lat, and depth bounds. Saves data to self.data attribute.

        Args:
        ----------
            - dates (tuple): A tuple of (date_min, date_max) in datetime format.
            - extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees.
            - depth (int, optional): The maximum depth in meters. Defaults to 1000.

        Returns:
        ----------
            - `None`
        """
        # unpack the dates and extent tuples
        text_name = self.raw_data.attrs["text_name"]

        date_min, date_max = dates
        lat_min, lon_min, lat_max, lon_max = extent

        # subset the data using the xarray .sel selector
        self.subset_data = self.raw_data.sel(
            time=slice(date_min, date_max),
            depth=slice(0, depth),
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )

        self.subset_data = self.subset_data.chunk("auto")

        print(f"{text_name}: Subsetted data.\n")


class RTOFS:
    """
    Class for handling Real-Time Ocean Forecast System (RTOFS) data.

    Attributes:
    ----------
        - raw_data (xr.Dataset): Raw data from CMEMS.
        - subset_data (xr.Dataset): Subset of raw data.
        - z_interpolated_data (xr.Dataset): Interpolated data to 1 meter depth intervals.
        - da_data (xr.Dataset): Depth averaged data.
    """

    def __init__(self) -> None:
        """Initialize the RTOFS instance."""
        self.raw_data: xr.Dataset = None
        self.subset_data: xr.Dataset = None
        self.z_interpolated_data: xr.Dataset = None
        self.da_data: xr.Dataset = None

    def load(self, source: str) -> None:
        """
        Loads Eastward and Northward current velocities from the RTOFS model. Saves data to self.raw_data attribute.

        Args:
        ----------
            - source (str): RTOFS model source. Valid args are: 'east', 'west', and 'parallel'.
                - 'east' - RTOFS (East Coast)
                - 'west' - RTOFS (West Coast)
                - 'parallel' - RTOFS (Experimental version running in parallel with the production version)

        Returns:
        ----------
            - `None`
        """
        print("Loading RTOFS data...")
        starttime = print_starttime()

        # Create a dictionary mapping source names to URLs
        # The URLs are Thredds Data Server URLs that point to the RTOFS model datasets
        url_dict = {
            "east": "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_scraped",
            "west": "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_west_scraped",
            "parallel": "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_parallel_scraped",
        }

        # Create a dictionary mapping source names to model names
        text_dict = {
            "east": "RTOFS (East Coast)",
            "west": "RTOFS (West Coast)",
            "parallel": "RTOFS (Parallel)",
        }

        # Create a dictionary mapping source names to model names formatted for filenames
        model_dict = {
            "east": "RTOFS-east",
            "west": "RTOFS-west",
            "parallel": "RTOFS-parallel",
        }

        # Check if the source is valid
        if source not in url_dict:
            raise ValueError(
                f"Invalid source: {source}. Must be one of: {list(url_dict.keys())}"
            )

        # Get the URL and model name from the dictionaries
        url = url_dict[source]
        text_name = text_dict[source]
        model_name = model_dict[source]

        # Open the dataset using xarray
        ds = xr.open_dataset(url)

        # Rename the variables to match the standard naming convention
        ds = ds.rename(
            {
                "Longitude": "lon",
                "Latitude": "lat",
                "MT": "time",
                "Depth": "depth",
                "X": "x",
                "Y": "y",
            }
        )

        # ds = ds.set_coords(["lon", "lat"])  # set lon and lat as coordinates

        # Add the model name as an attribute to the dataset
        ds.attrs["text_name"] = text_name
        print(f"Model source: {text_name}")

        # Add the filename as an attribute to the dataset
        ds.attrs["model_name"] = model_name

        # Store the dataset in the instance variable
        self.raw_data = ds

        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    def subset(self, dates: tuple, extent: tuple, depth: int = 1000) -> None:
        """
        Subsets the RTOFS dataset to the specified date, lon, lat, and depth bounds. Saves data to self.subset_data attribute.

        Args:
        ----------
            - dates (tuple): A tuple of (date_min, date_max) in datetime format.
            - extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees.
            - depth (int, optional): The maximum depth in meters. Defaults to 1000.

        Returns:
        ----------
            - `None`
        """
        # unpack the dates and extent tuples
        text_name = self.raw_data.attrs["text_name"]

        date_min, date_max = dates
        lat_min, lon_min, lat_max, lon_max = extent

        # subset the data using the xarray .sel selector
        self.subset_data = self.raw_data.sel(
            time=slice(date_min, date_max), depth=slice(0, depth)
        )

        # subset the data to the specified area. RTOFS is weird and uses a different indexing scheme
        # code adapted from Mike Smith

        # Get the grid lons and lats
        grid_lons = self.subset_data.lon.values[0, :]
        grid_lats = self.subset_data.lat.values[:, 0]

        # Find x, y indexes of the area we want to subset
        lons_ind = np.interp([lon_min, lon_max], grid_lons, self.subset_data.x.values)
        lats_ind = np.interp([lat_min, lat_max], grid_lats, self.subset_data.y.values)

        # Use np.floor on the first index and np.ceiling on  the second index
        # of each slice in order to widen the area of the extent slightly.
        extent = [
            np.floor(lons_ind[0]).astype(int),
            np.ceil(lons_ind[1]).astype(int),
            np.floor(lats_ind[0]).astype(int),
            np.ceil(lats_ind[1]).astype(int),
        ]

        # Use the xarray .isel selector on x/y
        # since we know the exact indexes we want to slice
        self.subset_data = self.subset_data.isel(
            x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3])
        )

        self.subset_data = self.subset_data.chunk("auto")

        print(f"{text_name}: Subsetted data.\n")
