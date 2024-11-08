from tracemalloc import start
from matplotlib.pyplot import grid
from nbformat import convert
import numpy as np
import xarray as xr
import copernicusmarine as cm
from scipy.interpolate import interp1d, interp2d
import os

from functions import print_starttime, print_endtime, print_runtime


class CMEMS:
    """Class for handling Copernicus Marine Environment Monitoring Service (CMEMS) data."""

    def __init__(self) -> None:
        """Initialize the CMEMS instance."""
        self.raw_data: xr.Dataset = None
        self.data: xr.Dataset = None
        self.interpolated_data: xr.Dataset = None
        self.da_data: xr.Dataset = None

    def load(
        self, username: str = "maristizabalvar", password: str = "MariaCMEMS2018"
    ) -> None:
        """
        Loads and subsets Eastward and Northward current velocities from the CMEMS model. Saves data to self.raw_data attribute.

        Args:
            username (str): CMEMS username. Defaults to "maristizabalvar".
            password (str): CMEMS password. Defaults to "MariaCMEMS2018".

        Returns:
            None
        """
        print("Loading CMEMS data...")
        starttime = print_starttime()
        ds_id = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"  # dataset id for CMEMS current model
        ds = cm.open_dataset(dataset_id=ds_id, username=username, password=password)
        ds = ds.rename(
            {"longitude": "lon", "latitude": "lat", "uo": "u", "vo": "v"}
        )  # rename variables for consistency across all datasets

        self.raw_data = ds  # keeps the raw data just in case

        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    def subset(
        self,
        dates: tuple,
        extent: tuple,
        depth: int = 1000,
    ) -> None:
        """
        Subsets the CMEMS dataset to the specified date, lon, lat, and depth bounds. Saves data to self.data attribute.

        Args:
            dates (tuple): A tuple of (date_min, date_max) in datetime format.
            extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees.
            depth (float): The maximum depth in meters. Defaults to 1000.

        Returns:
            None
        """
        # unpack the dates and extent tuples
        date_min, date_max = dates
        lat_min, lon_min, lat_max, lon_max = extent

        # subset the data using the xarray .sel selector
        self.data = self.raw_data.sel(
            time=slice(date_min, date_max),
            depth=slice(0, depth),
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )

        self.data = self.data.chunk("auto")

        print("Subsetted CMEMS data.\n")


class ESPC:
    """Class for handling Earth System Prediciton Capability (ESPC) data."""

    def __init__(self) -> None:
        """Initialize the ESPC instance."""
        self.raw_data: xr.Dataset = None
        self.data: xr.Dataset = None
        self.interpolated_data: xr.Dataset = None
        self.da_data: xr.Dataset = None

    def load(self) -> None:
        """
        Loads Eastward and Northward current velocities from the ESPC model. Saves data to self.raw_data attribute.

        Returns:
            None
        """
        print("Loading ESPC data...")
        starttime = print_starttime()

        url = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"
        ds = xr.open_dataset(url, drop_variables="tau")  # , chunks="auto"

        # ds["lon"] = ds["lon"] - 180  # shift longitude from 0-360 to -180-180

        ds.attrs["model"] = "ESPC"
        ds = ds.rename(
            {"water_u": "u", "water_v": "v"}
        )  # rename variables for consistency across all datasets

        self.raw_data = ds

        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    def convert_coords(self, extent: tuple) -> tuple:
        """
        Converts extent to a 0 - 360 degree longitude system.

        Args:
            extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees.

        Returns:
            new_extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees, with the latitude converted from -180 - 180 to 0 - 360.
        """
        lat_min, lon_min, lat_max, lon_max = extent

        lon_min = lon_min + 180
        lon_max = lon_max + 180

        new_extent = (lat_min, lon_min, lat_max, lon_max)

        return new_extent

    def subset(
        self,
        dates: tuple,
        extent: tuple,
        depth: float = 1000,
    ) -> None:
        """
        Subsets the ESPC dataset to the specified date, lon, lat, and depth bounds. Saves data to self.data attribute.

        Args:
            dates (tuple): A tuple of (date_min, date_max) in datetime format.
            extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees.
            depth (float): The maximum depth in meters. Defaults to 1000.

        Returns:
            None
        """
        # unpack the dates and extent tuples
        date_min, date_max = dates
        extent = self.convert_coords(extent)
        print(extent)
        lat_min, lon_min, lat_max, lon_max = extent
        print(lat_min, lon_min, lat_max, lon_max)

        # subset the data using the xarray .sel selector
        self.data = self.raw_data.sel(
            time=slice(date_min, date_max),
            depth=slice(0, depth),
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )

        # self.data = self.data.chunk("auto")

        print("Subsetted ESPC data.\n")


class RTOFS:
    """Class for handling Real-Time Ocean Forecast System (RTOFS) data."""

    def __init__(self) -> None:
        """Initialize the RTOFS instance."""
        self.raw_data: xr.Dataset = None
        self.data: xr.Dataset = None
        self.interpolated_data: xr.Dataset = None
        self.da_data: xr.Dataset = None

    def load(self, source: str) -> None:
        """
        Loads Eastward and Northward current velocities from the RTOFS model. Saves data to self.raw_data attribute.

        Args:
            source (str): RTOFS model source. Valid args are: 'east', 'west', and 'parallel'.
                - 'east' - RTOFS (East Coast)
                - 'west' - RTOFS (West Coast)
                - 'parallel' - RTOFS (Experimental version running in parallel with the production version)
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
        model_dict = {
            "east": "RTOFS (East Coast)",
            "west": "RTOFS (West Coast)",
            "parallel": "RTOFS (Parallel)",
        }

        # Check if the source is valid
        if source not in url_dict:
            raise ValueError(
                f"Invalid source: {source}. Must be one of: {list(url_dict.keys())}"
            )

        # Get the URL and model name from the dictionaries
        url = url_dict[source]
        model = model_dict[source]

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
        ds.attrs["model"] = model

        # Store the dataset in the instance variable
        self.raw_data = ds

        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)
        print()

    def subset(self, dates: tuple, extent: tuple, depth: float = 1000) -> None:
        """
        Subsets the RTOFS dataset to the specified date, lon, lat, and depth bounds. Saves data to self.data attribute.

        Args:
            dates (tuple): A tuple of (date_min, date_max) in datetime format.
            extent (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) in decimel degrees.
            depth (float): The maximum depth in meters. Defaults to 1000.

        Returns:
            None
        """
        # unpack the dates and extent tuples
        date_min, date_max = dates
        lat_min, lon_min, lat_max, lon_max = extent

        # subset the data using the xarray .sel selector
        self.raw_data = self.raw_data.sel(
            time=slice(date_min, date_max), depth=slice(0, depth)
        )

        # subset the data to the specified area. RTOFS is weird and uses a different indexing scheme
        # code adapted from Mike Smith

        # Get the grid lons and lats
        grid_lons = self.raw_data.lon.values[0, :]
        grid_lats = self.raw_data.lat.values[:, 0]

        # Find x, y indexes of the area we want to subset
        lons_ind = np.interp([lon_min, lon_max], grid_lons, self.raw_data.x.values)
        lats_ind = np.interp([lat_min, lat_max], grid_lats, self.raw_data.y.values)

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
        self.data = self.raw_data.isel(
            x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3])
        )

        self.data = self.data.chunk("auto")

        print("Subsetted RTOFS data.\n")
