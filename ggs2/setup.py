from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ggs2",
    version="1.2.0",
    description="Helper functions to process ocean current data for Slocum Glider mission planning. Can chart an optimal path between waypoints, and compare multiple models.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Matthew Learn",
    author_email="matthewalearn@gmail.com",
    install_requires=[
        "cartopy",
        "cmocean",
        "copernicusmarine",
        "cool_maps",
        "dask",
        "matplotlib",
        "numpy",
        "pandas",
        "python",
        "streamlit",
        "xarray",
        "xesmf",
    ],
)

print("Setup complete.")
