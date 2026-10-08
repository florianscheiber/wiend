wiend

Python library for downloading ERA5 wind speed time series from Earth Data Hub and applying Weibull-based downscaling and bias correction.

Features
Download ERA5 wind speed data from Earth Data Hub
Automatic spatial validation for Europe
Weibull-based downscaling
Weibull-based bias correction
Zarr output support
Xarray-native workflow
Installation
pip install wiend

Data

The package code is distributed through PyPI.

The Weibull parameter maps are distributed separately as GitHub Release assets and are downloaded automatically when required.

Local cache location:

~/.wiend/

Example
import wiend as wd

ds = wd.download(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-12-31",
)

ds_downscaled = wd.downscale(ds)

ds_downscaled.to_zarr("wind.zarr")


Or:

import wiend as wd

ds_downscaled = wd.download_and_downscale(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-12-31",
)

Design Decisions
Python >= 3.11
MIT License
Distribution via PyPI
Weibull maps distributed via GitHub Releases
Europe-only support
Outputs:
xarray.Dataset
Zarr
Development

The project uses:

Pixi
Ruff
pre-commit
pytest