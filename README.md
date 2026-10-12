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

Requirements

- Python 3.11 or newer
- EDH access token via `EDH_TOKEN` or `EDH_KEY`
- Internet access for the first Weibull data download

Quick start

1. Install the package:
   ```bash
   pip install wiend
   ```
2. Set your EDH token:
   ```bash
   export EDH_TOKEN="your-token"
   ```
3. Run the combined download + downscale workflow:
   ```python
   import wiend as wd

   out_path = wd.download_and_downscale(
       bbox=[5, 45, 15, 55],
       start="2020-01-01",
       end="2020-12-31",
       out="wind_downscaled.zarr",
       mapping_method="qm",
       spacing=0.0025,
   )

   print(out_path)
   ```

Data

The package code is distributed through PyPI.

The Weibull parameter maps are distributed separately as GitHub Release assets and are downloaded automatically when required due to the file size of the corrected parameters (~950 MB).

GitHub Release tag: weibull-parameters-v1

Release assets:

- `weibull-parameters-baseline-v1.zarr.zip`
- `weibull-parameters-corrected-v1.zarr.zip`

Local cache location: ~/.wiend/

The library downloads the release assets, extracts the Zarr archives, and reuses the cached data on later runs.

Example
import wiend as wd

ds = wd.download(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-12-31",
)

out_path = wd.downscale(
    input_data=ds,
    out="wind_downscaled.zarr",
    bbox=[5, 45, 15, 55],
    mapping_method="qm",
    spacing=0.0025,
)

print(out_path)


Or:

import wiend as wd

out_path = wd.download_and_downscale(
    bbox=[5, 45, 15, 55],  # lon_min, lat_min, lon_max, lat_max
    start="2020-01-01",
    end="2020-12-31",
    out="wind_downscaled.zarr",
    mapping_method="qm",
    spacing=0.0025,
)

print(out_path)

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

## Testing

Run tests with:
bash
pixi run pytest tests/ -v
All tests and linting run automatically on GitHub via CI when you push.