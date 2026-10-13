# wiend

Python library for downloading ERA5 wind speed time series from Earth Data
Hub and applying Weibull-based downscaling and bias correction.

## Features

- Download ERA5 wind data from Earth Data Hub
- Automatic spatial validation for Europe
- Weibull-based downscaling and bias correction
- Zarr output support
- Xarray-native workflow

## Installation

`wiend` requires Python 3.11 or newer. Install it from PyPI:

```bash
python -m pip install wiend
```

The runtime dependencies are installed automatically by `pip`.

## Quick start

Set an Earth Data Hub token before running a download. macOS/Linux:

```bash
export EDH_TOKEN="your-token"
```

Windows users should use the equivalent PowerShell or Command Prompt
environment-variable command.

Then run the combined download and downscale workflow:

```python
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
```

The token can also be passed directly to a function with `edh_key="your-token"`.
Using an environment variable is recommended so credentials do not appear in
source code or version control.

## Public API

The package exposes three main functions:

- `download()` downloads ERA5 wind data and returns an
  `xarray.DataArray`. It can optionally write the input data to Zarr.
- `downscale()` maps an existing wind `DataArray` or wind Zarr store to the
  Weibull reference grid and writes a Zarr output.
- `download_and_downscale()` combines both operations.

`downscale()` and `download_and_downscale()` accept a `spacing` argument. It
defaults to `0.0025` degrees, and larger positive integer multiples such as
`0.01` degrees are aggregated after Weibull correction to reduce output size.

For complete parameter documentation, examples, cache details, and
troubleshooting, see the
[full usage guide](https://github.com/florianscheiber/wiend/blob/main/docs/usage.md).

For example, the two separate operations can be used like this:

```python
import wiend as wd

wind = wd.download(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-01-02",
)

out_path = wd.downscale(
    input_data=wind,
    out="wind_downscaled.zarr",
    bbox=[5, 45, 15, 55],
    mapping_method="qm",
    spacing=0.0025,
)
```

## Weibull parameter data

The Weibull parameter maps are distributed separately as GitHub Release assets
because the corrected dataset is large (approximately 950 MB). They are
downloaded automatically when needed, extracted, and cached locally.

- Release tag: `weibull-parameters-v1`
- `weibull-parameters-baseline-v1.zarr.zip`
- `weibull-parameters-corrected-v1.zarr.zip`
- Local cache: `~/.wiend/`

An internet connection and sufficient disk space are required for the first
automatic data download.

The GitHub Release assets are the default Weibull parameter source. You can
instead provide local Baseline and Corrected Zarr stores through
`params_baseline_path` and `params_manipulator_path`; both local stores must
use the native `0.0025` degree grid.

## Development

Install the package from a source checkout in editable mode:

```bash
pixi run pip install -e .
```

Run the test suite:

```bash
pixi run pytest tests/ -v
```

Ruff, pytest, and package builds also run automatically in GitHub Actions on
pushes and pull requests.

The temporary-clone smoke test is a development check only. End users should
install `wiend` from PyPI and do not need to clone the repository.

## Design decisions

- License: MIT
- Distribution: PyPI
- Geographic scope: Europe only
- Supported outputs: `xarray.Dataset` and Zarr
- Minimum Python version: 3.11
