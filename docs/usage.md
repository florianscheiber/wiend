# Usage guide

## Installation

Install the package from PyPI:

```bash
python -m pip install wiend
```

`wiend` requires Python 3.11 or newer. Runtime dependencies are installed
automatically by `pip`.

## Authentication

The download functions access ERA5 data through Earth Data Hub (EDH). Provide
an EDH token in one of the following ways.

### Environment variable

macOS/Linux:

```bash
export EDH_TOKEN="your-token"
```

Windows PowerShell:

```powershell
$env:EDH_TOKEN = "your-token"
```

Windows Command Prompt:

```cmd
set EDH_TOKEN=your-token
```

The library also accepts `EDH_KEY` as an environment variable. Environment
variables are recommended because they keep credentials out of source files.

### Function argument

For a one-off script, pass the token directly as `edh_key`:

```python
import wiend as wd

wind = wd.download(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-01-02",
    edh_key="your-token",
)
```

Do not commit a real token to a repository or share it in logs.

## Coordinate and time conventions

The default bounding-box order is:

```text
[lon_min, lat_min, lon_max, lat_max]
```

For example, `[5, 45, 15, 55]` means 5°E to 15°E and 45°N to 55°N.
The `bbox_order="latlon"` option can be used when the input is supplied as
`[lat_min, lon_min, lat_max, lon_max]`.

`start` and `end` are date/time strings passed to the EDH dataset selection.
Use an ISO-like format such as `"2020-01-01"` or
`"2020-01-01T00:00:00"`.

## `download()`

`download()` loads ERA5 wind speed data from EDH. It combines the configured
ERA5 u- and v-wind components into a `float32` wind-speed `xarray.DataArray`
with `time`, `lat`, and `lon` dimensions.

```python
import wiend as wd

wind = wd.download(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-01-02",
    edh_key="your-token",
)
```

The returned value is an `xarray.DataArray`. Set `out` to also write the
downloaded data to a Zarr store:

```python
wind = wd.download(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-01-02",
    out="wind_input.zarr",
    edh_key="your-token",
)
```

Important parameters:

- `bbox`: geographic extent; Europe-only validation applies.
- `start`, `end`: requested time range.
- `out`: optional output Zarr path.
- `edh_key`: optional token; otherwise `EDH_TOKEN` or `EDH_KEY` is read.
- `edh_path`: optional EDH dataset path.
- `height`: wind height, default `"10m"`.
- `u_var`, `v_var`: optional source variable names.
- `bbox_order`: `"lonlat"` by default, or `"latlon"`.
- `overwrite`: allow replacement of an existing `out` store.
- `edh_time_chunk`, `tile_lat`, `tile_lon`: chunking controls.
- `compute`: compute the returned DataArray before returning it.
- `write_summary`: write a JSON summary next to the output store.

When `out` is provided, the selected ERA5 data is rechunked and written to
Zarr in time/latitude/longitude blocks, without first materializing the full
period in memory. `compute` controls eager computation when no output store is
requested.

## `downscale()`

`downscale()` maps already downloaded wind data to the Weibull reference grid
and writes the result to a Zarr store. `input_data` can be either an
`xarray.DataArray` or a path to a Zarr store containing a `wind` variable.

```python
import wiend as wd

out_path = wd.downscale(
    input_data="wind_input.zarr",
    out="wind_downscaled.zarr",
    bbox=[5, 45, 15, 55],
    mapping_method="qm",
    spacing=0.0025,
)
print(out_path)
```

The Weibull reference paths are optional. If both paths are omitted, `wiend`
resolves the `baseline` and `corrected` assets from the configured GitHub
release and caches them below `~/.wiend/`. To use local reference stores
instead, provide both paths:

```python
out_path = wd.downscale(
    input_data="wind_input.zarr",
    out="wind_downscaled.zarr",
    bbox=[5, 45, 15, 55],
    mapping_method="qm",
    params_baseline_path="/path/to/baseline.zarr",
    params_manipulator_path="/path/to/corrected.zarr",
)
```

Important parameters:

- `input_data`: wind `DataArray` or input Zarr path.
- `out`: output Zarr path.
- `bbox`: geographic extent in the selected `bbox_order`.
- `mapping_method`: currently supported methods are `"qm"`, `"mc"`, and
  `"dummy"`; use `"qm"` for the Weibull quantile-mapping workflow.
- `params_baseline_path`, `params_manipulator_path`: optional local reference
  paths; if either is missing, both release assets are resolved automatically.
- `spacing`: output grid spacing in degrees. Defaults to `0.0025`; larger
  positive integer multiples (for example `0.01`) are aggregated after
  Weibull correction. `None` is treated as the default.
- `cache_dir`: override the default `~/.wiend/` cache.
- `release_tag`: select a specific GitHub release; the default is the latest
  release.
- `github_repository`: override the default `florianscheiber/wiend` repository.
- `github_token`: optional GitHub token for release metadata/assets.
- `no_crop`: disable cropping of the reference maps to `bbox`.
- `overwrite`: allow replacement of an existing output store.
- `chunks`, `storage_dtype`, `scale`, `interp_method`, `use_shape_scaling`,
  `min_k`, and `eps_q`: advanced output and mapping controls.

## `download_and_downscale()`

`download_and_downscale()` combines the two workflows. It writes the requested
ERA5 data to an intermediate Zarr in blocks and maps that store to the Weibull
reference grid. The intermediate store is named `<out>.era5.zarr` and is
deleted after successful mapping by default. Set `keep_download=True` to keep
it; failed workflows retain it automatically.

```python
import wiend as wd

out_path = wd.download_and_downscale(
    bbox=[5, 45, 15, 55],
    start="2020-01-01",
    end="2020-01-02",
    out="wind_downscaled.zarr",
    mapping_method="qm",
    spacing=0.0025,
    edh_key="your-token",
)
print(out_path)
```

This function accepts the download-related options from `download()` and the
reference-data/output options from `downscale()`. The `edh_key` argument is
passed to the EDH download step; GitHub release access is configured
separately with `github_token`, `github_repository`, and `release_tag`.

Important parameter:

- `keep_download`: defaults to `False`; delete the intermediate
  `<out>.era5.zarr` after successful mapping. Set it to `True` to retain the
  intermediate ERA5 Zarr. It is retained automatically when mapping fails.

## Weibull data and local cache

The reference maps are distributed separately from the Python package in the
GitHub release `weibull-parameters-v1`, which is the default source:

- `weibull-parameters-baseline-v1.zarr.zip`
- `weibull-parameters-corrected-v1.zarr.zip`

On first automatic use, the archives are downloaded, extracted, and cached
under `~/.wiend/`. Later runs reuse the cached data. The corrected asset is
large, so the first automatic run requires sufficient disk space and a stable
internet connection.

Instead of downloading the default release assets, local Weibull stores can be
provided with `params_baseline_path` and `params_manipulator_path`. Both local
stores must contain `k` and `a` on the native `0.0025` degree latitude and
longitude grid.

## Common errors

- **No EDH token found:** set `EDH_TOKEN`/`EDH_KEY` or pass `edh_key`.
- **No release asset found:** check the release tag, repository, and asset
  names; the resolver expects `baseline` and `corrected`.
- **Output already exists:** pass `overwrite=True` or choose another path.
- **Missing `dask`, `fsspec`, or `scipy`:** reinstall `wiend` with
  `python -m pip install --upgrade wiend`; these are declared runtime
  dependencies.
- **SSL/network errors:** check the network, proxy/certificate configuration,
  and EDH availability.

## Development and testing

For development from a source checkout:

```bash
pixi run pip install -e .
pixi run pytest tests/ -v
```

End users do not need to clone the repository; they should install from PyPI.
