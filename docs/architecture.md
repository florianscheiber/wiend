# Architecture

## Overview

`wiend` is a Python library for downloading ERA5 wind data, downscaling it, and applying bias correction using Weibull parameter maps. The package is designed for a simple user-facing API while keeping data access, validation, and caching logic separated into internal modules.

## Core decisions

- Package name: `wiend`
- License: MIT
- Distribution: PyPI
- Data distribution: GitHub Release Assets
- Local cache directory: `~/.wiend/`
- Supported outputs: `xarray.Dataset` and Zarr
- Geographic scope: Europe only
- Minimum Python version: 3.11

## Public API

The public API is intentionally minimal and is exposed at package level:

- `download()`
- `downscale()`
- `download_and_downscale()`

Users should not need to import internal modules directly for standard workflows.

## Data distribution model

Weibull parameter maps are distributed separately from the Python package. They are published as GitHub Release assets rather than bundled in the package wheel. This keeps the library lightweight while allowing large geospatial reference datasets to be versioned and downloaded on demand.

The current release asset naming convention is:

- `weibull-parameters-baseline-v1.zarr.zip`
- `weibull-parameters-corrected-v1.zarr.zip`

These files are associated with the GitHub Release tag `weibull-parameters-v1` and are cached locally under `~/.wiend/` when first used. The library downloads the release assets, unpacks the Zarr archives, and reuses the cached data on later runs.

## Cache strategy

Reference datasets are downloaded once and stored in a local cache directory under the user home:

- `~/.wiend/`

This allows repeat usage without re-downloading the same asset, while keeping the package usable in offline or partially connected environments.

## Scope and compatibility

- Geographic scope: Europe only
- Output format: `xarray.Dataset` and Zarr
- Supported Python: 3.11 or newer

## Packaging and distribution

The library is published to PyPI and installed as a standard Python package. The data assets remain separate, because they are large geospatial datasets that are better managed through release-based distribution than by bundling them into the package itself.
