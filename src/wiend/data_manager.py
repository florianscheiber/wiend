"""Data access and cache helpers for wiend."""

from __future__ import annotations

import os
from pathlib import Path

import xarray as xr

from wiend.validation import (
    check_dims_and_coords,
    check_lat_lon_1d_ascending,
)

EDH_DEFAULT_PATH = "era5/reanalysis-era5-single-levels-v0.zarr"


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the local cache directory, defaulting to ~/.wiend."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    return (Path.home() / ".wiend").resolve()


def load_edh_key(edh_key: str | None = None) -> str:
    """Load EDH token from argument or environment."""
    if edh_key is not None:
        return edh_key

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    key = os.getenv("EDH_TOKEN") or os.getenv("EDH_KEY")
    if not key:
        raise ValueError("No EDH token found. Pass edh_key or set EDH_TOKEN/EDH_KEY.")
    return key


def build_edh_url(edh_path: str, edh_key: str) -> str:
    """Build full EDH URL from logical path and token."""
    if edh_path.startswith("http://") or edh_path.startswith("https://"):
        return edh_path
    normalized_path = edh_path.lstrip("/")
    return f"https://edh:{edh_key}@api.earthdatahub.destine.eu/{normalized_path}"


def load_wind_from_zarr(path_input: str | Path) -> xr.DataArray:
    """Load wind data from Zarr and validate expected structure."""
    ds = xr.open_zarr(str(path_input))

    if "wind" not in ds:
        raise KeyError(f"Zarr missing variable 'wind'. Found: {list(ds.data_vars)}")

    wind = ds["wind"].astype("float32")
    required_dims = ("time", "lat", "lon")
    required_coords = ("time", "lat", "lon")
    check_dims_and_coords(wind, required_dims, required_coords)
    check_lat_lon_1d_ascending(wind["lat"], wind["lon"])
    return wind


def load_weibull_params_from_zarr(
    path_input: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    crop: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Load Weibull parameters k and a with optional bbox crop."""
    ds = xr.open_zarr(str(path_input))

    if crop:
        if bbox is None:
            raise ValueError("bbox must be provided when crop=True.")
        lat_min, lat_max, lon_min, lon_max = bbox
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    if "k" not in ds or "a" not in ds:
        raise KeyError(f"Params Zarr {path_input} missing 'k' or 'a'.")

    k = ds["k"]
    a = ds["a"]

    if k.sizes["lat"] == 0 or k.sizes["lon"] == 0:
        raise ValueError(f"Cropping produced an empty grid for {path_input} and bbox={bbox}.")

    required_dims = ("lat", "lon")
    required_coords = ("lat", "lon")
    check_dims_and_coords(k, required_dims, required_coords)
    check_dims_and_coords(a, required_dims, required_coords)
    check_lat_lon_1d_ascending(k["lat"], k["lon"])
    check_lat_lon_1d_ascending(a["lat"], a["lon"])
    return k, a
