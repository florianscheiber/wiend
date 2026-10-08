"""Validation and normalization helpers for wiend."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


def parse_bbox(
    bbox: Sequence[float],
    *,
    bbox_order: str = "lonlat",
) -> tuple[float, float, float, float]:
    """Parse bbox into (lon_min, lat_min, lon_max, lat_max)."""
    if len(bbox) != 4:
        raise ValueError("bbox must be a sequence of four values.")

    lon_min, lat_min, lon_max, lat_max = map(float, bbox)

    if bbox_order == "lonlat":
        pass
    elif bbox_order == "latlon":
        lat_min, lat_max, lon_min, lon_max = lon_min, lat_min, lon_max, lat_max
    else:
        raise ValueError("bbox_order must be 'lonlat' or 'latlon'.")

    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError(f"Invalid bbox: {bbox}")

    return lon_min, lat_min, lon_max, lat_max


def check_dims_and_coords(
    da: xr.DataArray,
    required_dims: tuple[str, ...],
    required_coords: tuple[str, ...],
) -> None:
    """Raise when required dims or coords are missing."""
    missing_dims = [d for d in required_dims if d not in da.dims]
    if missing_dims:
        raise ValueError(f"Missing dims {missing_dims}; found dims={da.dims}")

    missing_coords = [c for c in required_coords if c not in da.coords]
    if missing_coords:
        raise ValueError(f"Missing coords {missing_coords}; found coords={list(da.coords)}")


def check_lat_lon_1d_ascending(lat: xr.DataArray, lon: xr.DataArray) -> None:
    """Validate 1D ascending latitude/longitude coordinates."""
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            f"lat/lon must be 1D "
            f"lat dims={lat.dims}, shape={lat.shape}"
            f"lon dims={lon.dims}, shape={lon.shape}"
        )

    latv = lat.to_numpy()
    lonv = lon.to_numpy()

    if not np.all(np.diff(latv) > 0) or not np.all(np.diff(lonv) > 0):
        raise ValueError(
            f"lat or lon is not strictly increasing. "
            f"First/last lat: {latv[0]} .. {latv[-1]}"
            f"First/last lon: {lonv[0]} .. {lonv[-1]}"
        )


def normalize_wind_da(da: xr.DataArray, *, time_dim: str = "time") -> xr.DataArray:
    """Normalize downloaded wind DataArray to (time, lat, lon) float32."""
    rename_map: dict[str, str] = {}
    for src, dst in (("latitude", "lat"), ("longitude", "lon"), ("y", "lat"), ("x", "lon")):
        if src in da.dims:
            rename_map[src] = dst
    if rename_map:
        da = da.rename(rename_map)

    if time_dim in da.dims:
        da = da.rename({time_dim: "time"})
    elif "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    elif "time" in da.dims:
        da = da.rename({"time": "time"})

    if "time" not in da.dims:
        raise ValueError(f"Could not find a time dimension in {da.dims}")
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected lat/lon dims, got dims={da.dims}")

    da = da.sortby("lat").sortby("lon").transpose("time", "lat", "lon")
    return da.astype("float32")
