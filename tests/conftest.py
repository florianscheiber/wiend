"""Pytest fixtures for wiend tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def sample_wind_da() -> xr.DataArray:
    """Create a minimal wind DataArray with (time, lat, lon) structure."""
    time = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
    lat = np.linspace(45, 55, 10, dtype="float32")
    lon = np.linspace(5, 15, 10, dtype="float32")
    wind_data = np.random.uniform(0, 20, (len(time), len(lat), len(lon))).astype("float32")

    da = xr.DataArray(
        wind_data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        name="wind",
    )
    return da


@pytest.fixture
def sample_weibull_params_da() -> tuple[xr.DataArray, xr.DataArray]:
    """Create minimal Weibull parameter DataArrays."""
    lat = np.linspace(45, 55, 5, dtype="float32")
    lon = np.linspace(5, 15, 5, dtype="float32")
    k_data = np.random.uniform(1.5, 2.5, (len(lat), len(lon))).astype("float32")
    a_data = np.random.uniform(5, 10, (len(lat), len(lon))).astype("float32")

    k = xr.DataArray(k_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="k")
    a = xr.DataArray(a_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="a")
    return k, a
