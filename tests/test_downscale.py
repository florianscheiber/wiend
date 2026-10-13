"""Tests for wiend.downscale module."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

from wiend.downscale import (
    _coarsen_axis,
    _default_compressor,
    _normalize_spacing,
    downscale,
    map_weibull_field_to_zarr,
)


class TestDownscaleSignature:
    """Tests for downscale() function signature."""

    def test_downscale_has_required_params(self):
        """Test that downscale() has required parameters."""
        sig = inspect.signature(downscale)
        required_params = ["input_data", "out", "bbox", "mapping_method"]
        for param in required_params:
            assert param in sig.parameters, f"Missing required parameter: {param}"

    def test_downscale_has_optional_params(self):
        """Test that downscale() has optional asset-resolver parameters."""
        sig = inspect.signature(downscale)
        optional_params = [
            "params_baseline_path",
            "params_manipulator_path",
            "spacing",
            "cache_dir",
            "github_repository",
            "release_tag",
            "github_token",
        ]
        for param in optional_params:
            assert param in sig.parameters, f"Missing optional parameter: {param}"

    def test_downscale_params_have_defaults(self):
        """Test that optional params have defaults (not required)."""
        sig = inspect.signature(downscale)
        params_with_defaults = {
            "params_baseline_path": None,
            "params_manipulator_path": None,
            "spacing": 0.0025,
            "cache_dir": None,
            "github_token": None,
        }
        for param_name, expected_default in params_with_defaults.items():
            param = sig.parameters[param_name]
            assert param.default == expected_default, (
                f"{param_name} should default to {expected_default}"
            )


class TestDownscaleCallability:
    """Tests for downscale() callability."""

    def test_downscale_is_callable(self):
        """Test that downscale is callable."""
        assert callable(downscale)

    def test_downscale_has_docstring(self):
        """Test that downscale has a docstring."""
        assert downscale.__doc__ is not None
        assert len(downscale.__doc__) > 0


class TestSpacing:
    """Tests for output spacing validation and native-grid grouping."""

    def test_default_and_none_use_native_spacing(self):
        assert _normalize_spacing(None) == (0.0025, 1)
        assert _normalize_spacing(0.0025) == (0.0025, 1)

    def test_coarser_spacing_returns_complete_groups_and_centres(self):
        values = np.arange(8, dtype=np.float64) * 0.0025
        groups, centres = _coarsen_axis(values, 4)
        np.testing.assert_array_equal(groups, [[0, 1, 2, 3], [4, 5, 6, 7]])
        np.testing.assert_allclose(centres, [0.00375, 0.01375])

    def test_mapping_aggregates_corrected_values(self, tmp_path):
        lat = np.arange(8, dtype=np.float32) * 0.0025
        lon = np.arange(8, dtype=np.float32) * 0.0025
        wind = xr.DataArray(
            np.arange(64, dtype=np.float32).reshape(1, 8, 8),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": lat, "lon": lon},
        )
        params = xr.DataArray(
            np.ones((8, 8), dtype=np.float32),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )

        map_weibull_field_to_zarr(
            "dummy",
            wind,
            params,
            params,
            params,
            params,
            tmp_path / "out.zarr",
            None,
            None,
            chunks={"time": 1, "lat": 2, "lon": 8},
            compressor=_default_compressor(),
            storage_dtype="float32",
            spacing=0.01,
        )
        result = xr.open_zarr(tmp_path / "out.zarr")["wind"]
        assert result.shape == (1, 2, 2)
        np.testing.assert_allclose(result.values, [[[13.5, 17.5], [45.5, 49.5]]])

    @pytest.mark.parametrize("spacing", [0, -0.0025, 0.003, float("nan")])
    def test_invalid_spacing_is_rejected(self, spacing):
        with pytest.raises(ValueError, match="spacing"):
            _normalize_spacing(spacing)
