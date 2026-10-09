"""Tests for wiend.downscale module."""

from __future__ import annotations

import inspect

from wiend.downscale import downscale


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
            "spacing": None,
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
