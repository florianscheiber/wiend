"""Tests for wiend.data_manager module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from wiend.data_manager import (
    _github_api_release_url,
    _select_asset,
    _spacing_token,
    get_cache_dir,
    load_edh_key,
)


class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_default_cache_dir(self):
        """Test that default cache dir is ~/.wiend/."""
        cache = get_cache_dir()
        assert cache == (Path.home() / ".wiend").resolve()

    def test_custom_cache_dir(self, tmp_path):
        """Test that custom cache dir is used when provided."""
        cache = get_cache_dir(tmp_path)
        assert cache == tmp_path.resolve()

    def test_expanduser_cache_dir(self, tmp_path, monkeypatch):
        """Test that ~ is expanded in cache dir."""
        monkeypatch.setenv("HOME", str(tmp_path))
        cache = get_cache_dir("~/custom/.wiend")
        assert "custom" in str(cache)


class TestLoadEdhKey:
    """Tests for load_edh_key()."""

    def test_load_from_argument(self):
        """Test loading key from argument."""
        key = load_edh_key("test_key_123")
        assert key == "test_key_123"

    def test_load_from_edh_token_env(self, monkeypatch):
        """Test loading key from EDH_TOKEN env var."""
        monkeypatch.setenv("EDH_TOKEN", "env_token_456")
        monkeypatch.delenv("EDH_KEY", raising=False)
        key = load_edh_key()
        assert key == "env_token_456"

    def test_load_from_edh_key_env(self, monkeypatch):
        """Test loading key from EDH_KEY env var."""
        monkeypatch.delenv("EDH_TOKEN", raising=False)
        monkeypatch.setenv("EDH_KEY", "env_key_789")
        key = load_edh_key()
        assert key == "env_key_789"

    def test_no_key_available(self, monkeypatch):
        """Test error when no key is available."""
        monkeypatch.delenv("EDH_TOKEN", raising=False)
        monkeypatch.delenv("EDH_KEY", raising=False)
        with pytest.raises(ValueError, match="No EDH token found"):
            load_edh_key()


class TestSpacingToken:
    """Tests for _spacing_token()."""

    def test_spacing_token_none(self):
        """Test spacing token with None."""
        token = _spacing_token(None)
        assert token is None

    def test_spacing_token_float(self):
        """Test spacing token with float."""
        token = _spacing_token(0.0025)
        assert token == "spacing-0-0025"

    def test_spacing_token_simple(self):
        """Test spacing token with simple decimal."""
        token = _spacing_token(1.5)
        assert token == "spacing-1-5"


class TestGitHubApiReleaseUrl:
    """Tests for _github_api_release_url()."""

    def test_latest_release_url(self):
        """Test URL for latest release."""
        url = _github_api_release_url("florianscheiber/wiend", "latest")
        assert url == "https://api.github.com/repos/florianscheiber/wiend/releases/latest"

    def test_tagged_release_url(self):
        """Test URL for specific tagged release."""
        url = _github_api_release_url("florianscheiber/wiend", "v1.0.0")
        assert url == "https://api.github.com/repos/florianscheiber/wiend/releases/tags/v1.0.0"

    def test_repository_with_special_chars(self):
        """Test URL with special chars in repo name."""
        url = _github_api_release_url("user/repo-name", "latest")
        assert "repo-name" in url


class TestSelectAsset:
    """Tests for _select_asset()."""

    def test_select_single_matching_asset(self):
        """Test selecting a single matching asset."""
        assets = [
            {"name": "weibull_era5-edh_spacing-0-0025.zarr", "browser_download_url": "url1"},
            {"name": "weibull_corrected_spacing-0-0025.zarr", "browser_download_url": "url2"},
        ]
        asset = _select_asset(assets, role_key="era5-edh", spacing=0.0025)
        assert asset["name"] == "weibull_era5-edh_spacing-0-0025.zarr"

    def test_select_asset_ignores_spacing_when_none(self):
        """Test that spacing filter is skipped when None."""
        assets = [
            {"name": "weibull_era5-edh.zarr", "browser_download_url": "url1"},
        ]
        asset = _select_asset(assets, role_key="era5-edh", spacing=None)
        assert asset["name"] == "weibull_era5-edh.zarr"

    def test_no_matching_asset(self):
        """Test error when no asset matches."""
        assets = [{"name": "other_file.txt"}]
        with pytest.raises(RuntimeError, match="No release asset found"):
            _select_asset(assets, role_key="era5-edh", spacing=0.0025)

    def test_multiple_matching_assets(self):
        """Test error when multiple assets match."""
        assets = [
            {"name": "weibull_era5-edh_spacing-0-0025_v1.zarr", "browser_download_url": "url1"},
            {"name": "weibull_era5-edh_spacing-0-0025_v2.zarr", "browser_download_url": "url2"},
        ]
        with pytest.raises(RuntimeError, match="Multiple release assets matched"):
            _select_asset(assets, role_key="era5-edh", spacing=0.0025)


class TestResolveWeibullReferencePaths:
    """Tests for resolve_weibull_reference_paths()."""

    @patch("wiend.data_manager._github_release_payload")
    def test_resolve_with_mock_github_api(self, mock_payload, tmp_path):
        """Test resolving paths with mocked GitHub API."""
        mock_payload.return_value = {
            "assets": [
                {
                    "name": "weibull_era5-edh_spacing-0-0025.zip",
                    "browser_download_url": "https://github.com/...",
                },
                {
                    "name": "weibull_corrected_spacing-0-0025.zip",
                    "browser_download_url": "https://github.com/...",
                },
            ]
        }

        from wiend.data_manager import resolve_weibull_reference_paths

        with patch("wiend.data_manager._materialize_release_asset") as mock_materialize:
            mock_materialize.side_effect = [tmp_path / "baseline", tmp_path / "manipulator"]

            baseline, manipulator = resolve_weibull_reference_paths(
                spacing=0.0025,
                cache_dir=tmp_path,
            )

            assert baseline == tmp_path / "baseline"
            assert manipulator == tmp_path / "manipulator"
            mock_payload.assert_called_once()
            assert mock_materialize.call_count == 2
