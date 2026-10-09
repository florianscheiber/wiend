"""Tests for wiend package import and public API."""

from __future__ import annotations


def test_import_wiend():
    """Test that wiend can be imported and exposes required API."""
    import wiend as wd

    assert hasattr(wd, "download")
    assert hasattr(wd, "downscale")
    assert hasattr(wd, "download_and_downscale")
    assert hasattr(wd, "__version__")


def test_version():
    """Test that version is correctly set."""
    import wiend as wd

    assert wd.__version__ == "0.1.0"


def test_api_callability():
    """Test that public API functions are callable."""
    import wiend as wd

    assert callable(wd.download)
    assert callable(wd.downscale)
    assert callable(wd.download_and_downscale)
