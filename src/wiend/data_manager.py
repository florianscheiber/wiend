"""Data access and cache helpers for wiend."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import xarray as xr

from wiend.validation import (
    check_dims_and_coords,
    check_lat_lon_1d_ascending,
)

EDH_DEFAULT_PATH = "era5/reanalysis-era5-single-levels-v0.zarr"
DEFAULT_GITHUB_REPOSITORY = "florianscheiber/wiend"
DEFAULT_WEIBULL_RELEASE_TAG = "latest"


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


def _spacing_token(spacing: float | None) -> str | None:
    if spacing is None:
        return None
    return f"spacing-{str(spacing).replace('.', '-')}"


def _github_api_release_url(repository: str, release_tag: str) -> str:
    owner, repo = repository.split("/", 1)
    owner_q = urllib.parse.quote(owner, safe="")
    repo_q = urllib.parse.quote(repo, safe="")
    if release_tag == "latest":
        return f"https://api.github.com/repos/{owner_q}/{repo_q}/releases/latest"
    tag_q = urllib.parse.quote(release_tag, safe="")
    return f"https://api.github.com/repos/{owner_q}/{repo_q}/releases/tags/{tag_q}"


def _github_release_payload(
    repository: str,
    release_tag: str,
    github_token: str | None = None,
) -> dict[str, Any]:
    url = _github_api_release_url(repository, release_tag)
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to fetch GitHub release metadata from {url}: {exc}") from exc
    return json.loads(body)


def _select_asset(
    assets: list[dict[str, Any]],
    *,
    role_key: str,
    spacing: float | None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for asset in assets:
        name = str(asset.get("name", "")).lower()
        if role_key not in name:
            continue
        matches.append(asset)

    if not matches:
        raise RuntimeError(f"No release asset found for role={role_key!r}.")
    if len(matches) > 1:
        names = ", ".join(str(a.get("name", "<unknown>")) for a in matches)
        raise RuntimeError(f"Multiple release assets matched role={role_key!r}: {names}")
    return matches[0]


def _download_file(url: str, target: Path, github_token: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, target.open("wb") as file_obj:
            shutil.copyfileobj(response, file_obj)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download asset from {url}: {exc}") from exc


def _materialize_release_asset(
    asset: dict[str, Any],
    *,
    cache_dir: Path,
    github_token: str | None = None,
) -> Path:
    name = str(asset["name"])
    asset_dir = cache_dir / "assets"
    archive_path = asset_dir / name
    extract_dir = cache_dir / "datasets" / Path(name).stem

    if extract_dir.exists():
        return _resolve_dataset_path(extract_dir)

    if not archive_path.exists():
        download_url = str(asset["browser_download_url"])
        _download_file(download_url, archive_path, github_token=github_token)

    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        shutil.unpack_archive(str(archive_path), str(extract_dir))
        return _resolve_dataset_path(extract_dir)

    # Fallback: non-archive asset, expose directly.
    return archive_path


def _resolve_dataset_path(extract_dir: Path) -> Path:
    """
    Resolve the actual dataset root from an extracted release asset directory.
    """
    zarr_dirs = list(extract_dir.rglob("*.zarr"))
    if len(zarr_dirs) == 1:
        return zarr_dirs[0]
    if len(zarr_dirs) > 1:
        raise RuntimeError(
            f"Ambiguous extracted asset {extract_dir}: multiple .zarr directories found."
        )

    children = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1:
        return children[0]
    return extract_dir


def resolve_weibull_reference_paths(
    *,
    spacing: float | None,
    cache_dir: str | Path | None = None,
    github_repository: str = DEFAULT_GITHUB_REPOSITORY,
    release_tag: str = DEFAULT_WEIBULL_RELEASE_TAG,
    github_token: str | None = None,
) -> tuple[Path, Path]:
    """
    Resolve baseline/manipulator Weibull parameter paths from GitHub release assets.
    """
    cache_path = get_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    payload = _github_release_payload(
        github_repository,
        release_tag,
        github_token=github_token,
    )
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("GitHub release payload does not include an assets list.")

    baseline_asset = _select_asset(
        assets,
        role_key="era5-edh",
        spacing=spacing,
    )
    manipulator_asset = _select_asset(
        assets,
        role_key="corrected",
        spacing=spacing,
    )

    baseline_path = _materialize_release_asset(
        baseline_asset,
        cache_dir=cache_path,
        github_token=github_token,
    )
    manipulator_path = _materialize_release_asset(
        manipulator_asset,
        cache_dir=cache_path,
        github_token=github_token,
    )

    return baseline_path, manipulator_path
