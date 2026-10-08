"""Download API for wiend."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import xarray as xr

from wiend.data_manager import (
    DEFAULT_GITHUB_REPOSITORY,
    DEFAULT_WEIBULL_RELEASE_TAG,
    EDH_DEFAULT_PATH,
    build_edh_url,
    load_edh_key,
)
from wiend.downscale import downscale
from wiend.validation import normalize_wind_da, parse_bbox


def _get_wind_da(
    *,
    bbox: tuple[float, float, float, float],
    start: str | None = None,
    end: str | None = None,
    edh_key: str | None = None,
    edh_path: str | None = None,
    height: str = "10m",
    u_var: str | None = None,
    v_var: str | None = None,
    bbox_order: str = "lonlat",
    time_dim: str = "time",
    chunks: dict[str, int] | None = None,
) -> xr.DataArray:
    key = load_edh_key(edh_key)
    path = edh_path or EDH_DEFAULT_PATH
    url = build_edh_url(path, key)

    ds_src = xr.open_dataset(
        url,
        engine="zarr",
        chunks=chunks or {},
        decode_coords="all",
        mask_and_scale=False,
    )

    if u_var is None:
        preferred_u = ["u10", "u10m", "u"] if height == "10m" else ["u100", "u100m", "u"]
        for name in preferred_u:
            if name in ds_src.data_vars:
                u_var = name
                break

    if v_var is None:
        preferred_v = ["v10", "v10m", "v"] if height == "10m" else ["v100", "v100m", "v"]
        for name in preferred_v:
            if name in ds_src.data_vars:
                v_var = name
                break

    if u_var is None or v_var is None:
        raise ValueError(f"Could not infer u/v variables. Available: {list(ds_src.data_vars)}")

    u_da = normalize_wind_da(ds_src[u_var], time_dim=time_dim)
    v_da = normalize_wind_da(ds_src[v_var], time_dim=time_dim)

    lon_min, lat_min, lon_max, lat_max = parse_bbox(bbox, bbox_order=bbox_order)
    u_da = u_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    v_da = v_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    if start is not None or end is not None:
        u_da = u_da.sel(time=slice(start, end))
        v_da = v_da.sel(time=slice(start, end))

    u_da, v_da = xr.align(u_da, v_da, join="inner")
    wind_da = (u_da**2 + v_da**2) ** 0.5
    return wind_da.rename("wind").astype("float32")


def download(
    *,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    out: str | Path | None = None,
    edh_key: str | None = None,
    edh_path: str | None = None,
    height: str = "10m",
    u_var: str | None = None,
    v_var: str | None = None,
    bbox_order: str = "lonlat",
    time_dim: str = "time",
    overwrite: bool = False,
    edh_time_chunk: int = 168,
    tile_lat: int = 50,
    tile_lon: int = 50,
    min_valid: int = 100,
    compute: bool = True,
    write_summary: bool = False,
) -> xr.DataArray:
    """Download ERA5 wind data and optionally write it to Zarr."""
    wind_da = _get_wind_da(
        bbox=bbox,
        start=start,
        end=end,
        edh_key=edh_key,
        edh_path=edh_path,
        height=height,
        u_var=u_var,
        v_var=v_var,
        bbox_order=bbox_order,
        time_dim=time_dim,
        chunks={"time": int(edh_time_chunk)},
    )

    if compute:
        wind_da = wind_da.compute()

    if out is None:
        return wind_da

    out_path = Path(out)
    if out_path.exists():
        if not overwrite:
            raise ValueError(f"Output {out_path} exists; set overwrite=True to replace.")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    ds_out = wind_da.to_dataset()
    ds_out = ds_out.assign_coords(
        lat=ds_out["lat"].astype("float32"), lon=ds_out["lon"].astype("float32")
    )
    ds_out.attrs.update(
        {
            "start": start,
            "end": end,
            "bbox": list(map(float, bbox)),
            "height": height,
            "edh_path": edh_path or EDH_DEFAULT_PATH,
            "time_chunk": int(edh_time_chunk),
            "min_valid": int(min_valid),
            "source": "era5_edh",
            "tile_lat": int(tile_lat),
            "tile_lon": int(tile_lon),
        }
    )

    chunks_time = min(int(edh_time_chunk), int(ds_out.sizes.get("time", edh_time_chunk)))
    chunks_lat = min(int(tile_lat), int(ds_out.sizes.get("lat", tile_lat)))
    chunks_lon = min(int(tile_lon), int(ds_out.sizes.get("lon", tile_lon)))
    encoding = {
        "wind": {"dtype": "float32", "chunks": (chunks_time, chunks_lat, chunks_lon)},
        "lat": {"dtype": "float32"},
        "lon": {"dtype": "float32"},
    }

    for var_name in ds_out.data_vars:
        ds_out[var_name].encoding = {}
    for coord_name in ds_out.coords:
        ds_out.coords[coord_name].encoding = {}
    ds_out.encoding = {}

    ds_out.to_zarr(str(out_path), mode="w", compute=True, encoding=encoding)

    if write_summary:
        summary = {
            "bbox": list(map(float, bbox)),
            "start": start,
            "end": end,
            "out": str(out_path),
            "tile_lat": int(tile_lat),
            "tile_lon": int(tile_lon),
            "min_valid": int(min_valid),
            "height": height,
            "written_vars": list(ds_out.data_vars.keys()),
        }
        summary_path = Path(str(out_path) + ".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return wind_da


def download_and_downscale(
    *,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    out: str | Path,
    mapping_method: str,
    params_baseline_path: str | Path | None = None,
    params_manipulator_path: str | Path | None = None,
    spacing: float | None = None,
    cache_dir: str | Path | None = None,
    github_repository: str = DEFAULT_GITHUB_REPOSITORY,
    release_tag: str = DEFAULT_WEIBULL_RELEASE_TAG,
    github_token: str | None = None,
    edh_key: str | None = None,
    edh_path: str | None = None,
    height: str = "10m",
    u_var: str | None = None,
    v_var: str | None = None,
    bbox_order: str = "lonlat",
    time_dim: str = "time",
    overwrite: bool = False,
    no_crop: bool = False,
    edh_time_chunk: int = 168,
) -> Path:
    """Download ERA5 wind and run downscaling in one call."""
    wind_da = download(
        bbox=bbox,
        start=start,
        end=end,
        out=None,
        edh_key=edh_key,
        edh_path=edh_path,
        height=height,
        u_var=u_var,
        v_var=v_var,
        bbox_order=bbox_order,
        time_dim=time_dim,
        edh_time_chunk=edh_time_chunk,
        compute=True,
    )
    return downscale(
        input_data=wind_da,
        out=out,
        bbox=bbox,
        mapping_method=mapping_method,
        params_baseline_path=params_baseline_path,
        params_manipulator_path=params_manipulator_path,
        spacing=spacing,
        cache_dir=cache_dir,
        github_repository=github_repository,
        release_tag=release_tag,
        github_token=github_token,
        bbox_order=bbox_order,
        no_crop=no_crop,
        overwrite=overwrite,
    )
