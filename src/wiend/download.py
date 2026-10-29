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
from wiend.downscale import (
    DEFAULT_CHUNKS,
    DEFAULT_SCALE,
    _default_compressor,
    downscale,
    map_pointwise_to_zarr,
)
from wiend.pointwise import (
    get_pointwise_wind_da,
    load_pointwise_params,
    load_pointwise_wind_checkpoint,
    load_station_locations,
    write_pointwise_wind_checkpoint,
)
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

    if out is None:
        if compute:
            wind_da = wind_da.compute()
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
    ds_out = ds_out.chunk({"time": chunks_time, "lat": chunks_lat, "lon": chunks_lon})
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

    ds_out.to_zarr(
        str(out_path),
        mode="w",
        compute=True,
        encoding=encoding,
        consolidated=False,
    )

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
    bbox: tuple[float, float, float, float] | None = None,
    stations: str | Path | None = None,
    start: str,
    end: str,
    out: str | Path,
    mapping_method: str,
    params_baseline_path: str | Path | None = None,
    params_manipulator_path: str | Path | None = None,
    spacing: float | None = 0.0025,
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
    keep_download: bool = False,
) -> Path:
    """Download ERA5 wind and run downscaling in one call."""
    if stations is not None:
        if bbox is not None:
            raise ValueError("Pass either stations or bbox, not both.")
        return _download_and_downscale_pointwise(
            stations=stations,
            start=start,
            end=end,
            out=out,
            mapping_method=mapping_method,
            params_baseline_path=params_baseline_path,
            params_manipulator_path=params_manipulator_path,
            cache_dir=cache_dir,
            github_repository=github_repository,
            release_tag=release_tag,
            github_token=github_token,
            edh_key=edh_key,
            edh_path=edh_path,
            height=height,
            u_var=u_var,
            v_var=v_var,
            time_dim=time_dim,
            overwrite=overwrite,
            edh_time_chunk=edh_time_chunk,
            keep_download=keep_download,
        )
    if bbox is None:
        raise ValueError("bbox is required unless stations is provided.")

    out_path = Path(out)
    download_path = Path(f"{out_path}.era5.zarr")
    try:
        download(
            bbox=bbox,
            start=start,
            end=end,
            out=download_path,
            edh_key=edh_key,
            edh_path=edh_path,
            height=height,
            u_var=u_var,
            v_var=v_var,
            bbox_order=bbox_order,
            time_dim=time_dim,
            overwrite=True,
            edh_time_chunk=edh_time_chunk,
            compute=False,
        )
        result = downscale(
            input_data=download_path,
            out=out_path,
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
    except Exception:
        raise
    else:
        if not keep_download and download_path.exists():
            if download_path.is_dir():
                shutil.rmtree(download_path)
            else:
                download_path.unlink()
        return result


def _download_and_downscale_pointwise(
    *,
    stations: str | Path,
    start: str,
    end: str,
    out: str | Path,
    mapping_method: str,
    params_baseline_path: str | Path | None,
    params_manipulator_path: str | Path | None,
    cache_dir: str | Path | None,
    github_repository: str,
    release_tag: str,
    github_token: str | None,
    edh_key: str | None,
    edh_path: str | None,
    height: str,
    u_var: str | None,
    v_var: str | None,
    time_dim: str,
    overwrite: bool,
    edh_time_chunk: int,
    keep_download: bool,
) -> Path:
    """Run the experimental station-based download and downscale workflow."""
    locations = load_station_locations(stations)
    out_path = Path(out)
    era5_checkpoint_path = Path(f"{out_path}.pointwise.era5.zarr")
    mapping_checkpoint_path = Path(f"{out_path}.pointwise.mapping.zarr")
    if out_path.exists() and not overwrite:
        raise ValueError(f"Output {out_path} exists; set overwrite=True to replace.")
    if params_baseline_path is None or params_manipulator_path is None:
        from wiend.data_manager import resolve_weibull_reference_paths

        baseline_path, manipulator_path = resolve_weibull_reference_paths(
            spacing=None,
            cache_dir=cache_dir,
            github_repository=github_repository,
            release_tag=release_tag,
            github_token=github_token,
        )
    else:
        baseline_path = Path(params_baseline_path)
        manipulator_path = Path(params_manipulator_path)

    k_b, a_b, _, _ = load_pointwise_params(
        baseline_path,
        station_lat=locations.lat,
        station_lon=locations.lon,
    )
    k_m, a_m, _, _ = load_pointwise_params(
        manipulator_path,
        station_lat=locations.lat,
        station_lon=locations.lon,
    )
    if era5_checkpoint_path.exists():
        print(f"pointwise: loading ERA5 checkpoint {era5_checkpoint_path}")
        wind = load_pointwise_wind_checkpoint(era5_checkpoint_path)
    else:
        print(
            "pointwise: loading all selected ERA5 cells from the Earth Data Hub",
            flush=True,
        )
        wind = get_pointwise_wind_da(
            locations=locations,
            start=start,
            end=end,
            edh_key=edh_key,
            edh_path=edh_path,
            height=height,
            u_var=u_var,
            v_var=v_var,
            time_dim=time_dim,
            chunks={"time": int(edh_time_chunk)},
        )
        temporary_era5_path = Path(f"{era5_checkpoint_path}.tmp")
        if temporary_era5_path.exists():
            shutil.rmtree(temporary_era5_path)
        write_pointwise_wind_checkpoint(wind, temporary_era5_path)
        temporary_era5_path.replace(era5_checkpoint_path)
        print(f"pointwise: ERA5 checkpoint written to {era5_checkpoint_path}", flush=True)

    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    map_pointwise_to_zarr(
        mapping_method,
        wind,
        k_b,
        a_b,
        k_m,
        a_m,
        mapping_checkpoint_path,
        station=locations.station,
        lat=locations.lat,
        lon=locations.lon,
        lat_era5=wind["lat_era5"].values,
        lon_era5=wind["lon_era5"].values,
        chunks=DEFAULT_CHUNKS,
        compressor=_default_compressor(),
        storage_dtype="int16",
        scale=DEFAULT_SCALE,
        resume=mapping_checkpoint_path.exists(),
    )
    mapping_checkpoint_path.replace(out_path)
    if not keep_download and era5_checkpoint_path.exists():
        shutil.rmtree(era5_checkpoint_path)
    return out_path
