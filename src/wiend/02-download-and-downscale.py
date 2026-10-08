#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Sequence

import truststore
import xarray as xr
import zarr
from dotenv import load_dotenv

from wiend.utils import load_weibull_params_from_zarr, map_weibull_field_to_zarr

truststore.inject_into_ssl()

# CLI usage:
# --edh-key KEY                            EarthDataHub token (or set EDH_TOKEN/EDH_KEY)
# --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX   Bounding box (lon_min lat_min lon_max lat_max)
# --start YYYY-MM-DD                       Start date (inclusive)
# --end YYYY-MM-DD                         End date (inclusive)
# --mapping_method STR                      Mapping method: qm, dummy, mc, sh2, sh2-simple
# --out PATH                               Output path for mapped wind Zarr
# --spacing FLOAT                          Grid spacing in degrees (required)
# Optional flags: --save-era5 PATH (save intermediate ERA5 Zarr), --no-crop,
#                 --edh-path, --u-var, --v-var,
#                 --bbox-order, --edh-time-chunk, --tile-lat, --tile-lon,
#                 --min-valid, --overwrite

EDH_DEFAULT_PATH = "era5/reanalysis-era5-single-levels-v0.zarr"


def _load_edh_key(edh_key: str | None = None) -> str:
    if edh_key is not None:
        return edh_key
    load_dotenv()
    key = os.getenv("EDH_TOKEN") or os.getenv("EDH_KEY")
    if not key:
        raise ValueError("No EDH token found. Pass --edh-key or set EDH_TOKEN/EDH_KEY.")
    return key


def _build_edh_url(edh_path: str, edh_key: str) -> str:
    if edh_path.startswith("http://") or edh_path.startswith("https://"):
        return edh_path
    if edh_path.startswith("/"):
        edh_path = edh_path.lstrip("/")
    return f"https://edh:{edh_key}@api.earthdatahub.destine.eu/{edh_path}"


def _parse_bbox(
    bbox: Sequence[float], *, bbox_order: str = "lonlat"
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must be a sequence of four values.")
    lon_min, lat_min, lon_max, lat_max = map(float, bbox)

    if bbox_order == "lonlat":
        lon_min, lat_min, lon_max, lat_max = lon_min, lat_min, lon_max, lat_max
    elif bbox_order == "latlon":
        lat_min, lat_max, lon_min, lon_max = lon_min, lat_min, lon_max, lat_max
    else:
        raise ValueError("bbox_order must be 'lonlat' or 'latlon'.")

    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError(f"Invalid bbox: {bbox}")
    return lon_min, lat_min, lon_max, lat_max


def _normalize_wind_da(da: xr.DataArray, *, time_dim: str = "time") -> xr.DataArray:
    rename_map = {}
    for src, dst in [("latitude", "lat"), ("longitude", "lon"), ("y", "lat"), ("x", "lon")]:
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


def get_wind_da(
    *,
    bbox: Sequence[float],
    start: str | None = None,
    end: str | None = None,
    edh_key: str | None = None,
    edh_path: str | None = None,
    u_var: str | None = None,
    v_var: str | None = None,
    bbox_order: str = "lonlat",
    time_dim: str = "time",
    chunks: dict | None = None,
) -> xr.DataArray:
    key = _load_edh_key(edh_key)

    if edh_path is None:
        edh_path = EDH_DEFAULT_PATH

    url = _build_edh_url(edh_path, key)

    ds_src = xr.open_dataset(
        url,
        engine="zarr",
        chunks=chunks or {},
        decode_coords="all",
        mask_and_scale=False,
    )

    if u_var is None:
        preferred_u = ["u10", "u10m", "u"]
        for name in preferred_u:
            if name in ds_src.data_vars:
                u_var = name
                break

    if v_var is None:
        preferred_v = ["v10", "v10m", "v"]
        for name in preferred_v:
            if name in ds_src.data_vars:
                v_var = name
                break

    if u_var is None or v_var is None:
        raise ValueError(f"Could not infer u/v variables. Available: {list(ds_src.data_vars)}")

    u_da = _normalize_wind_da(ds_src[u_var], time_dim=time_dim)
    v_da = _normalize_wind_da(ds_src[v_var], time_dim=time_dim)

    lon_min, lat_min, lon_max, lat_max = _parse_bbox(bbox, bbox_order=bbox_order)
    u_da = u_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    v_da = v_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    if start is not None or end is not None:
        u_da = u_da.sel(time=slice(start, end))
        v_da = v_da.sel(time=slice(start, end))

    u_da, v_da = xr.align(u_da, v_da, join="inner")
    wind_da = (u_da**2 + v_da**2) ** 0.5
    wind_da = wind_da.rename("wind").astype("float32")
    return wind_da


# --------------------- script02: download + map (one-step) ---------------------
STORAGE_DTYPE = "int16"
SCALE = 0.1
CHUNKS = {"time": 168, "lat": 100, "lon": 400}

# COMPRESSOR: same settings as script01b (Blosc zstd + bitshuffle, clevel=5).
# Used when writing the mapped wind Zarr via map_weibull_field_to_zarr.
COMPRESSOR = zarr.codecs.BloscCodec(
    cname="zstd",
    clevel=5,
    shuffle=zarr.codecs.BloscShuffle.bitshuffle,
)

# Hardcoded weibull-param paths (unchanged per your request)
PATH_PARAMS_BASELINE_ZARR = (
    "/Users/florianscheiber/PycharmProjects/downscaling_wind/data/data-after-task21/"
    "weibull-parameters-era5-edh/weibull-parameters_era5-edh-21/"
    "a-k_era5-edh_EUR_10m_2001-2020_spacing-0-0025.zarr"
)

PATH_PARAMS_MANIPULATOR_ZARR = (
    "/Users/florianscheiber/PycharmProjects/downscaling_wind/data/data-after-task21/"
    "weibull-parameters-corrected/"
    "weibull-parameters_EUR-isd_2001-2020-h_reg16-12-only-onshore_mle_train70p-"
    "mincount-87660_10m_spacing-0-0025.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box as lon_min lat_min lon_max lat_max",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--mapping_method", required=True, help="Mapping method: qm, dummy, mc, sh2, sh2-simple"
    )
    parser.add_argument("--out", required=True, help="Output path for mapped zarr")
    parser.add_argument(
        "--spacing", type=float, required=True, help="Grid spacing in degrees (kept)"
    )
    parser.add_argument(
        "--no-crop", action="store_true", help="Disable cropping of Weibull parameter Zarrs to bbox"
    )
    parser.add_argument("--edh-key")
    parser.add_argument("--edh-path", default=EDH_DEFAULT_PATH)
    parser.add_argument("--u-var")
    parser.add_argument("--v-var")
    parser.add_argument("--bbox-order", default="lonlat")
    parser.add_argument("--edh-time-chunk", type=int, default=168)
    parser.add_argument("--tile-lat", type=int, default=50)
    parser.add_argument("--tile-lon", type=int, default=50)
    parser.add_argument("--min-valid", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-era5",
        default=None,
        help="Optional path to save ERA5 wind zarr before mapping (if omitted, not saved)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lon_min, lat_min, lon_max, lat_max = map(float, args.bbox)
    bbox_for_download = (
        lon_min,
        lat_min,
        lon_max,
        lat_max,
    )  # get_wind_da expects lonlat by default
    bbox_for_params = (
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )  # load_weibull expects latmin, latmax, lonmin, lonmax
    mapping_method = args.mapping_method

    path_save_mapping = args.out

    print("Downloading ERA5 wind from EDH...")
    wind_da = get_wind_da(
        bbox=bbox_for_download,
        start=args.start,
        end=args.end,
        edh_key=args.edh_key,
        edh_path=args.edh_path,
        u_var=args.u_var,
        v_var=args.v_var,
        bbox_order=args.bbox_order,
        chunks={"time": int(args.edh_time_chunk)},
    )

    print("Materializing ERA5 wind into memory...")
    wind_da = wind_da.compute()
    print(wind_da)

    # optional: save ERA5 intermediate (disabled by default)
    if args.save_era5:
        out_era5 = Path(args.save_era5)
        if out_era5.exists():
            if args.overwrite:
                if out_era5.is_dir():
                    shutil.rmtree(out_era5)
                else:
                    out_era5.unlink()
            else:
                raise ValueError(f"ERA5 output {out_era5} exists; pass --overwrite to replace.")
        print("Writing ERA5 wind Zarr:", out_era5)
        ds_tmp = wind_da.to_dataset()
        # Clear all legacy encodings (including coords) to avoid numcodecs -> zarr v3 mismatch
        for v in ds_tmp.data_vars:
            ds_tmp[v].encoding = {}
        for c in ds_tmp.coords:
            ds_tmp.coords[c].encoding = {}
        ds_tmp.encoding = {}
        ds_tmp.to_zarr(str(out_era5), mode="w")

    # load weibull param zarrs (hardcoded paths)
    print("Loading Weibull parameters (baseline/manipulator) — using hardcoded paths")
    k_b, a_b = load_weibull_params_from_zarr(
        PATH_PARAMS_BASELINE_ZARR, bbox=bbox_for_params, crop=not args.no_crop
    )
    k_m, a_m = load_weibull_params_from_zarr(
        PATH_PARAMS_MANIPULATOR_ZARR, bbox=bbox_for_params, crop=not args.no_crop
    )

    # mapping
    print("Mapping (bias-correction / downscaling) — method=", mapping_method)
    map_weibull_field_to_zarr(
        mapping_method,
        wind_da,
        k_b,
        a_b,
        k_m,
        a_m,
        path_save_mapping,
        None,
        None,
        chunks=CHUNKS,
        compressor=COMPRESSOR,
        scale=SCALE,
        storage_dtype=STORAGE_DTYPE,
    )

    # check results
    root = zarr.open(path_save_mapping, mode="r")
    arr = root["wind"]
    print("Mapped wind written to:", path_save_mapping)
    print("dtype:", arr.dtype)
    print("chunks:", arr.chunks)
    print("compressors:", arr.compressors)
    print("filters:", getattr(arr, "filters", None))


if __name__ == "__main__":
    main()
