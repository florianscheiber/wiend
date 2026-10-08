#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import truststore
import xarray as xr
from dotenv import load_dotenv

truststore.inject_into_ssl()

# CLI usage:
# --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX   Bounding box (lon_min lat_min lon_max lat_max)
# --start YYYY-MM-DD                       Start date (inclusive)
# --end YYYY-MM-DD                         End date (inclusive)
# --out PATH                               Output path for ERA5 wind Zarr
# --edh-key KEY                            EarthDataHub token (or set EDH_TOKEN/EDH_KEY)
# Optional flags: --height, --u-var, --v-var, --bbox-order, --edh-time-chunk
#               --tile-lat, --tile-lon, --min-valid, --overwrite

# --- EDH helpers (lightweight, adapted) ---
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

    # ensure 1D lat/lon & canonical ordering
    da = da.sortby("lat").sortby("lon").transpose("time", "lat", "lon")
    da = da.astype("float32")
    # attach lightweight provenance if available
    prov = {}
    try:
        prov["coords_lon_convention"] = "signed(-180..180)"
    except Exception:
        pass
    if prov:
        da.attrs.update(prov)
    return da


def get_wind_da(
    *,
    bbox: Sequence[float],
    start: str | None = None,
    end: str | None = None,
    edh_key: str | None = None,
    edh_path: str | None = None,
    height: str = "10m",
    u_var: str | None = None,
    v_var: str | None = None,
    bbox_order: str = "lonlat",
    time_dim: str = "time",
    chunks: Mapping[str, int] | None = None,
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


# --- Main script: download + save wind only (no a/k) ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, help="Output path (zarr).")
    parser.add_argument("--edh-key")
    parser.add_argument("--edh-path", default=EDH_DEFAULT_PATH)
    parser.add_argument("--height", default="10m")
    parser.add_argument("--u-var")
    parser.add_argument("--v-var")
    parser.add_argument("--bbox-order", default="lonlat")
    parser.add_argument("--edh-time-chunk", type=int, default=168)
    parser.add_argument("--tile-lat", type=int, default=50)
    parser.add_argument("--tile-lon", type=int, default=50)
    parser.add_argument("--min-valid", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    print("Loading ERA5 wind from EDH...")
    wind_da = get_wind_da(
        bbox=args.bbox,
        start=args.start,
        end=args.end,
        edh_key=args.edh_key,
        edh_path=args.edh_path,
        height=args.height,
        u_var=args.u_var,
        v_var=args.v_var,
        bbox_order=args.bbox_order,
        chunks={"time": int(args.edh_time_chunk)},
    )

    # basic validation
    if tuple(wind_da.dims) != ("time", "lat", "lon"):
        wind_da = wind_da.transpose("time", "lat", "lon")

    print("Materializing data into memory (compute)...")
    wind_da = wind_da.compute()
    print(wind_da)

    # dataset + metadata
    ds_out = wind_da.to_dataset()
    ds_out = ds_out.assign_coords(
        lat=ds_out["lat"].astype("float32"), lon=ds_out["lon"].astype("float32")
    )

    metadata = {
        "start": args.start,
        "end": args.end,
        "bbox": list(map(float, args.bbox)),
        "height": args.height,
        "edh_path": args.edh_path or EDH_DEFAULT_PATH,
        "time_chunk": int(args.edh_time_chunk),
        "min_valid": int(args.min_valid),
        "source": "era5_edh",
        "tile_lat": int(args.tile_lat),
        "tile_lon": int(args.tile_lon),
    }
    # copy possible provenance attrs
    prov_keys = ("coords_lon_convention", "coords_lat_order", "coords_normalization_actions")
    prov = {k: wind_da.attrs.get(k) for k in prov_keys if k in wind_da.attrs}
    metadata.update(prov)
    ds_out.attrs.update(metadata)

    # encoding/chunking
    # Note: This script does not configure a compressor variable. If compression is
    # desired here, add a COMPRESSOR variable (e.g. BloscCodec) and include it in
    # the encoding dict for the 'wind' entry: "compressor": COMPRESSOR
    chunks_time = min(int(args.edh_time_chunk), int(ds_out.sizes.get("time", args.edh_time_chunk)))
    chunks_lat = min(int(args.tile_lat), int(ds_out.sizes.get("lat", args.tile_lat)))
    chunks_lon = min(int(args.tile_lon), int(ds_out.sizes.get("lon", args.tile_lon)))
    encoding = {
        "wind": {"dtype": "float32", "chunks": (chunks_time, chunks_lat, chunks_lon)},
        "lat": {"dtype": "float32"},
        "lon": {"dtype": "float32"},
    }

    # write zarr with overwrite handling
    if out_path.exists():
        if args.overwrite:
            if out_path.is_dir():
                shutil.rmtree(out_path)
            else:
                out_path.unlink()
        else:
            raise ValueError(f"Output {out_path} exists; pass --overwrite to replace.")

    print("Writing Zarr:", out_path)

    # Clear all legacy/numcodecs encodings (including coords) to avoid
    # numcodecs.Blosc incompatibility with zarr v3
    for v in ds_out.data_vars:
        ds_out[v].encoding = {}
    for c in ds_out.coords:
        ds_out.coords[c].encoding = {}
    ds_out.encoding = {}

    ds_out.to_zarr(str(out_path), mode="w", compute=True, encoding=encoding)

    # summary
    summary = {
        "bbox": list(map(float, args.bbox)),
        "start": args.start,
        "end": args.end,
        "out": str(out_path),
        "tile_lat": int(args.tile_lat),
        "tile_lon": int(args.tile_lon),
        "min_valid": int(args.min_valid),
        "height": args.height,
        "written_vars": list(ds_out.data_vars.keys()),
    }
    summary_path = Path(str(out_path) + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary written:", summary_path)
    print("DONE")


if __name__ == "__main__":
    main()
