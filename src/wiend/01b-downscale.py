#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import truststore
import zarr

from wiend.utils import (
    load_weibull_params_from_zarr,
    load_wind_from_zarr,
    map_weibull_field_to_zarr,
)

truststore.inject_into_ssl()

# CLI usage:
# --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX   Bounding box (lon_min lat_min lon_max lat_max)
# --start YYYY-MM-DD                       Start date (inclusive)
# --end YYYY-MM-DD                         End date (inclusive)
# --spacing FLOAT                          Grid spacing in degrees (required)
# --mapping_method STR                      Mapping method: qm, dummy, mc, sh2, sh2-simple
# --input PATH                              Path to input original wind Zarr (required)
# --out PATH                                Output path for mapped wind Zarr (required)
# Optional flags: --no-crop, --edh-key, --height

STORAGE_DTYPE = "int16"
SCALE = 0.1

CHUNKS = {"time": 168, "lat": 100, "lon": 400}

# COMPRESSOR: used when saving mapped wind Zarrs. Blosc (zstd) with bitshuffle is
# chosen for good speed / compression tradeoff; clevel=5. To change compression,
# modify these parameters here (or make them CLI options).
COMPRESSOR = zarr.codecs.BloscCodec(
    cname="zstd",
    clevel=5,
    shuffle=zarr.codecs.BloscShuffle.bitshuffle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--mapping_method", required=True, help="Mapping method: qm, dummy, mc, sh2, sh2-simple"
    )
    parser.add_argument("--input", required=True, help="Path to input original wind Zarr")
    parser.add_argument("--out", required=True, help="Output path for mapped wind Zarr")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box as lon_min lat_min lon_max lat_max",
    )
    parser.add_argument("--spacing", type=float, required=True, help="Grid spacing in degrees")
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable cropping of Weibull parameter Zarrs to bbox",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lon_min, lat_min, lon_max, lat_max = args.bbox
    mapping_method = args.mapping_method
    # mapping/loading functions expect bbox as (lat_min, lat_max, lon_min, lon_max)
    bbox = (lat_min, lat_max, lon_min, lon_max)

    # require CLI-specified input/output paths for consistency
    path_original_wind_zarr = args.input
    path_save_mapping = str(Path(args.out).resolve())  # convert to absolute path

    path_params_baseline_zarr = (
        "/Users/florianscheiber/PycharmProjects/downscaling_wind/data/data-after-task21/"
        "weibull-parameters-era5-edh/weibull-parameters_era5-edh-21/"
        "a-k_era5-edh_EUR_10m_2001-2020_spacing-0-0025.zarr"
    )

    path_params_manipulator_zarr = (
        "/Users/florianscheiber/PycharmProjects/downscaling_wind/data/data-after-task21/"
        "weibull-parameters-corrected/"
        "weibull-parameters_EUR-isd_2001-2020-h_reg16-12-only-onshore_mle_train70p-"
        "mincount-87660_10m_spacing-0-0025.zarr"
    )

    original_wind = load_wind_from_zarr(path_original_wind_zarr)

    k_b, a_b = load_weibull_params_from_zarr(
        path_params_baseline_zarr,
        bbox=bbox,
        crop=not args.no_crop,
    )
    k_m, a_m = load_weibull_params_from_zarr(
        path_params_manipulator_zarr,
        bbox=bbox,
        crop=not args.no_crop,
    )

    map_weibull_field_to_zarr(
        mapping_method,
        original_wind,
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

    root = zarr.open(path_save_mapping, mode="r")
    arr = root["wind"]

    print("dtype:", arr.dtype)
    print("chunks:", arr.chunks)
    print("compressors:", arr.compressors)
    print("filters:", getattr(arr, "filters", None))


if __name__ == "__main__":
    main()
