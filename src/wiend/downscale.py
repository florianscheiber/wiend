"""Downscaling and mapping API for wiend."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

from wiend.data_manager import (
    load_weibull_params_from_zarr,
    load_wind_from_zarr,
)
from wiend.validation import parse_bbox

DEFAULT_STORAGE_DTYPE: Literal["float32", "int16"] = "int16"
DEFAULT_SCALE = 0.1
DEFAULT_CHUNKS: dict[str, int] = {"time": 168, "lat": 100, "lon": 400}


def _default_compressor() -> Any:
    return zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=5,
        shuffle=zarr.codecs.BloscShuffle.bitshuffle,
    )


def valid_weibull_params_vec(*arrs: np.ndarray) -> np.ndarray:
    ok = np.ones(arrs[0].shape, dtype=bool)
    for arr in arrs:
        ok &= np.isfinite(arr) & (arr > 0.0)
    return ok


def _mask_invalid_2d(out: np.ndarray, valid: np.ndarray) -> None:
    flat = out.reshape(out.shape[0], -1)
    valid_flat = valid.ravel()
    if not np.all(valid_flat):
        flat[:, ~valid_flat] = np.nan


def _map_block_mean_correction(
    v64: np.ndarray,
    ab: np.ndarray,
    am: np.ndarray,
) -> np.ndarray:
    out = np.empty(v64.shape, dtype=np.float64)
    valid = valid_weibull_params_vec(ab, am)
    ab2 = np.where(valid, ab, 1.0)
    am2 = np.where(valid, am, 1.0)
    out[...] = v64 * (am2[None, :, :] / ab2[None, :, :])
    _mask_invalid_2d(out, valid)
    return out


def _map_block_qm(
    v64: np.ndarray,
    kb: np.ndarray,
    ab: np.ndarray,
    km: np.ndarray,
    am: np.ndarray,
    eps_q: float,
) -> np.ndarray:
    out = np.empty(v64.shape, dtype=np.float64)
    valid = valid_weibull_params_vec(kb, ab, km, am)

    kb2 = np.where(valid, kb, 1.0)
    ab2 = np.where(valid, ab, 1.0)
    km2 = np.where(valid, km, 1.0)
    am2 = np.where(valid, am, 1.0)

    x = v64 / ab2[None, :, :]
    q = 1.0 - np.exp(-np.power(x, kb2[None, :, :]))
    np.clip(q, eps_q, 1.0 - eps_q, out=q)
    out[...] = am2[None, :, :] * np.power(-np.log1p(-q), 1.0 / km2[None, :, :])

    _mask_invalid_2d(out, valid)
    return out


def init_wind_zarr(
    path_out: str | Path,
    *,
    time_values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    chunks: dict[str, int],
    compressor: Any,
    storage_dtype: Literal["float32", "int16"],
    scale: float = 0.02,
    add_offset: float = 0.0,
    fill_value: int = -32768,
    attrs: dict[str, Any] | None = None,
    mode: str = "w",
) -> zarr.Group:
    root = zarr.open_group(str(path_out), mode=mode)

    root.create_array(
        "time",
        data=time_values,
        overwrite=True,
        chunks=(min(time_values.size, chunks["time"]),),
        dimension_names=("time",),
    )
    root.create_array(
        "lat",
        data=lat.astype(np.float32, copy=False),
        overwrite=True,
        chunks=(min(lat.size, chunks["lat"]),),
        dimension_names=("lat",),
    )
    root.create_array(
        "lon",
        data=lon.astype(np.float32, copy=False),
        overwrite=True,
        chunks=(min(lon.size, chunks["lon"]),),
        dimension_names=("lon",),
    )

    t_size, y_size, x_size = time_values.size, lat.size, lon.size
    if storage_dtype == "float32":
        dtype = np.dtype("float32")
        fill = np.nan
        wind_attrs: dict[str, Any] = {}
    else:
        dtype = np.dtype("int16")
        fill = int(fill_value)
        wind_attrs = {
            "scale_factor": float(scale),
            "add_offset": float(add_offset),
            "_FillValue": int(fill_value),
        }

    wind_arr = root.create_array(
        "wind",
        shape=(t_size, y_size, x_size),
        chunks=(chunks["time"], chunks["lat"], chunks["lon"]),
        dtype=dtype,
        overwrite=True,
        fill_value=fill,
        compressors=compressor,
        dimension_names=("time", "lat", "lon"),
    )
    wind_arr.attrs.update(wind_attrs)

    if attrs:
        root.attrs.update(attrs)
    root.attrs.update({"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]})
    return root


def map_wind_block(
    *,
    method: Literal["qm", "qdm", "sdm", "dummy", "sh2", "sh2-simple", "mc"],
    v: np.ndarray,
    k_b: np.ndarray,
    a_b: np.ndarray,
    k_m: np.ndarray,
    a_m: np.ndarray,
    k_bc: np.ndarray | None,
    a_bc: np.ndarray | None,
    use_shape_scaling: bool,
    min_k: float,
    eps_q: float,
) -> np.ndarray:
    del k_bc, a_bc, use_shape_scaling, min_k

    if method == "dummy":
        out64 = v
    elif method == "qm":
        out64 = _map_block_qm(v, k_b, a_b, k_m, a_m, eps_q)
    elif method == "mc":
        out64 = _map_block_mean_correction(v, a_b, a_m)
    else:
        raise ValueError(f"unknown method {method!r}")

    return out64.astype(np.float32, copy=False)


def encode_float32_to_int16_block(
    x: np.ndarray,
    *,
    scale: float,
    add_offset: float,
    fill_value: int,
) -> tuple[np.ndarray, int, int]:
    if scale <= 0:
        raise ValueError("scale must be > 0 for int16 encoding")

    iinfo = np.iinfo(np.int16)
    reserved_min = iinfo.min
    usable_min = reserved_min + 1 if fill_value == reserved_min else iinfo.min
    usable_max = iinfo.max

    enc = np.empty(x.shape, dtype=np.int16)
    finite = np.isfinite(x)
    n_filled = int((~finite).sum())

    y = np.rint((x.astype(np.float64, copy=False) - add_offset) / scale)
    y_clipped = np.clip(y, usable_min, usable_max)
    n_clipped = int((finite & (y_clipped != y)).sum())

    enc[finite] = y_clipped[finite].astype(np.int16, copy=False)
    enc[~finite] = np.int16(fill_value)
    return enc, n_clipped, n_filled


def write_wind_block_to_zarr(
    root: zarr.Group,
    *,
    t0: int,
    t1: int,
    i0: int,
    i1: int,
    block_float32: np.ndarray,
    storage_dtype: Literal["float32", "int16"],
    scale: float = 0.02,
    add_offset: float = 0.0,
    fill_value: int = -32768,
) -> tuple[int, int]:
    wind_arr = root["wind"]

    if storage_dtype == "float32":
        wind_arr[t0:t1, i0:i1, :] = block_float32.astype(np.float32, copy=False)
        return 0, 0

    enc, n_clipped, n_filled = encode_float32_to_int16_block(
        block_float32,
        scale=scale,
        add_offset=add_offset,
        fill_value=fill_value,
    )
    wind_arr[t0:t1, i0:i1, :] = enc
    return n_clipped, n_filled


def map_weibull_field_to_zarr(
    mapping_method: str,
    original_wind: xr.DataArray,
    k_b: xr.DataArray,
    a_b: xr.DataArray,
    k_m: xr.DataArray,
    a_m: xr.DataArray,
    path_out: str | Path,
    k_bc: xr.DataArray | None,
    a_bc: xr.DataArray | None,
    *,
    chunks: dict[str, int],
    compressor: Any,
    storage_dtype: Literal["float32", "int16"],
    interp_method: str = "linear",
    use_shape_scaling: bool = True,
    min_k: float = 0.01,
    eps_q: float = 1e-6,
    mode: str = "w",
    scale: float = 0.02,
    add_offset: float = 0.0,
    fill_value: int = -32768,
) -> None:
    method = mapping_method.lower()
    if method not in ("qm", "qdm", "sdm", "dummy", "mc", "sh2", "sh2-simple"):
        raise ValueError(f"Unknown mapping_method: {mapping_method!r}")
    if method in ("qdm", "sdm") and (k_bc is None or a_bc is None):
        raise ValueError(f"mapping_method {method!r} requires k_bc and a_bc.")

    lat_t = np.asarray(k_b["lat"].values)
    lon_t = np.asarray(k_b["lon"].values)

    def _grid_allclose(da: xr.DataArray, name: str) -> None:
        if da.dims != ("lat", "lon"):
            raise ValueError(f"{name} must have dims ('lat','lon'), got {da.dims}")
        if not np.allclose(da["lat"].values, lat_t, rtol=0, atol=1e-10):
            raise ValueError(f"{name} lat grid differs from k_b lat grid.")
        if not np.allclose(da["lon"].values, lon_t, rtol=0, atol=1e-10):
            raise ValueError(f"{name} lon grid differs from k_b lon grid.")

    _grid_allclose(a_b, "a_b")
    _grid_allclose(k_m, "k_m")
    _grid_allclose(a_m, "a_m")
    if method in ("qdm", "sdm"):
        _grid_allclose(k_bc, "k_bc")  # type: ignore[arg-type]
        _grid_allclose(a_bc, "a_bc")  # type: ignore[arg-type]

    t_size = original_wind.sizes["time"]
    y_size = lat_t.size

    k_b_np = k_b.to_numpy().astype(np.float64, copy=False)
    a_b_np = a_b.to_numpy().astype(np.float64, copy=False)
    k_m_np = k_m.to_numpy().astype(np.float64, copy=False)
    a_m_np = a_m.to_numpy().astype(np.float64, copy=False)
    if method in ("qdm", "sdm"):
        k_bc_np = k_bc.to_numpy().astype(np.float64, copy=False)  # type: ignore[union-attr]
        a_bc_np = a_bc.to_numpy().astype(np.float64, copy=False)  # type: ignore[union-attr]
    else:
        k_bc_np = None
        a_bc_np = None

    attrs = {
        "processing": "mapped",
        "mapping_method": method,
        "interp_method": interp_method,
        "time_min": str(original_wind["time"].values[0]),
        "time_max": str(original_wind["time"].values[-1]),
        "lat_min": float(lat_t[0]),
        "lat_max": float(lat_t[-1]),
        "lon_min": float(lon_t[0]),
        "lon_max": float(lon_t[-1]),
        "height_m": 10,
        "unit wind": "m s-1",
        "storage_dtype": storage_dtype,
        "chunks_time": int(chunks["time"]),
        "chunks_lat": int(chunks["lat"]),
        "chunks_lon": int(chunks["lon"]),
    }

    root = init_wind_zarr(
        path_out,
        time_values=np.asarray(original_wind["time"].values),
        lat=lat_t,
        lon=lon_t,
        chunks=chunks,
        compressor=compressor,
        storage_dtype=storage_dtype,
        scale=scale,
        add_offset=add_offset,
        fill_value=fill_value,
        attrs=attrs,
        mode=mode,
    )

    block_lat = int(chunks["lat"])
    time_block = int(chunks["time"])

    n_clipped_total = 0
    n_filled_total = 0

    n_t = (t_size + time_block - 1) // time_block
    n_i = (y_size + block_lat - 1) // block_lat
    total = n_t * n_i
    pbar = tqdm(total=total, desc=f"mapping: {method}", unit="block")

    t_interp_sum = 0.0
    t_map_sum = 0.0
    n_blocks = 0

    for t0 in range(0, t_size, time_block):
        t1 = min(t_size, t0 + time_block)
        for i0 in range(0, y_size, block_lat):
            i1 = min(y_size, i0 + block_lat)
            lat_block = lat_t[i0:i1]

            t_interp0 = time.perf_counter()
            v_da = original_wind.isel(time=slice(t0, t1)).interp(
                lat=lat_block,
                lon=lon_t,
                method=interp_method,
            )
            v64 = v_da.to_numpy().astype(np.float64, copy=False)
            t_interp1 = time.perf_counter()

            t_map0 = time.perf_counter()
            block = map_wind_block(
                method=method,  # type: ignore[arg-type]
                v=v64,
                k_b=k_b_np[i0:i1, :],
                a_b=a_b_np[i0:i1, :],
                k_m=k_m_np[i0:i1, :],
                a_m=a_m_np[i0:i1, :],
                k_bc=(k_bc_np[i0:i1, :] if k_bc_np is not None else None),
                a_bc=(a_bc_np[i0:i1, :] if a_bc_np is not None else None),
                use_shape_scaling=use_shape_scaling,
                min_k=min_k,
                eps_q=eps_q,
            )
            t_map1 = time.perf_counter()

            t_interp_sum += t_interp1 - t_interp0
            t_map_sum += t_map1 - t_map0
            n_blocks += 1

            n_clipped, n_filled = write_wind_block_to_zarr(
                root,
                t0=t0,
                t1=t1,
                i0=i0,
                i1=i1,
                block_float32=block,
                storage_dtype=storage_dtype,
                scale=scale,
                add_offset=add_offset,
                fill_value=fill_value,
            )
            n_clipped_total += n_clipped
            n_filled_total += n_filled
            pbar.update(1)

    pbar.close()

    if n_blocks > 0:
        t_total = t_interp_sum + t_map_sum
        print(
            f"Timing summary:\n"
            f"  interpolation total : {t_interp_sum:.1f}s ({t_interp_sum / t_total:.1%})\n"
            f"  mapping total       : {t_map_sum:.1f}s ({t_map_sum / t_total:.1%})\n"
            f"  avg per block       : interp {t_interp_sum / n_blocks:.3f}s, "
            f"map {t_map_sum / n_blocks:.3f}s\n"
            f"  blocks              : {n_blocks}"
        )

    print(f"n_clipped: {n_clipped_total}, n_filled: {n_filled_total}")
    root.attrs.update(
        {
            "int16_clipped_values": int(n_clipped_total),
            "int16_filled_values": int(n_filled_total),
        }
    )


def downscale(
    *,
    input_data: str | Path | xr.DataArray,
    out: str | Path,
    bbox: tuple[float, float, float, float],
    mapping_method: str,
    params_baseline_path: str | Path,
    params_manipulator_path: str | Path,
    bbox_order: str = "lonlat",
    no_crop: bool = False,
    overwrite: bool = False,
    chunks: dict[str, int] | None = None,
    compressor: Any | None = None,
    storage_dtype: Literal["float32", "int16"] = DEFAULT_STORAGE_DTYPE,
    scale: float = DEFAULT_SCALE,
    interp_method: str = "linear",
    use_shape_scaling: bool = True,
    min_k: float = 0.01,
    eps_q: float = 1e-6,
) -> Path:
    """Map ERA5 wind to the Weibull reference grid and write result to Zarr."""
    out_path = Path(out)
    if out_path.exists():
        if not overwrite:
            raise ValueError(f"Output {out_path} exists; set overwrite=True to replace.")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    if isinstance(input_data, xr.DataArray):
        original_wind = input_data
    else:
        original_wind = load_wind_from_zarr(input_data)

    lon_min, lat_min, lon_max, lat_max = parse_bbox(bbox, bbox_order=bbox_order)
    bbox_for_params = (lat_min, lat_max, lon_min, lon_max)

    k_b, a_b = load_weibull_params_from_zarr(
        params_baseline_path,
        bbox=bbox_for_params,
        crop=not no_crop,
    )
    k_m, a_m = load_weibull_params_from_zarr(
        params_manipulator_path,
        bbox=bbox_for_params,
        crop=not no_crop,
    )

    map_weibull_field_to_zarr(
        mapping_method,
        original_wind,
        k_b,
        a_b,
        k_m,
        a_m,
        path_out=out_path,
        k_bc=None,
        a_bc=None,
        chunks=chunks or DEFAULT_CHUNKS,
        compressor=compressor if compressor is not None else _default_compressor(),
        scale=scale,
        storage_dtype=storage_dtype,
        interp_method=interp_method,
        use_shape_scaling=use_shape_scaling,
        min_k=min_k,
        eps_q=eps_q,
    )
    return out_path
