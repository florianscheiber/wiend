from __future__ import annotations

from importlib import import_module

import numpy as np
import xarray as xr

download_module = import_module("wiend.download")


def _sample_wind() -> xr.DataArray:
    return xr.DataArray(
        np.ones((4, 3, 5), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(4),
            "lat": np.arange(3, dtype=np.float32),
            "lon": np.arange(5, dtype=np.float32),
        },
        name="wind",
    )


def test_download_writes_chunked_output_without_eager_return_compute(monkeypatch, tmp_path):
    monkeypatch.setattr(download_module, "_get_wind_da", lambda **_: _sample_wind())

    result = download_module.download(
        bbox=(0, 0, 1, 1),
        start="2020-01-01",
        end="2020-01-02",
        out=tmp_path / "wind.zarr",
        edh_key="test",
        edh_time_chunk=2,
        tile_lat=2,
        tile_lon=3,
    )

    assert result.name == "wind"
    written = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)["wind"]
    assert written.chunks == ((2, 2), (2, 1), (3, 2))
    np.testing.assert_array_equal(written.values, np.ones((4, 3, 5), dtype=np.float32))


def test_download_and_downscale_removes_intermediate_store_after_success(monkeypatch, tmp_path):
    intermediate_paths: list[str] = []

    def fake_download(**kwargs):
        path = kwargs["out"]
        intermediate_paths.append(str(path))
        path.mkdir()
        return _sample_wind()

    def fake_downscale(**kwargs):
        assert kwargs["input_data"] == tmp_path / "result.zarr.era5.zarr"
        return kwargs["out"]

    monkeypatch.setattr(download_module, "download", fake_download)
    monkeypatch.setattr(download_module, "downscale", fake_downscale)

    result = download_module.download_and_downscale(
        bbox=(0, 0, 1, 1),
        start="2020-01-01",
        end="2020-01-02",
        out=tmp_path / "result.zarr",
        mapping_method="dummy",
        edh_key="test",
    )

    assert result == tmp_path / "result.zarr"
    assert intermediate_paths
    assert not (tmp_path / "result.zarr.era5.zarr").exists()


def test_download_and_downscale_pointwise_writes_station_output(monkeypatch, tmp_path):
    stations_path = tmp_path / "stations.zarr"
    xr.Dataset(
        {"wind": (("time", "station"), np.ones((1, 2)))},
        coords={
            "time": [999],
            "station": ["a", "b"],
            "lat": ("station", [50.1, 50.2]),
            "lon": ("station", [8.1, 8.2]),
        },
    ).to_zarr(stations_path)

    wind = xr.DataArray(
        np.array([[2.0, 3.0]], dtype=np.float32),
        dims=("time", "station"),
        coords={
            "time": [0],
            "station": ["a", "b"],
            "lat": ("station", [50.1, 50.2]),
            "lon": ("station", [8.1, 8.2]),
            "lat_era5": ("station", [50.0, 50.25]),
            "lon_era5": ("station", [8.0, 8.25]),
        },
        name="wind",
    )
    params = xr.DataArray(
        np.ones(2, dtype=np.float32),
        dims=("station",),
        coords={"station": ["a", "b"]},
    )
    monkeypatch.setattr(
        download_module,
        "get_pointwise_wind_da",
        lambda **_: wind,
    )
    monkeypatch.setattr(
        download_module,
        "load_pointwise_params",
        lambda *args, **kwargs: (params, params, np.array([]), np.array([])),
    )

    result = download_module.download_and_downscale(
        stations=stations_path,
        start="2020-01-01",
        end="2020-01-02",
        out=tmp_path / "result.zarr",
        mapping_method="dummy",
        params_baseline_path=tmp_path / "baseline.zarr",
        params_manipulator_path=tmp_path / "corrected.zarr",
        edh_key="test",
    )

    assert result == tmp_path / "result.zarr"
    output = xr.open_zarr(result, consolidated=False)
    assert output["time"].values.tolist() == [0]
    assert output["station"].values.tolist() == ["a", "b"]
    np.testing.assert_allclose(output["lat"].values, [50.1, 50.2])
    np.testing.assert_allclose(output["lon"].values, [8.1, 8.2])
    np.testing.assert_allclose(output["lat_era5"].values, [50.0, 50.25])
