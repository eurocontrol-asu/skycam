from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from skywind.config import ExportConfig, Site
from skywind.export import era5_on_grid, export_grid, residual_dataset, to_met_dataset
from tests.unit.test_windfield import ConstantResidual


def test_export_grid_contains_native_levels_and_finer_times(era5_ds: xr.Dataset) -> None:
    cfg = ExportConfig(half_width_deg=1.0, level_step_hpa=10.0, time_step_min=10.0, pad_before_min=20,
                       pad_after_min=0)
    g = export_grid(era5_ds, Site(), cfg, pd.Timestamp("2024-03-18T13:00", tz="UTC"),
                    pd.Timestamp("2024-03-18T14:00", tz="UTC"))
    assert {200.0, 250.0, 260.0}.issubset(set(g["level"]))
    assert len(g["time"]) == 9 and np.all(np.abs(g["latitude"] - 48.6005) <= 1.0)


def test_exported_dataset_has_corrected_winds_and_uncertainty(era5_ds: xr.Dataset) -> None:
    cfg = ExportConfig(half_width_deg=0.5, level_step_hpa=25.0, pad_before_min=0, pad_after_min=0)
    g = export_grid(era5_ds, Site(), cfg, pd.Timestamp("2024-03-18T13:00", tz="UTC"),
                    pd.Timestamp("2024-03-18T13:30", tz="UTC"))
    on_grid = era5_on_grid(era5_ds, g)
    res = residual_dataset(on_grid, ConstantResidual(), Site())
    ds = to_met_dataset(on_grid, res, {"estimator": "constant"})
    assert ds["eastward_wind"].dims == ("longitude", "latitude", "level", "time")
    np.testing.assert_allclose(ds["eastward_wind"], 11.5)
    np.testing.assert_allclose(ds["era5_northward_wind"], -5.0)
    np.testing.assert_allclose(ds["residual_northward_wind_std"], 2.0)
    assert ds.attrs["estimator"] == "constant"
