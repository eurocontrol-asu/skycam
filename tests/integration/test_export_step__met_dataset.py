from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from skywind.advect import advect, met_dataset
from skywind.config import Settings
from skywind.era5 import Era5, load_era5
from skywind.field.data import ObsSet
from skywind.field.gp import GPEstimator, GPParams
from skywind.pipeline import export_step
from skywind.windfield import CorrectedWind
from tests.integration.conftest import DAY

pytestmark = pytest.mark.integration


def test_exported_field_is_a_drop_in_met_dataset(tmp_settings: Settings) -> None:
    era = Era5(load_era5(tmp_settings.paths.era5(DAY)))
    w = pd.read_parquet(tmp_settings.paths.prototype_dir / "modes" / f"winds_{DAY}.parquet")
    w = w.rename(columns={"u_ehs": "u_obs", "v_ehs": "v_obs", "p": "p_hpa"}).head(400)
    est = GPEstimator(ObsSet.from_frame(w, tmp_settings.site), GPParams())
    tmp_settings.export.half_width_deg = 1.0
    tmp_settings.export.level_step_hpa = 25.0
    t0 = pd.Timestamp(w["t"].min(), unit="s", tz="UTC")
    path = export_step(tmp_settings, DAY, CorrectedWind(era, est, tmp_settings), t0, t0 + pd.Timedelta(minutes=30))
    ds = xr.open_dataset(path)
    corrected = ds["eastward_wind"] - ds["era5_eastward_wind"]
    np.testing.assert_allclose(corrected, ds["residual_eastward_wind"], atol=1e-4)
    assert float(ds["residual_eastward_wind_std"].max()) <= GPParams().sigma_field + 1e-5
    pts = pd.DataFrame({"flight_id": ["f"], "waypoint": [0], "latitude": [48.6], "longitude": [2.35],
                        "level": [250.0], "emission_time": [t0 + pd.Timedelta(minutes=10)]})
    a = advect(pts, met_dataset(ds, "era5"), t0 + pd.Timedelta(minutes=30), max_age_min=30)
    b = advect(pts, met_dataset(load_era5(tmp_settings.paths.era5(DAY))), t0 + pd.Timedelta(minutes=30),
               max_age_min=30)
    np.testing.assert_allclose(a["latitude"].to_numpy(), b["latitude"].to_numpy(), atol=1e-6)
