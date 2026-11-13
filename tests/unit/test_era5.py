from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from skywind.era5 import Era5


def test_interpolation_linear_in_pressure_and_time(era5_ds: xr.Dataset) -> None:
    ds = era5_ds.copy(deep=True)
    ds["u"] = ds["u"] + ds["level"] * 0.1 + (ds["time"] - ds["time"][0]) / np.timedelta64(1, "h")
    era = Era5(ds)
    t = pd.Timestamp("2024-03-18T12:30").value / 1e9
    u = era("u", t, 225.0, 48.6, 2.35)
    np.testing.assert_allclose(u, 10.0 + 22.5 + 0.5)
    assert np.isnan(era("u", t, 225.0, 60.0, 2.35))
