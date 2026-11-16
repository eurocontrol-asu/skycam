from __future__ import annotations

import numpy as np
import xarray as xr

from skywind.config import Settings
from skywind.era5 import Era5
from skywind.field.data import Points, Prediction
from skywind.windfield import CorrectedWind


class ConstantResidual:
    name = "constant"
    prior_var = 4.0

    def predict(self, points: Points, variance: object = None) -> Prediction:
        n = len(points)
        return Prediction(np.full(n, 1.5), np.full(n, -0.5), np.full(n, 1.0), np.full(n, 4.0), np.zeros(n), 4.0)


def test_corrected_wind_adds_residual_and_reports_uncertainty(era5_ds: xr.Dataset) -> None:
    cw = CorrectedWind(Era5(era5_ds), ConstantResidual(), Settings())
    t = era5_ds["time"].values[1].astype("datetime64[s]").astype(float)
    out = cw.at([t, t], [250.0, 260.0], [48.6, 48.7], [2.3, 2.4])
    np.testing.assert_allclose(out["u"], 11.5)
    np.testing.assert_allclose(out["v"], -5.5)
    np.testing.assert_allclose(out["std_v"], 2.0)
    np.testing.assert_allclose(out["information"], 1 - 0.5 * 5.0 / 4.0)
    alt = cw.at_altitude(t, 10363.0, 48.6, 2.3)
    np.testing.assert_allclose(alt["u"], 11.5)
