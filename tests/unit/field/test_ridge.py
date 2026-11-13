from __future__ import annotations

import numpy as np

from skywind.config import RidgeConfig
from skywind.field.ridge import RidgeEstimator
from tests.unit.field.test_gp import grid_points, tracks, true_field


def test_ridge_recovers_field_from_crossing_traffic() -> None:
    obs = tracks([0.0, 90.0, 45.0, 135.0])
    pts = grid_points()
    pred = RidgeEstimator(obs, RidgeConfig(length_h_km=30, length_z_m=500, length_t_min=60)).predict(pts)
    du, dv = true_field(pts.X[:, 0], pts.X[:, 1])
    assert np.corrcoef(pred.du, du)[0, 1] > 0.9
    assert np.corrcoef(pred.dv, dv)[0, 1] > 0.9


def test_ridge_loao_ignores_own_aircraft() -> None:
    obs = tracks([90.0], n_tracks=1)  # a single aircraft: nothing else to learn from
    t = RidgeEstimator(obs, RidgeConfig()).loao()
    np.testing.assert_allclose(t["pred"], 0.0, atol=1e-12)
