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


def test_ridge_grid_solver_matches_pointwise() -> None:
    obs = tracks([0.0, 90.0, 45.0], n_tracks=4)
    est = RidgeEstimator(obs, RidgeConfig(length_h_km=30, length_z_m=300, length_t_min=20))
    times, z = np.array([1.7e9 - 300, 1.7e9]), np.array([9800.0, 10000.0, 10300.0])
    xy = np.stack(np.meshgrid(np.linspace(-80, 80, 5), np.linspace(-60, 60, 4)), -1).reshape(-1, 2)
    grid = est.predict_grid(times, z, xy)
    T, Z, H = np.meshgrid(times, z, np.arange(len(xy)), indexing="ij")
    from skywind.field.data import Points

    pts = est.predict(
        Points(np.stack([xy[H.ravel(), 0], xy[H.ravel(), 1], Z.ravel(), T.ravel()], 1), np.zeros((T.size, 2)))
    )
    np.testing.assert_allclose(grid["du"].ravel(), pts.du, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(grid["var_v"].ravel(), pts.var_v, rtol=1e-9)
