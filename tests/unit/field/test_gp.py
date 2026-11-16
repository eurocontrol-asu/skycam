from __future__ import annotations

import numpy as np
import pytest

from skywind.field.data import ALONG, ObsSet, Points
from skywind.field.gp import GPEstimator, GPParams

T0 = 1.7e9
PARAMS = GPParams(
    length_h_km=60.0,
    length_z_m=400.0,
    length_t_min=60.0,
    sigma_field=3.0,
    sigma_bias=0.5,
    mean_bias=0.0,
    sigma_noise=0.5,
)


def true_field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 3.0 * np.sin(x / 70.0), 2.5 * np.cos(y / 55.0)


def tracks(headings_deg: list[float], n_tracks: int = 8, seed: int = 0) -> ObsSet:
    """Straight tracks across a 300 km box at 10 km altitude, one observation every 30 s (about 7 km)."""
    rng = np.random.default_rng(seed)
    rows, ac = [], 0
    for hdg in headings_deg:
        hx, hy = np.sin(np.radians(hdg)), np.cos(np.radians(hdg))
        for k in range(n_tracks):
            offset = -140 + 280 * k / max(n_tracks - 1, 1)
            s = np.arange(-150, 150, 7.0)
            x, y = s * hx + offset * hy, s * hy - offset * hx
            keep = (np.abs(x) < 150) & (np.abs(y) < 150)
            rows.append((x[keep], y[keep], np.full(keep.sum(), hdg), np.full(keep.sum(), ac)))
            ac += 1
    x, y, h, a = (np.concatenate(v) for v in zip(*rows, strict=True))
    du, dv = true_field(x, y)
    e = np.stack([np.sin(np.radians(h)), np.cos(np.radians(h))], 1)
    bias = rng.normal(0, 0.5, ac)[a.astype(int)]
    obs_y = du * e[:, 0] + dv * e[:, 1] + bias + rng.normal(0, 0.5, len(x))
    X = np.stack([x, y, np.full_like(x, 10000.0), np.full_like(x, T0)], 1)
    n = len(x)
    return ObsSet(
        X,
        e,
        obs_y,
        np.zeros(n, np.int64),
        a.astype(np.int64),
        np.zeros((n, 2)),
        np.array(["d"] * n),
        np.array([f"a{int(i)}" for i in a]),
    )


def grid_points() -> Points:
    gx, gy = np.meshgrid(np.linspace(-100, 100, 9), np.linspace(-100, 100, 9))
    X = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 10000.0), np.full(gx.size, T0)], 1)
    return Points(X, np.zeros((len(X), 2)))


def test_crossing_traffic_recovers_both_components() -> None:
    obs = tracks([0.0, 90.0, 45.0, 135.0])
    pts = grid_points()
    pred = GPEstimator(obs, PARAMS, tile_minutes=60).predict(pts)
    du, dv = true_field(pts.X[:, 0], pts.X[:, 1])
    assert np.sqrt(np.mean((pred.du - du) ** 2)) < 0.5
    assert np.sqrt(np.mean((pred.dv - dv) ** 2)) < 0.5
    assert np.all(pred.std_u < 0.5 * PARAMS.sigma_field)
    assert np.all(pred.std_v < 0.5 * PARAMS.sigma_field)


def test_one_directional_traffic_leaves_orthogonal_component_at_prior() -> None:
    obs = tracks([90.0])  # everybody flies east: only du is observed
    pts = grid_points()
    pred = GPEstimator(obs, PARAMS, tile_minutes=60).predict(pts)
    du, _ = true_field(pts.X[:, 0], pts.X[:, 1])
    assert np.sqrt(np.mean((pred.du - du) ** 2)) < 0.6
    np.testing.assert_allclose(pred.dv, 0.0, atol=1e-9)
    np.testing.assert_allclose(pred.var_v, PARAMS.sigma_field**2, rtol=1e-9)
    assert np.all(pred.var_u < 0.3 * PARAMS.sigma_field**2)


def test_far_from_observations_falls_back_to_era5_with_prior_uncertainty() -> None:
    obs = tracks([0.0, 90.0])
    far = Points(np.array([[2000.0, 2000.0, 10000.0, T0], [0.0, 0.0, 10000.0, T0 + 86400]]), np.zeros((2, 2)))
    pred = GPEstimator(obs, PARAMS).predict(far)
    np.testing.assert_allclose(pred.du, 0.0, atol=1e-12)
    np.testing.assert_allclose(pred.var_u, PARAMS.sigma_field**2)
    np.testing.assert_allclose(pred.information, 0.0, atol=1e-12)


def _brute_force_loao(obs: ObsSet, p: GPParams) -> np.ndarray:
    out = np.zeros(len(obs))
    for a in np.unique(obs.aircraft):
        own = obs.aircraft == a
        rest = GPEstimator(obs.subset(~own), p, cutoff=50.0, tile_minutes=1e6)
        pr = rest.predict(Points(obs.X[own], obs.bg[own]))
        out[own] = pr.du * obs.e[own, 0] + pr.dv * obs.e[own, 1]
    return out


@pytest.mark.parametrize("use_bias", [True, False])
def test_loao_matches_refitting_without_each_aircraft(use_bias: bool) -> None:
    obs = tracks([0.0, 90.0], n_tracks=3)
    p = GPParams(**{**PARAMS.as_dict(), "use_bias": use_bias, "mean_bias": 0.3})
    fast = GPEstimator(obs, p, cutoff=50.0, tile_minutes=1e6).loao(vectors=True)
    np.testing.assert_allclose(fast["pred"].to_numpy(), _brute_force_loao(obs, p)[fast["obs_index"]], atol=1e-6)
    np.testing.assert_allclose(
        fast["du"] * obs.e[fast["obs_index"], 0] + fast["dv"] * obs.e[fast["obs_index"], 1], fast["pred"], atol=1e-6
    )
    assert np.all(obs.kind[fast["obs_index"]] == ALONG)


def test_loao_variance_includes_bias_and_noise() -> None:
    obs = tracks([0.0, 90.0], n_tracks=3)
    t = GPEstimator(obs, PARAMS, cutoff=50.0, tile_minutes=1e6).loao()
    np.testing.assert_allclose(t["var_total"] - t["var_field"], PARAMS.sigma_bias**2 + PARAMS.sigma_noise**2)


def test_flow_following_shifts_structures_with_background_wind() -> None:
    # one observation at x=0 at T0; background wind 50 m/s eastward; query 20 min later
    X = np.array([[0.0, 0.0, 10000.0, T0]])
    obs = ObsSet(
        X,
        np.array([[1.0, 0.0]]),
        np.array([3.0]),
        np.zeros(1, np.int64),
        np.zeros(1, np.int64),
        np.array([[50.0, 0.0]]),
        np.array(["d"]),
        np.array(["a"]),
    )
    p = GPParams(
        length_h_km=20,
        length_z_m=300,
        length_t_min=60,
        sigma_field=2,
        sigma_bias=0.1,
        mean_bias=0,
        sigma_noise=0.3,
        flow_following=True,
    )
    q = Points(np.array([[60.0, 0.0, 10000.0, T0 + 1200], [0.0, 0.0, 10000.0, T0 + 1200]]), np.full((2, 2), 0.0))
    q = Points(q.X, np.array([[50.0, 0.0], [50.0, 0.0]]))
    pred = GPEstimator(obs, p).predict(q)
    assert pred.du[0] > 2.0 > pred.du[1]  # the structure moved 60 km downstream


@pytest.mark.parametrize("flow", [False, True])
def test_grid_solver_matches_pointwise_prediction(flow: bool) -> None:
    obs = tracks([0.0, 90.0, 45.0], n_tracks=4)
    obs = ObsSet(
        obs.X
        + np.c_[
            np.zeros((len(obs), 2)),
            np.linspace(-300, 300, len(obs))[:, None],
            np.linspace(-1800, 1800, len(obs))[:, None],
        ],
        obs.e,
        obs.y,
        obs.kind,
        obs.aircraft,
        np.tile([[30.0, -10.0]], (len(obs), 1)),
        obs.day,
        obs.icao24,
    )
    p = GPParams(**{**PARAMS.as_dict(), "kernel": "matern32", "flow_following": flow})
    est = GPEstimator(obs, p, cutoff=50.0, tile_minutes=1e-3, max_neighbours=100_000)
    times, z = np.array([T0 - 600, T0 + 600]), np.array([9800.0, 10200.0])
    xy = np.stack(np.meshgrid(np.linspace(-80, 80, 5), np.linspace(-60, 60, 4)), -1).reshape(-1, 2)
    grid = est.predict_grid(times, z, xy)
    T, Z, H = np.meshgrid(times, z, np.arange(len(xy)), indexing="ij")
    X = np.stack([xy[H.ravel(), 0], xy[H.ravel(), 1], Z.ravel(), T.ravel()], 1)
    pts = est.predict(Points(X, np.zeros((len(X), 2))))
    np.testing.assert_allclose(grid["du"].ravel(), pts.du, atol=1e-8)
    np.testing.assert_allclose(grid["var_v"].ravel(), pts.var_v, atol=1e-8)


def test_two_scale_grid_solver_and_prior_variance() -> None:
    obs = tracks([0.0, 90.0], n_tracks=3)
    p = GPParams(
        **{**PARAMS.as_dict(), "kernel": "matern32", "sigma_field2": 1.5, "length_h2_km": 300.0, "length_t2_min": 600.0}
    )
    est = GPEstimator(obs, p, cutoff=50.0, tile_minutes=1e-3, max_neighbours=100_000)
    assert est.prior_var == pytest.approx(PARAMS.sigma_field**2 + 1.5**2)
    times, z = np.array([T0]), np.array([10000.0])
    xy = np.array([[0.0, 0.0], [50.0, -20.0], [3000.0, 3000.0]])
    grid = est.predict_grid(times, z, xy)
    pts = est.predict(Points(np.c_[xy, np.full(3, 10000.0), np.full(3, T0)], np.zeros((3, 2))))
    np.testing.assert_allclose(grid["du"].ravel(), pts.du, atol=1e-8)
    np.testing.assert_allclose(grid["var_u"].ravel(), pts.var_u, atol=1e-8)
    assert grid["var_u"].ravel()[2] == pytest.approx(est.prior_var)
