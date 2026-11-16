from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from skywind.field.data import CROSS, ObsSet
from skywind.field.gp import GPParams
from skywind.field.hyper import (
    block_loglik,
    free_parameters,
    make_blocks,
    params_from_vector,
    vector_from_params,
)
from tests.unit.field.test_gp import tracks


@pytest.mark.parametrize("kernel", ["gaussian", "matern32"])
@pytest.mark.parametrize("cross", [False, True])
def test_loglik_gradient_matches_finite_differences(kernel: str, cross: bool) -> None:
    obs = tracks([0.0, 90.0], n_tracks=2)
    if cross:
        c = obs.subset(np.arange(0, len(obs), 3))
        obs = obs.concat(
            ObsSet(c.X, c.e[:, ::-1] * [1, -1], c.y, np.full(len(c), CROSS), c.aircraft, c.bg, c.day, c.icao24)
        )
    obs = ObsSet(
        obs.X
        + np.c_[
            np.zeros((len(obs), 2)), np.linspace(0, 300, len(obs))[:, None], np.linspace(0, 600, len(obs))[:, None]
        ],
        obs.e,
        obs.y,
        obs.kind,
        obs.aircraft,
        obs.bg,
        obs.day,
        obs.icao24,
    )
    p = GPParams(
        length_h_km=50,
        length_z_m=250,
        length_t_min=8,
        sigma_field=2.0,
        sigma_bias=0.6,
        mean_bias=0.2,
        sigma_noise=0.8,
        kernel=kernel,
    )  # type: ignore[arg-type]
    names = free_parameters(p, cross)
    _, grad = block_loglik(obs, p, names)
    x = vector_from_params(p, names)
    for j in range(len(names)):
        h = np.zeros_like(x)
        h[j] = 1e-5
        up = block_loglik(obs, params_from_vector(p, names, x + h), names)[0]
        down = block_loglik(obs, params_from_vector(p, names, x - h), names)[0]
        assert grad[j] == pytest.approx((up - down) / 2e-5, rel=1e-4, abs=1e-4)


def test_vector_round_trip() -> None:
    p = GPParams(mean_bias=-0.4)
    names = free_parameters(p, True)
    q = params_from_vector(p, names, vector_from_params(p, names))
    assert q.length_h_km == pytest.approx(p.length_h_km) and q.mean_bias == pytest.approx(-0.4)


def test_no_bias_parameters_when_bias_disabled() -> None:
    names = free_parameters(replace(GPParams(), use_bias=False), False)
    assert "log_sigma_bias" not in names and "mean_bias" not in names


def test_blocks_are_disjoint_in_time_and_thinned() -> None:
    obs = tracks([0.0, 90.0])
    obs = ObsSet(
        obs.X + np.c_[np.zeros((len(obs), 3)), np.arange(len(obs))[:, None] * 10.0],
        obs.e,
        obs.y,
        obs.kind,
        obs.aircraft,
        obs.bg,
        obs.day,
        obs.icao24,
    )
    blocks = make_blocks(obs, block_minutes=10, max_size=50)
    assert all(len(b) <= 50 for b in blocks)
    spans = [(b.X[:, 3].min(), b.X[:, 3].max()) for b in blocks]
    assert all(s1[1] < s2[0] for s1, s2 in itertools.pairwise(spans))


def test_two_scale_gradient() -> None:
    obs = tracks([0.0, 90.0], n_tracks=2)
    obs = ObsSet(
        obs.X
        + np.c_[
            np.zeros((len(obs), 2)), np.linspace(0, 300, len(obs))[:, None], np.linspace(0, 3600, len(obs))[:, None]
        ],
        obs.e,
        obs.y,
        obs.kind,
        obs.aircraft,
        obs.bg,
        obs.day,
        obs.icao24,
    )
    p = GPParams(
        length_h_km=30,
        length_z_m=250,
        length_t_min=20,
        sigma_field=1.5,
        sigma_bias=0.6,
        mean_bias=0.1,
        sigma_noise=0.7,
        kernel="matern32",
        sigma_field2=1.2,
        length_h2_km=200,
        length_t2_min=120,
    )
    names = free_parameters(p, False)
    assert {"log_sigma_field2", "log_length_h2_km", "log_length_t2_min"} <= set(names)
    _, grad = block_loglik(obs, p, names)
    x = vector_from_params(p, names)
    for j in range(len(names)):
        h = np.zeros_like(x)
        h[j] = 1e-5
        up = block_loglik(obs, params_from_vector(p, names, x + h), names)[0]
        down = block_loglik(obs, params_from_vector(p, names, x - h), names)[0]
        assert grad[j] == pytest.approx((up - down) / 2e-5, rel=1e-4, abs=1e-4)
