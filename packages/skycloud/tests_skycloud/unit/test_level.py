from __future__ import annotations

import numpy as np

from skycloud.cues import level_basis
from skycloud.level import fit_level, shape_support_joint, smooth_states
from skycloud.params import CloudMapParams

P = CloudMapParams()


def test_joint_fit_finds_the_clear_level_and_ignores_an_overcast_minority() -> None:
    rng = np.random.default_rng(1)
    g = rng.uniform(15, 170, 3000)
    z = rng.uniform(0, 85, 3000)
    x = level_basis(g, z)
    r = 0.01 + rng.normal(0, 0.002, 3000)  # clear sky 0.01 bluer than the table
    rl = 0.05 + rng.normal(0, 0.01, 3000)
    r[:600] -= 0.2  # an overcast fifth: far from the clear colour and brightness
    rl[:600] -= 0.4
    beta, al, w = fit_level(r, x, P, rl, np.full(3000, 0.03))
    assert abs(float((x[600:] @ beta).mean()) - 0.01) < 0.002  # the fitted surface on the clear sky
    assert abs(al - 0.05) < 0.01
    assert w[:600].mean() < 0.01 < w[600:].mean()


def test_shape_support_joint_finds_the_common_offsets() -> None:
    rng = np.random.default_rng(2)
    dn = 0.02 + rng.normal(0, 0.001, 500)
    dl = -0.1 + rng.normal(0, 0.005, 500)
    f, an, al = shape_support_joint(dn, np.full(500, 0.005), dl, np.full(500, 0.02))
    assert f > 0.95 and abs(an - 0.02) < 0.01 and abs(al + 0.1) < 0.02
    assert shape_support_joint(dn[:10], dn[:10], dl[:10], dl[:10]) == (0.0, 0.0, 0.0)


def _state(beta0: float, f: float, elev: float = 20.0, n: int = 1000) -> dict[str, object]:
    return {
        "elev": elev,
        "aN": beta0,
        "fN": f,
        "aL": 0.0,
        "fL": f,
        "beta": [beta0, 0.0, 0.0, 0.0],
        "n": n,
        "pedR": 10.0,
    }


def test_smooth_states_weights_frames_by_support_and_time() -> None:
    states = {
        "20240101120000": _state(0.02, 0.9),
        "20240101120200": _state(0.00, 0.0, n=0),  # unreadable colour: imports the level
        "20240101180000": _state(-0.5, 0.9),  # 6 h away: no weight
    }
    out = smooth_states(states, ["20240101120200"], P)
    lv = out["20240101120200"]
    assert np.isclose(
        lv["aN"],
        0.02 * 0.9 * np.exp(-0.5 * (2 / 8) ** 2) / (0.9 * np.exp(-0.5 * (2 / 8) ** 2) + P.t_prior),
        rtol=1e-3,
    )
    assert lv["gate"] == 1.0


def test_own_support_gate_stops_an_overcast_importing_the_level() -> None:
    states = {
        "20240101120000": _state(0.02, 0.9, elev=20.0),
        "20240101120030": _state(0.0, 0.0, elev=20.0),
    }
    lv = smooth_states(states, ["20240101120030"], P)["20240101120030"]
    assert lv["gate"] == 0.0 and lv["aN"] == 0.0


def test_daytime_targets_read_the_legacy_levels() -> None:
    s = _state(0.02, 0.9, elev=11.0)
    s["legacy"] = {"aN": 0.05, "beta": [0.05, 0.0, 0.0, 0.0], "fN": 0.9, "aL": 0.0, "fL": 0.9}
    out = smooth_states({"20240101120000": s}, ["20240101120000"], P)
    assert out["20240101120000"]["aN"] > 0.04
