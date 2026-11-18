from __future__ import annotations

import numpy as np

from skycloud.structure import certification, cloud_objects, lobe_filtered, shape_clear
from tests_skycloud.unit.test_geometry import fisheye


def test_cloud_objects_certifies_a_bounded_body_only() -> None:
    lg = np.zeros((200, 200), np.float32)
    rng = np.random.default_rng(0)
    lg += rng.normal(0, 0.002, lg.shape).astype(np.float32)
    lg[80:120, 80:120] += 0.5  # a bright body with a sharp edge all round
    alpha = np.zeros_like(lg)
    alpha[80:120, 80:120] = 0.9
    info: dict[str, object] = {}
    obj = cloud_objects(lg, alpha, np.ones(lg.shape, bool), 0.2, 0.5, info)
    assert obj[100, 100] == 1.0 and obj[20, 20] == 0.0
    smooth = np.zeros_like(lg) + rng.normal(0, 0.002, lg.shape).astype(np.float32)
    assert cloud_objects(smooth, alpha, np.ones(lg.shape, bool), 0.2, 0.5, {})[100, 100] == 0.0


def test_lobe_filter_drops_the_opposite_sign_beside_a_line() -> None:
    bpm = np.zeros((40, 40), np.float32)
    bpm[:, 20] = 1.0
    bpm[:, 17] = -0.3  # side lobe
    a_m = np.ones_like(bpm)
    out = lobe_filtered(a_m, bpm, 2.0)
    assert out[10, 20] == 1.0 and out[10, 17] == 0.0


def test_certification_levels() -> None:
    shape = (60, 60)
    valid = np.ones(shape, bool)
    stru = np.zeros(shape, np.float32)
    a_t = np.zeros(shape, np.float32)
    alpha = np.zeros(shape, np.float32)
    stru[10:20, 10:20] = 1.0
    alpha[10:20, 10:20] = 0.8
    a_t[40:45, 40:45] = 0.6  # weak line evidence with a value above the read-clear level
    alpha[40:45, 40:45] = 0.3
    cert, fld, strong = certification(
        stru, a_t, np.zeros(shape, np.float32), alpha, alpha, valid, 5.0, 0.2, 0.3
    )
    assert np.isclose(cert[15, 15], 1.0) and strong[15, 15]
    assert cert[42, 42] == 0.5 and not fld.any()
    own_clear = certification(
        stru,
        a_t,
        np.zeros(shape, np.float32),
        alpha,
        np.zeros(shape, np.float32),
        valid,
        5.0,
        0.2,
        0.3,
    )[0]
    assert own_clear.max() == 0.0  # the pixel's own measured value says clear: never certified


def test_shape_clear_maps_the_ratio_between_the_archive_bounds() -> None:
    geo = fisheye(201)
    rng = np.random.default_rng(3)
    lum = rng.normal(0, 0.01, geo.shape)
    lc = np.zeros(geo.shape)
    lo = lum + rng.normal(0, 0.1, geo.shape)  # clear-shaped: far from the overcast table
    tab = {"elev": [0.0, 90.0], "R0": [0.2, 0.2], "R1": [0.6, 0.6]}
    s, ratio = shape_clear(geo, lum, lc, lo, geo.inside, (0.0, 70.0), {}, tab)
    assert ratio is not None and ratio < 0.2 and s == 1.0
    s2, _ = shape_clear(geo, lum, lc, lo, geo.inside, (0.0, 85.0), {"ratio_s": 0.7}, tab)
    assert s2 == 0.0  # below +10 deg the time-smoothed ratio replaces it
