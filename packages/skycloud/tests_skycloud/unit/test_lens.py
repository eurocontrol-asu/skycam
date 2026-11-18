from __future__ import annotations

import numpy as np

from skycloud.lens import _coherent, _ring_profile, lens_flags


def test_coherent_keeps_rings_all_quadrants_agree_on() -> None:
    hp = np.array(
        [[0.01, 0.02, -0.01], [0.02, 0.01, 0.01], [np.nan, 0.03, -0.02], [0.01, 0.05, -0.01]]
    )
    out = _coherent(hp, 2)
    assert np.allclose(out, [0.01, 0.01, 0.0])


def test_ring_profile_finds_a_concentric_stripe() -> None:
    yy, xx = np.mgrid[0:400, 0:400].astype(np.float64)
    r = np.hypot(xx - 200, yy - 200)
    d = 0.003 * np.cos(2 * np.pi * r / 40.0)
    hp = _ring_profile(d.ravel(), xx.ravel(), yy.ravel(), (200.0, 200.0), 300, 2.0, 15)
    assert np.nanmax(np.abs(hp[:, 20:80])) > 0.001
    assert np.abs(_coherent(hp, 2)[20:80]).max() > 0.001


def test_lens_flags_without_visible_sun_has_no_ghost() -> None:
    img = np.full((60, 60, 3), 120, np.uint8)
    valid = np.ones((60, 60), bool)
    vec = np.zeros((60, 60, 3), np.float32)
    vec[..., 2] = 1.0
    res = lens_flags(
        img,
        np.zeros((60, 60), np.float32),
        valid,
        np.full((60, 60), 50, np.float32),
        vec,
        (30.0, 30.0),
        (10, 10),
        False,
        ~valid,
    )
    assert not res.ghosts.any() and res.ring_bins == 0
