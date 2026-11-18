from __future__ import annotations

import numpy as np

from skycloud.cues import (
    PED_GR,
    clear_baseline,
    colour_opacity,
    cues,
    fill_low_info,
    fuse,
    guided,
    level_basis,
    mad,
    ped_g,
    texture,
)


def test_cues_colour_index_and_brightness_in_pedestal_units() -> None:
    ped = np.array([10.0, 99.0, 99.0])
    n, lum = cues(np.array(40.0), np.array(20.0 * PED_GR + 10 * PED_GR), np.array(60.0), ped)
    assert np.isclose(n, (60 - 40) / 100)
    # G - pedestal = 20 pedestal units of G above it -> L = log(20 PED_GR / (10 PED_GR)) = log 2
    assert np.isclose(lum, np.log(2.0))
    assert np.isclose(ped_g(ped), PED_GR * 10.0)


def test_mad_is_robust_sd() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    assert np.isclose(mad(x), 1.4826)
    assert np.isnan(mad(np.array([])))


def test_level_basis_terms() -> None:
    b = level_basis(np.array([0.0, 25.0]), np.array([90.0, 82.0]))
    assert b.shape == (2, 4)
    assert np.allclose(b[0], [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(b[1], [1.0, np.exp(-1.0), np.exp(-25 / 70), np.exp(-1.0)])


def test_guided_filter_keeps_flat_field_and_follows_an_edge() -> None:
    guide = np.where(np.arange(40)[None, :] < 20, 50.0, 150.0).repeat(40, 0).astype(np.float32)
    src = (guide * 0.5).astype(np.float32)
    w = np.ones_like(guide)
    out_src, out_guide = guided(guide, [src, guide], w, 3, 1.0)
    assert np.allclose(out_src[:, 5], 25.0, atol=1e-3) and np.allclose(
        out_src[:, 35], 75.0, atol=1e-3
    )
    assert np.allclose(out_guide[:, 19], 50.0, atol=0.5)  # the edge is kept, not blurred


def test_texture_is_zero_on_flat_sky_and_positive_on_a_line() -> None:
    lg = np.zeros((64, 64), np.float32)
    w = np.ones_like(lg)
    assert np.allclose(texture(lg, w), 0.0)
    lg[:, 32] = 1.0
    assert texture(lg, w)[32, 32] > 0.05


def test_colour_opacity_shrinks_small_departures() -> None:
    ab = np.array([0.05, 1.0], np.float32)
    a = colour_opacity(ab, np.array([0.1, 0.1], np.float32), np.zeros(2, np.float32), 1.0)
    assert a[0] < 0.25 * ab[0]  # within the clear spread: shrunk
    assert np.isclose(a[1], 1.0 / (1.0 + 0.01), rtol=1e-6)


def test_clear_baseline_is_negative_only() -> None:
    ab = np.full((128, 128), -0.2, np.float32)
    base = clear_baseline(ab, np.ones(ab.shape, bool), 10.0, 3.0, 25)
    assert base.max() <= 0.0 and base[64, 64] < -0.1
    pos = clear_baseline(
        np.full((128, 128), 0.5, np.float32), np.ones((128, 128), bool), 10.0, 3.0, 25
    )
    assert np.all(pos == 0.0)  # an overcast is never absorbed


def test_fill_low_info_takes_value_from_informative_surroundings() -> None:
    alpha = np.zeros((128, 128), np.float32)
    w = np.full_like(alpha, 50.0)
    alpha[:, :] = 0.8
    alpha[60:68, 60:68] = 0.0
    w[60:68, 60:68] = 0.0  # no information there
    out = fill_low_info(alpha, w, np.ones(alpha.shape, bool), 6.0)
    assert out[64, 64] > 0.7
    assert np.isclose(out[10, 10], 0.8, atol=1e-3)


def test_fuse_weights_cues_by_their_verdict_precision() -> None:
    one = np.ones(1)
    a, k = fuse(
        1.0 * one, 0.0 * one, 0.0 * one, 100.0 * one, 1.0 * one, 0.0 * one, 0.0 * one, 0.0 * one
    )
    assert np.isclose(a[0], 1.0, atol=1e-6) and np.isclose(k[0], 100.0)
    a, _ = fuse(
        0.0 * one, 1.0 * one, 0.0 * one, 1.0 * one, 1.0 * one, 3.0 * one, 1.0 * one, 0.0 * one
    )
    assert np.isclose(
        a[0], 0.75, atol=1e-6
    )  # colour clear (weight 1) vs brightness cloud (weight 3)
