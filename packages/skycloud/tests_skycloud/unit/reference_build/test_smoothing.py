from __future__ import annotations

from typing import Any

import numpy as np

from skycloud.reference_build.binning import BinMeans
from skycloud.reference_build.params import BinParams, SmoothParams
from skycloud.reference_build.smoothing import (
    adapted_spread,
    fill_nearest,
    rebuild_spreads,
    smooth_tables,
    weighted_smooth,
)

SP = SmoothParams()
BINS = BinParams(gamma_bin_deg=30.0, n_gamma=6, zenith_bin_deg=15.0, n_zenith=6)
SHAPE = (27, 6, 6)


def test_fill_nearest_fills_inside_a_node_then_from_the_nearest_node() -> None:
    s = np.full((3, 2, 3), np.nan)
    s[1, 0, 0], s[1, 1, 2] = 1.0, 2.0
    out = fill_nearest(s, np.isfinite(s))
    assert out[1, 0, 0] == 1.0 and out[1, 1, 2] == 2.0 and out[1, 1, 0] == 1.0
    assert np.isfinite(out).all() and (out[0] == out[1]).all() and (out[2] == out[1]).all()


def test_weighted_smooth_is_a_normalised_convolution() -> None:
    v = np.full((5, 5, 5), 3.0)
    w = np.zeros_like(v)
    w[2, 2, 2] = 10.0
    s = weighted_smooth(v, w, (1.0, 1.0, 1.0), 0.5)
    assert np.isclose(s[2, 2, 2], 3.0) and np.isnan(s[0, 0, 0])


def test_smooth_tables_keep_a_constant_and_fill_every_bin() -> None:
    nb = 36
    raw = {k: np.full((27, nb), np.nan, np.float32) for k in (
        "Nc", "Nc_sd", "No", "No_sd", "Lc", "Lc_sd", "Lo", "Lo_sd", "Tc", "To")}  # fmt: skip
    raw["n_c"] = np.zeros((27, nb), np.int32)
    raw["n_o"] = np.zeros((27, nb), np.int32)
    for k in raw:
        if not k.startswith("n_"):
            raw[k][10:15, :20] = 0.25
    raw["n_c"][10:15, :20] = 100
    raw["n_o"][10:15, :20] = 100
    out = smooth_tables(raw, BINS, SP)
    assert out["Nc"].shape == SHAPE and out["Nc"].dtype == np.float32
    assert np.allclose(out["Nc"], 0.25) and np.allclose(out["Lo_sd"], 0.25)
    assert out["n_c"].shape == SHAPE and out["GB"] == 30.0 and len(out["enodes"]) == 27


def test_adapted_spread_removes_the_frame_level() -> None:
    rng = np.random.default_rng(3)
    k, nb = 400, 36
    level = rng.normal(0, 0.05, k)[:, None]  # each frame's own level, large
    noise = rng.normal(0, 0.004, (k, nb))  # the clear sky's spread around it
    mn = (0.1 + level + noise).astype(np.float16)
    elev = np.full(k, 30.0)
    bm = BinMeans(
        ts=np.array([str(i) for i in range(k)]), sun=np.stack([np.zeros(k), 90 - elev], 1),
        corner=np.zeros((k, 3), np.float32), MN=mn, ML=mn, MT=mn,
    )  # fmt: skip
    smooth = {"Nc": np.full(SHAPE, 0.1, np.float32), "Lc": np.full(SHAPE, 0.1, np.float32)}
    out = adapted_spread(bm, np.ones(k, np.int8), smooth, BINS, SP)
    assert out["Nc_sda"].shape == SHAPE
    assert 0.003 < float(np.median(out["Nc_sda"])) < 0.005  # not the 0.05 of the levels


def test_rebuild_spreads_shrinks_noise_and_keeps_structure() -> None:
    s = np.full(SHAPE, 0.01, np.float32)
    s[13, 3, 3] = 0.011  # a 10 % blip on a bin of 4 frames: within its statistical error
    s[5:9] = 0.03  # a real step over whole nodes with many frames
    n = np.full(SHAPE, 4, np.int32)
    n[5:9] = 200
    ref: dict[str, Any] = dict.fromkeys(("Nc_sdf", "Nc_sd", "Lc_sda", "Lc_sd", "No_sd", "Lo_sd"), s)
    ref.update(n_c=n, n_o=n, enodes=np.arange(27.0), GB=2.0, ZB=1.5)
    out = rebuild_spreads(ref, SP)
    assert abs(out["Nc_sd"][13, 3, 3] - 0.01) < 0.0005
    assert float(out["Nc_sd"][7, 3, 3]) > 0.027  # the heavy smooth alone reads about 0.02
    assert set(out) == {
        "enodes",
        "GB",
        "ZB",
        "Nc_sdf",
        "Nc_sd",
        "Lc_sda",
        "Lc_sd",
        "No_sd",
        "Lo_sd",
    }
