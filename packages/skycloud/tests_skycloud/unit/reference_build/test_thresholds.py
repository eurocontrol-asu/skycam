from __future__ import annotations

from typing import Any

import numpy as np

from skycloud.params import CloudMapParams
from skycloud.reference_build.features import DayFeatures
from skycloud.reference_build.params import BinParams, LowCountParams, ThresholdParams
from skycloud.reference_build.postfit import table_grid
from skycloud.reference_build.thresholds import (
    FrameStats,
    frame_stats,
    knowable,
    lift_colour,
    lowcount,
    midscale_frame,
    midscale_static,
    midscale_table,
    release_gate,
    shape_table,
)

TP, BINS = ThresholdParams(), BinParams()
SHAPE = (27, 91, 60)


def stats() -> FrameStats:
    rng = np.random.default_rng(0)
    k = 200
    elev = np.tile(np.linspace(0.5, 11.5, 50), 4)
    lab = np.repeat([1, 2, 1, 2], 50)
    clear = lab == 1
    return FrameStats(
        ts=[str(i) for i in range(k)], elev=elev, lab=lab,
        shape=np.where(clear, rng.uniform(0.1, 0.3, k), rng.uniform(0.6, 1.2, k)),
        support=np.where(clear, rng.uniform(0.7, 1.0, k), rng.uniform(0.1, 0.5, k)),
        opacity=np.where(clear, rng.uniform(-0.2, 0.2, k), rng.uniform(0.8, 1.2, k)),
    )  # fmt: skip


def test_shape_release_and_lift_tables_from_frame_statistics() -> None:
    st = stats()
    shape = shape_table(st, TP)
    assert shape["elev"] == [2.0, 4.0, 6.5, 10.0]  # bands with 10 clear and 10 overcast frames
    assert all(r0 < 0.3 < 0.6 < r1 for r0, r1 in zip(shape["R0"], shape["R1"], strict=True))
    gate = release_gate(st)
    assert gate["T_o"] == st.support[st.lab == 2].max()
    assert gate["T_c"] == np.median(st.support[st.lab == 1])
    lift = lift_colour(st, TP)
    assert lift["e"] == [1.0, 2.5, 3.5, 4.5, 5.5, 7.0, 10.0]
    assert all(c < 0.2 and o > 0.8 for c, o in zip(lift["clear_p95"], lift["ov_p5"], strict=True))
    assert all(round(v, 2) == v for v in lift["clear_p95"])


def test_knowable_finds_the_elevation_from_which_the_tables_separate() -> None:
    nodes = np.array(BINS.elevation_nodes, float)
    con = np.clip((nodes - 2.0) / 4.0, 0.0, 1.0)[:, None, None] * 0.1  # contrast from +2 to +6
    t: dict[str, Any] = {k: np.full(SHAPE, 0.004, np.float32) for k in (
        "Nc_sdf", "Lc_sda", "Nc_sd", "Lc_sd", "No_sd", "Lo_sd")}  # fmt: skip
    t.update(
        Nc=np.broadcast_to(0.1 + con, SHAPE).astype(np.float32),
        No=np.full(SHAPE, 0.1, np.float32), Lc=np.zeros(SHAPE, np.float32),
        Lo=np.zeros(SHAPE, np.float32), enodes=nodes, GB=2.0, ZB=1.5,
    )  # fmt: skip
    az = np.repeat(np.arange(0, 360, 2.0), 90)
    zen = np.tile(np.arange(90.0), 180)
    k = knowable(table_grid(t), az, zen, TP, CloudMapParams())
    assert 2.0 < k["E_030_supported"] <= k["E_030_unsupported"] <= k["E_K"] < 6.0


def test_frame_stats_read_each_labelled_frame_once() -> None:
    az = np.repeat(np.arange(0, 360, 2.0), 90)
    zen = np.tile(np.arange(90.0), 180)
    values = {"Nc": 0.2, "No": 0.0, "Lc": 0.5, "Lo": 1.0, "Nc_sda": 0.005, "Lc_sda": 0.02}
    t: dict[str, Any] = {k: np.full(SHAPE, v, np.float32) for k, v in values.items()}
    t.update(enodes=np.array(BINS.elevation_nodes, float), GB=2.0, ZB=1.5)
    f = np.zeros((az.size, 5), np.float32)
    f[:, 0], f[:, 1], f[:, 2] = 80.0, 0.545 * 10 * (1 + np.exp(0.5)), 120.0  # N = 0.2, L = 0.5
    day = DayFeatures(
        ts=np.array(["20240310080000", "20240310090000", "20240310100000"]),
        sun=np.array([[120.0, 85.0], [130.0, 80.0], [140.0, 75.0]]), feat=np.stack([f, f, f]),
        corner=np.tile(np.array([10.0, 5.0, 9.0], np.float32), (3, 1)),
        valid_n=np.full(az.size, 81, np.uint8),
    )  # fmt: skip
    st = frame_stats(
        [day], {"20240310080000": 1, "20240310100000": 2}, az, zen, table_grid(t), BINS, TP
    )
    assert st.ts == ["20240310080000", "20240310100000"] and st.lab.tolist() == [1, 2]
    assert np.allclose(st.opacity, 0.0, atol=1e-3)  # N equals the clear table
    # every direction on the clear colour and brightness; no support read above +12 deg
    assert st.support[0] == 1.0 and np.isnan(st.support[1])


def test_midscale_reads_the_band_pass_of_a_frame() -> None:
    rng = np.random.default_rng(4)
    n = 400
    img = np.full((n, n, 3), 100, np.uint8)
    img[:60, :60] = img[:60, -60:] = img[-60:, :60] = img[-60:, -60:] = 10
    noisy = img.copy()
    noisy[60:-60, 60:-60, 1] = rng.integers(95, 106, (n - 120, n - 120))
    zen = np.full((n, n), 40.0, np.float32)
    g = np.full((n, n), 90.0)
    sky = np.zeros((n, n), bool)
    sky[60:-60, 60:-60] = True
    static = midscale_static(sky, 4.0, TP)
    assert static.sum() < sky.sum()  # eroded by 3 deg (12 px)
    tp = ThresholdParams(midscale_min_pixels=100)
    flat = midscale_frame(img, 20.0, g, zen, static, 4.0, tp)
    rough = midscale_frame(noisy, 20.0, g, zen, static, 4.0, tp)
    assert flat is not None and rough is not None and flat[1] < rough[1]
    tab = midscale_table([rough, rough, rough, flat], TP)
    assert tab["elev"] == [27.5] and tab["sd"] == [rough[1]]


def test_lowcount_measures_the_excess_change_across_gain_steps() -> None:
    rng = np.random.default_rng(5)
    nd = 3000
    zen = np.full(nd, 30.0)
    pattern = rng.normal(0, 0.03, nd)  # the colour shift of the high gain, per direction
    gains = [1, 1, 2, 2] * 15  # pairs alternate: no step, step, no step, step...
    feats = []
    for gain in gains:
        f = np.zeros((nd, 5), np.float32)
        n = rng.normal(0, 0.003, nd) + (pattern if gain == 2 else 0.0)
        f[:, 0], f[:, 2], f[:, 1] = 50 * (1 - n), 50 * (1 + n), 80
        feats.append(f)
    day = DayFeatures(
        ts=np.array([f"20240310{10 + i // 6:02d}{i % 6 * 10:02d}00" for i in range(60)]),
        sun=np.tile([180.0, 50.0], (60, 1)), feat=np.stack(feats),
        corner=np.array([[20.0 * g, 10, 18] for g in gains], np.float32),
        valid_n=np.full(nd, 81, np.uint8),
    )  # fmt: skip
    out = lowcount([day], zen, BINS, LowCountParams(none_keep=1.0, min_samples=10))
    c = out["counts"]
    j = 5  # S = B + R = 100 counts: the bin 100-125
    assert c["q_step"][j] > 5 * c["q_none"][j] > 0
    assert np.isclose(c["e"][j], np.sqrt(c["q_step"][j] ** 2 - c["q_none"][j] ** 2))
    assert np.isnan(c["q_step"][0]) and c["e"][0] == 0.0  # no sample: no excess
    assert out["pedestal"]["S"].tolist()[:2] == [0.5, 1.25]
