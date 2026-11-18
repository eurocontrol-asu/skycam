from __future__ import annotations

import numpy as np

from skycloud.params import CloudMapParams
from skycloud.reference_build.features import DayFeatures
from skycloud.reference_build.params import BinParams, SmoothParams
from skycloud.reference_build.postfit import frame_residual_bins, postfit_tables, table_grid

BINS, SP, CP = BinParams(), SmoothParams(), CloudMapParams()
SHAPE = (27, 91, 60)


def tables(nc: float) -> dict[str, object]:
    return {
        "Nc": np.full(SHAPE, nc, np.float32), "No": np.full(SHAPE, 0.0, np.float32),
        "Lc": np.full(SHAPE, 1.0, np.float32), "Lc_sd": np.full(SHAPE, 0.05, np.float32),
        "enodes": np.array(BINS.elevation_nodes, float), "GB": 2.0, "ZB": 1.5,
    }  # fmt: skip


def frame(n_offset: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.repeat(np.arange(0, 360, 2.0), 90)
    zen = np.tile(np.arange(90.0), 180)
    # R, B with N = (B - R) / (B + R) = 0.2 + offset; G gives L = 1 in pedestal units
    nn = 0.2 + n_offset + rng.normal(0, 0.002, az.size)
    s = 200.0
    f = np.zeros((az.size, 5), np.float32)
    f[:, 0], f[:, 2] = s * (1 - nn) / 2, s * (1 + nn) / 2
    f[:, 1] = 0.545 * 10 * (1 + np.e)
    return f, az, zen


def test_the_level_fit_absorbs_a_uniform_offset_of_a_clear_frame() -> None:
    rng = np.random.default_rng(0)
    f, az, zen = frame(0.006, rng)
    ref = table_grid(tables(0.2))
    sun = np.array([180.0, 50.0])
    cor = np.array([10.0, 5.0, 9.0], np.float32)
    vn = np.full(az.size, 81, np.uint8)
    res = frame_residual_bins(f, cor, sun, vn, az, zen, ref, BINS, SP, CP).astype(np.float32)
    assert res.dtype == np.float32 and np.isfinite(res).sum() > 1000
    assert abs(float(np.nanmedian(res))) < 0.002  # most of the 0.006 is the frame's level
    f0, *_ = frame(0.006, rng)
    few = vn.copy()
    few[:] = 0  # no readable direction: no level fitted, no bin
    assert not np.isfinite(frame_residual_bins(f0, cor, sun, few, az, zen, ref, BINS, SP, CP)).any()


def test_postfit_tables_read_only_the_clear_frames() -> None:
    rng = np.random.default_rng(1)
    rows = [frame(0.0, rng)[0] for _ in range(10)]
    _, az, zen = frame(0.0, rng)
    day = DayFeatures(
        ts=np.array([f"2024031012{m:02d}00" for m in range(10)]),
        sun=np.tile([180.0, 50.0], (10, 1)), feat=np.stack(rows),
        corner=np.tile(np.array([10.0, 5.0, 9.0], np.float32), (10, 1)),
        valid_n=np.full(az.size, 81, np.uint8),
    )  # fmt: skip
    labels = {str(t): 1 for t in day.ts[:9]}  # the tenth frame is not clear
    out, n = postfit_tables([day], labels, az, zen, tables(0.2), BINS, SP, CP)
    assert n == 9
    assert out["Nc_bias"].shape == SHAPE and np.isfinite(out["Nc_sdf"]).all()
    assert abs(float(np.median(out["Nc_bias"][16]))) < 0.002
    # the 0.002 noise per direction, averaged over the directions of a bin
    assert 0.0001 < float(np.median(out["Nc_sdf"][16])) < 0.002
