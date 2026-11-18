from __future__ import annotations

import numpy as np

from skycloud.cues import cues
from skycloud.reference_build.binning import BinMeans, day_bin_means
from skycloud.reference_build.features import DayFeatures
from skycloud.reference_build.params import BinParams

P = BinParams()


def day(feat: np.ndarray, valid_n: np.ndarray) -> DayFeatures:
    return DayFeatures(
        ts=np.array(["20240101120000"]),
        sun=np.array([[0.0, 0.0]]),  # sun at the zenith: the angle to the sun is the zenith angle
        feat=feat[None].astype(np.float32),
        corner=np.array([[20.0, 11.0, 18.0]], np.float32),
        valid_n=valid_n,
    )


def test_bin_means_average_two_directions_and_drop_lone_or_saturated_ones() -> None:
    az = np.array([0.0, 180.0, 90.0, 0.0])
    zen = np.array([10.2, 10.4, 40.0, 10.3])
    feat = np.array(
        [[50, 70, 90, 0, 1.0], [60, 80, 120, 0, 2.0], [50, 70, 90, 0, 1.0], [60, 80, 120, 0.1, 1.0]]
    )  # the fourth is saturated
    mn, ml, mt = day_bin_means(day(feat, np.full(4, 81, np.uint8)), az, zen, P)
    key = 5 * P.n_zenith + 6  # angle-to-sun bin 5 (10-12 deg), zenith bin 6 (9-10.5 deg)
    n, lum = cues(feat[:2, 0], feat[:2, 1], feat[:2, 2], np.array([20.0, 11.0, 18.0]))
    assert np.isclose(float(mn[0, key]), n.mean(), atol=1e-3)
    assert np.isclose(float(ml[0, key]), lum.mean(), atol=2e-3)
    assert np.isclose(float(mt[0, key]), (1 / 59 + 2 / 69) / 2, atol=1e-4)
    assert np.isfinite(mn[0]).sum() == 1  # the lone direction at 40 deg makes no bin
    assert mn.dtype == np.float16


def test_bin_means_skip_windows_with_too_little_sky() -> None:
    az, zen = np.array([0.0, 180.0]), np.array([10.2, 10.4])
    feat = np.array([[50, 70, 90, 0, 1.0], [60, 80, 120, 0, 2.0]])
    mn, _, _ = day_bin_means(day(feat, np.array([81, 69], np.uint8)), az, zen, P)
    assert not np.isfinite(mn).any()


def test_bin_means_from_arrays_gives_the_sun_elevation() -> None:
    d = {
        "ts": np.array(["a", "b"]), "sun": np.array([[0.0, 80.0], [0.0, 30.0]]),
        "corner": np.zeros((2, 3), np.float32), "MN": np.zeros((2, 3)), "ML": np.zeros((2, 3)),
        "MT": np.zeros((2, 3)),
    }  # fmt: skip
    assert BinMeans.from_arrays(d).elev.tolist() == [10.0, 60.0]
