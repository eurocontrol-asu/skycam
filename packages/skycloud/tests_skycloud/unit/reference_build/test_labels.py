from __future__ import annotations

import numpy as np

from skycloud.reference_build.binning import BinMeans
from skycloud.reference_build.labels import CLEAR, OVERCAST, bin_centres, label_frames, node_of
from skycloud.reference_build.params import BinParams, LabelParams

# 4 angle-to-sun bins of 30 deg x 3 zenith bins of 20 deg: 9 bins beyond 15 deg of the sun
BINS = BinParams(gamma_bin_deg=30.0, n_gamma=4, zenith_bin_deg=20.0, n_zenith=3)
LAB = LabelParams(node_min_frames=4, node_min_bins=3)


def bin_means(ts: list[str], elev: list[float], n: list[float]) -> BinMeans:
    k = len(ts)
    mn = np.repeat(np.asarray(n, np.float16)[:, None], 12, 1)
    mn[:, :3] = np.nan  # the bins next to the sun are never read
    return BinMeans(
        ts=np.array(ts), sun=np.stack([np.zeros(k), 90 - np.asarray(elev)], 1),
        corner=np.zeros((k, 3), np.float32), MN=mn, ML=np.zeros_like(mn), MT=np.zeros_like(mn),
    )  # fmt: skip


def test_node_of_is_the_nearest_node_and_centres_are_mid_bin() -> None:
    assert node_of(np.array([3.4, 3.6, 70.0]), BINS.elevation_nodes).tolist() == [7, 8, 26]
    gc, zc = bin_centres(BINS)
    assert gc[:4].tolist() == [15.0, 15.0, 15.0, 45.0] and zc[:3].tolist() == [10.0, 30.0, 50.0]


def test_blue_frames_are_clear_grey_ones_overcast_and_low_sun_takes_its_episode() -> None:
    rng = np.random.default_rng(0)
    ts = [f"20240310{h:02d}{m:02d}00" for h in (8, 9) for m in range(0, 60, 6)]  # morning, 20
    elev = [30.0] * 20
    n = [0.20 + 0.002 * rng.standard_normal() if k % 2 else 0.0 for k in range(20)]
    n[4] = 0.12  # in between: neither
    # the next morning is clear at 6-20 deg (episode) and has two frames below 6 deg
    ts += [f"20240311{h:02d}0000" for h in (6, 7, 8, 9, 10)] + ["20240311050000", "20240311051000"]
    elev += [10.0, 12.0, 14.0, 16.0, 18.0, 3.0, 4.0]
    n += [0.20] * 5 + [0.0, 0.0]  # grey at low sun, yet the morning is clear
    lb = label_frames(bin_means(ts, elev, n), BINS, LAB)
    first = lb.lab[:20]
    assert (first[1::2] == CLEAR).all()
    assert (first[[0, 2, 6, 8]] == OVERCAST).all() and first[4] == 0
    assert lb.lab[-2:].tolist() == [0, 0]  # the 10-18 deg node has too few frames: no episode
    node = node_of(30.0, BINS.elevation_nodes)
    assert np.isclose(lb.p85[node, 3], np.percentile(np.asarray(n[:20], np.float16), 85), atol=1e-3)


def test_episode_label_needs_most_of_the_half_day() -> None:
    from skycloud.reference_build.labels import episode_labels

    ts = np.array(["20240311060000", "20240311070000", "20240311080000", "20240311090000",
                   "20240311050000", "20240311140000"])  # fmt: skip
    elev = np.array([8.0, 10.0, 12.0, 14.0, 3.0, 3.0])
    q50 = np.zeros(6)
    clear = np.array([True, True, True, True, False, False])
    lab, ep = episode_labels(ts, elev, q50, clear, ~clear, LAB)
    assert ep == {"20240311am": CLEAR} and lab.tolist() == [1, 1, 1, 1, 1, 0]
    clear[0] = False  # 3 of 4: under 80 %
    lab, ep = episode_labels(ts, elev, q50, clear, np.zeros(6, bool), LAB)
    assert ep == {"20240311am": 0} and lab[4] == 0
