from __future__ import annotations

import numpy as np

from skycloud.reference_build.binning import BinMeans
from skycloud.reference_build.params import BinParams
from skycloud.reference_build.raw_tables import column_percentiles, raw_tables, robust_median_sd

BINS = BinParams(gamma_bin_deg=90.0, n_gamma=2, zenith_bin_deg=90.0, n_zenith=1)


def test_robust_median_sd_keeps_the_input_precision() -> None:
    m = np.arange(20, dtype=np.float32).reshape(10, 2)
    m[0, 1] = np.nan
    med, sd = robust_median_sd(m)
    assert med.dtype == np.float32 and med[0] == 9.0
    assert np.isclose(sd[0], (np.percentile(m[:, 0], 84) - np.percentile(m[:, 0], 16)) / 2)
    assert column_percentiles(m, [50]).dtype == np.float32


def test_raw_tables_accumulate_clear_and_overcast_frames_per_node() -> None:
    k = 12
    elev = np.r_[np.full(10, 30.0), np.full(2, 30.0)]
    mn = np.zeros((k, 2), np.float16)
    mn[:10, 0] = np.arange(10) / 100  # ten clear frames
    mn[10:, 0] = 0.5  # two overcast frames: fewer than min_frames
    mn[:, 1] = np.nan
    bm = BinMeans(
        ts=np.array([str(i) for i in range(k)]), sun=np.stack([np.zeros(k), 90 - elev], 1),
        corner=np.zeros((k, 3), np.float32), MN=mn, ML=mn * 2, MT=mn,
    )  # fmt: skip
    lab = np.r_[np.ones(10, np.int8), np.full(2, 2, np.int8)]
    out = raw_tables(bm, lab, BINS, min_frames=8)
    c = 19  # node 30 deg
    assert np.isclose(out["Nc"][c, 0], np.median(mn[:10, 0].astype(np.float32)))
    assert np.isclose(out["Lc"][c, 0], 2 * out["Nc"][c, 0], atol=1e-3)
    assert out["Nc_sd"][c, 0] > 0 and np.isnan(out["Nc"][c, 1])
    assert out["n_c"][c].tolist() == [10, 0] and out["elev_c"][c, 0] == 30.0
    assert np.isnan(out["No"]).all() and (out["n_o"] == 0).all()
    assert np.isnan(np.delete(out["Nc"], c, 0)).all()
