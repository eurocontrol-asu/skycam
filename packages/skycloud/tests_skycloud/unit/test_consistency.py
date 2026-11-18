from __future__ import annotations

import numpy as np

from skycloud.consistency import _cap_mask, _smooth_d


def test_cap_where_value_switches_on_unchanged_sky() -> None:
    ok = np.ones((60, 60), bool)
    d = np.zeros((60, 60), np.float32)
    d[20:40, 20:40] = 0.5
    same = np.ones((60, 60), bool)
    cap = _cap_mask(ok, same, d, 0.15, 2.0)
    assert cap[30, 30] and not cap[5, 5]
    moved = same.copy()
    moved[15:45, 15:45] = False  # the sky changed there: no cap
    assert not _cap_mask(ok, moved, d, 0.15, 2.0)[30, 30]


def test_smooth_d_is_a_normalised_mean() -> None:
    ok = np.zeros((20, 20), bool)
    ok[:, :10] = True
    d = np.where(ok, 0.3, 9.0).astype(np.float32)
    s = _smooth_d(d, ok, 2.0)
    assert np.allclose(s[:, :10], 0.3, atol=1e-5)
