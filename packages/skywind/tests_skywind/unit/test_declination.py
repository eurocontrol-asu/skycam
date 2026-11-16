from __future__ import annotations

from datetime import datetime

import numpy as np

from skywind.declination import declination_deg


def test_declination_near_paris_is_small_and_east() -> None:
    d = declination_deg([48.6], [2.35], 11.0, datetime(2024, 3, 18))
    assert 1.0 < d[0] < 3.5  # IGRF: about +2 deg east around Paris in 2024


def test_declination_varies_with_longitude_and_handles_empty() -> None:
    d = declination_deg([48.6, 48.6], [-2.0, 7.0], 11.0, datetime(2024, 3, 18))
    assert d[1] > d[0]
    assert declination_deg(np.array([]), np.array([]), 11.0, datetime(2024, 1, 1)).size == 0
