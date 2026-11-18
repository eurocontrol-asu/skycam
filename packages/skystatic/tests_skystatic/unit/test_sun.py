from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from skystatic.geometry import SkyGeometry
from skystatic.sun import sun_clearance, sun_position

PARIS = (48.6005, 2.3468)


def test_sun_position_at_solar_noon_and_in_the_morning() -> None:
    # 2023-10-01, Bretigny: solar noon ~11:40 UTC, elevation ~38 degrees, azimuth ~180.
    zenith, azimuth = sun_position(datetime(2023, 10, 1, 11, 40), *PARIS)
    assert zenith == pytest.approx(51.5, abs=1)
    assert azimuth == pytest.approx(180, abs=3)
    _, morning = sun_position(datetime(2023, 10, 1, 7, 0), *PARIS)
    assert 90 < morning < 120  # east-south-east


def test_clearance_is_zero_where_the_sun_is() -> None:
    zenith = np.array([[51.5, 51.5], [10.0, 80.0]], dtype=np.float32)
    azimuth = np.array([[180.0, 0.0], [0.0, 90.0]], dtype=np.float32)
    geometry = SkyGeometry.from_array(zenith).with_sun(azimuth, *PARIS)
    span = (datetime(2023, 10, 1, 11, 35), datetime(2023, 10, 1, 11, 45))
    clearance = sun_clearance(geometry, [span])[0]
    assert clearance[0, 0] < 2
    assert clearance[0, 1] > 90  # the opposite side of the sky


def test_clearance_needs_azimuths_and_a_site() -> None:
    with pytest.raises(ValueError, match="azimuth"):
        sun_clearance(SkyGeometry.from_array(np.zeros((2, 2))), [])
