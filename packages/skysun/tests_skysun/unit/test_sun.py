"""The sun's position against published reference values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from skysun import Site, sun_position


def test_spa_reference_example() -> None:
    """Reda & Andreas (2004), Solar Position Algorithm for Solar Radiation Applications,
    NREL/TP-560-34302, table A5.1: Golden (Colorado), 2003-10-17 12:30:30 local time (UTC-7):
    topocentric zenith 50.11162 deg, azimuth 194.34024 deg (NOAA vs SPA: about 0.01 deg)."""
    site = Site(latitude_deg=39.742476, longitude_deg=-105.1786)
    local = datetime(2003, 10, 17, 12, 30, 30, tzinfo=timezone(timedelta(hours=-7)))
    sun = sun_position(local, site)
    assert sun.zenith_deg == pytest.approx(50.11162, abs=0.03)
    assert sun.azimuth_deg == pytest.approx(194.34024, abs=0.03)


def test_solstice_noon_at_the_site() -> None:
    """At true solar noon of the June solstice the sun is due south at 90 - (lat - 23.44)."""
    site = Site(latitude_deg=48.6005, longitude_deg=2.3468)
    # 2024-06-20 20:51 UTC solstice; solar noon at 2.35 E is ~11:52 UTC (equation of time -1.7 min)
    times = [datetime(2024, 6, 20, 11, 40, tzinfo=UTC) + timedelta(minutes=m) for m in range(25)]
    highest = max((sun_position(t, site) for t in times), key=lambda s: s.elevation_deg)
    assert highest.elevation_deg == pytest.approx(90 - (48.6005 - 23.44), abs=0.05)
    assert highest.azimuth_deg == pytest.approx(180, abs=0.6)


def test_night_and_morning_sides() -> None:
    site = Site(latitude_deg=48.6005, longitude_deg=2.3468)
    assert sun_position(datetime(2024, 1, 15, 2, tzinfo=UTC), site).elevation_deg < -30
    morning = sun_position(datetime(2024, 3, 18, 8, tzinfo=UTC), site)
    assert 90 < morning.azimuth_deg < 180
    assert 10 < morning.elevation_deg < 30


def test_naive_times_are_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        sun_position(datetime(2024, 1, 1, 12), Site(latitude_deg=48.6, longitude_deg=2.3))
