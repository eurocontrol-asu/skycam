from __future__ import annotations

import numpy as np

from skycloud.geometry import CameraGeometry, Site, angle_to_sun, to_vectors


def fisheye(size: int) -> CameraGeometry:
    """Equidistant fisheye: zenith 0 at the centre, 90 deg at radius size / 2; north up."""
    y, x = np.indices((size, size), dtype=np.float64)
    c = (size - 1) / 2
    r = np.hypot(x - c, y - c)
    zen = r / (size / 2) * 90.0
    az = np.degrees(np.arctan2(x - c, -(y - c))) % 360
    zen[zen > 90] = np.nan
    return CameraGeometry(az, zen)


def test_derived_maps() -> None:
    geo = fisheye(101)
    assert geo.inside[50, 50] and not geo.inside[0, 0]
    assert np.isclose(geo.px_deg, 101 / 2 / 90, rtol=0.02)
    assert tuple(geo.centre) == (50.0, 50.0)
    assert geo.zen90[0, 0] == 90.0 and geo.zen99[0, 0] == 99.0
    assert geo.az4.shape == (26, 26)
    assert np.allclose(np.linalg.norm(geo.vec, axis=-1), 1.0, atol=1e-6)


def test_invalid_directions_are_nan_in_both_maps() -> None:
    zen = np.array([[10.0, np.nan]])
    az = np.array([[np.inf, 5.0]])
    geo = CameraGeometry(az, zen)
    assert np.isnan(geo.zenith_deg).all() and np.isnan(geo.azimuth_deg).all()


def test_angle_to_sun_and_vectors() -> None:
    assert np.isclose(angle_to_sun(0.0, 10.0, 180.0, 10.0), 20.0)
    assert np.allclose(to_vectors(90.0, 90.0), [1.0, 0.0, 0.0], atol=1e-12)
    assert Site(latitude_deg=48.6, longitude_deg=2.35).latitude_deg == 48.6
