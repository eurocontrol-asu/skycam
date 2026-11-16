from __future__ import annotations

import numpy as np
import pytest

from skywind.gvccs.camera import Camera


def _camera() -> Camera:
    # synthetic planar sky: x = 512 + 400 tan(z) sin(az), y = 512 + 400 tan(z) cos(az) in the fitted convention
    yy, xx = np.indices((1024, 1024)).astype(float)
    dx, dy = (xx - 512) / 400, (yy - 512) / 400
    az = np.arctan2(dx, dy) - np.pi
    zen = np.arctan(np.hypot(dx, dy))
    grid = np.stack([az, zen], -1)
    A = np.stack([np.tan(zen) * np.sin(az + np.pi), np.tan(zen) * np.cos(az + np.pi), np.ones_like(zen)], -1)
    coef = np.linalg.lstsq(A[::8, ::8].reshape(-1, 3), np.stack([xx, yy], -1)[::8, ::8].reshape(-1, 2), rcond=None)[0]
    return Camera(48.6, 2.35, 90.0, grid, coef, x_offset=0.0)


def test_zenith_point_and_round_trip() -> None:
    cam = _camera()
    _, zen = cam.az_zen(np.array([48.6]), np.array([2.35]), np.array([11000.0]))
    assert zen[0] == pytest.approx(0.0, abs=1e-6)
    x, y = cam.pixel(np.array([0.7]), np.array([0.5]))
    a2, z2 = cam.direction(x, y)
    assert z2[0] == pytest.approx(0.5, abs=5e-3)
    assert ((a2[0] - 0.7 + np.pi) % (2 * np.pi) - np.pi) == pytest.approx(0.0, abs=5e-3)


def test_azimuth_is_clockwise_from_north() -> None:
    cam = _camera()
    az, _ = cam.az_zen(np.array([48.6]), np.array([2.6]), np.array([11000.0]))  # due east
    assert az[0] == pytest.approx(np.pi / 2, abs=1e-2)
