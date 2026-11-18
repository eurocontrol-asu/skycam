from __future__ import annotations

import numpy as np

from skycloud.geometry import CameraGeometry
from skycloud.reference_build.features import FeatureGrid, corner_sd, frame_features, grid_pixels
from skycloud.reference_build.params import BinParams


def fisheye(n: int = 101, radius: float = 45.0) -> CameraGeometry:
    """A synthetic equidistant fisheye: zenith = 90 r / radius, azimuth from north (up)."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    dx, dy = x - n // 2, n // 2 - y
    r = np.hypot(dx, dy)
    zen = np.where(r <= radius, 90.0 * r / radius, np.nan)
    az = np.where(r <= radius, np.degrees(np.arctan2(dx, dy)) % 360, np.nan)
    return CameraGeometry(az, zen)


def test_grid_pixels_pick_the_nearest_pixel_of_each_direction() -> None:
    geo = fisheye()
    p = BinParams()
    gp = grid_pixels(geo, p)
    assert gp.shape == (16200,)
    # direction (azimuth 0, zenith 0) is the centre; (azimuth 90, zenith 45) is right of it
    centre = 50 * 101 + 50
    assert gp[0] == centre
    iy, ix = np.divmod(gp[90 * 45 + 45], 101)  # azimuth index 45 (90 deg), zenith 45 deg
    assert abs(geo.azimuth_deg[iy, ix] - 90) < 3 and abs(geo.zenith_deg[iy, ix] - 45) < 3


def test_frame_features_average_the_sky_pixels_of_each_window() -> None:
    geo = fisheye()
    p = BinParams()
    sky = np.isfinite(geo.zenith_deg)
    sky[50, 51] = False  # a non-sky pixel next to the centre: left out of the means
    grid = FeatureGrid.build(geo, sky, p)
    img = np.zeros((101, 101, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 40, 80, 120
    img[50, 51] = (0, 0, 0)
    img[50, 70] = (255, 80, 120)  # a saturated pixel east of the centre, 40 deg of zenith
    f = frame_features(img, grid, p)
    centre = int(np.flatnonzero(grid.gp == 50 * 101 + 50)[0])
    assert f[centre].tolist() == [40, 80, 120, 0, 0]
    assert grid.valid_n[centre] == 80
    east = int(np.flatnonzero(grid.gp == 50 * 101 + 70)[0])
    assert np.isclose(f[east, 3], 1 / 81) and np.isclose(f[east, 0], 40 + 215 / 81)
    assert f[east, 4] == 0  # G unchanged


def test_corner_sd_reads_the_four_corners() -> None:
    img = np.zeros((200, 200, 3), np.uint8)
    img[:60, :60, 0] = 10  # one corner of four at 10: sd = sqrt(0.25 * 0.75) * 10
    assert np.isclose(corner_sd(img)[0], np.sqrt(0.1875) * 10)
    assert corner_sd(img)[1] == 0
