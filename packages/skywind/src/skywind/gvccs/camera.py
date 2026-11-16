"""Camera geometry: local ENU frame of the camera, azimuth / zenith of 3-D points, GVCCS image pixels.

The GVCCS images (1024 x 1024) come with ``azimuth_zenith_grid.npy`` (1026 x 1026 x 2, one-pixel border) giving the
azimuth and zenith of every pixel. The inverse mapping (angles -> pixel) is the linear least-squares fit of the pixel
coordinates on (tan(zenith) sin(azimuth + pi), tan(zenith) cos(azimuth + pi), 1) over the grid (the images are a planar
projection of the sky), as used to build the prototype's calibration ribbons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skywind.geo import FloatArray

__all__ = ["Camera"]

_A, _E2 = 6378137.0, 6.6943799901413165e-3


def _ecef(lat: FloatArray, lon: FloatArray, h: FloatArray) -> FloatArray:
    la, lo = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.stack(
        ((n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1 - _E2) + h) * np.sin(la)), -1
    )


@dataclass
class Camera:
    lat: float
    lon: float
    alt_m: float
    grid: FloatArray  # (1024, 1024, 2): azimuth, zenith of each image pixel (row = y, column = x)
    coef: FloatArray  # (3, 2) inverse mapping
    x_offset: float = 1.0  # GVCCS image x = fitted x + 1 (matches the dataset's own PIXEL_X of the aircraft tracks)

    @classmethod
    def from_grid(cls, path: Path, lat: float, lon: float, alt_m: float, x_offset: float = 1.0) -> Camera:
        grid = np.load(path)[1:-1, 1:-1].astype(np.float64)
        ga, gz = grid[..., 0] + np.pi, grid[..., 1]
        rad = np.tan(gz)
        yy, xx = np.indices(grid.shape[:2])
        A = np.stack([rad * np.sin(ga), rad * np.cos(ga), np.ones_like(rad)], -1)[::8, ::8].reshape(-1, 3)
        B = np.stack([xx, yy], -1)[::8, ::8].reshape(-1, 2).astype(float)
        coef = np.linalg.lstsq(A, B, rcond=None)[0]
        return cls(lat, lon, alt_m, grid, coef, x_offset)

    def _basis(self) -> FloatArray:
        la, lo = np.radians(self.lat), np.radians(self.lon)
        return np.array(
            [
                [-np.sin(lo), np.cos(lo), 0.0],
                [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
            ]
        )

    def enu(self, lat: FloatArray, lon: FloatArray, h: FloatArray) -> FloatArray:
        """East / north / up (m) of points relative to the camera."""
        origin = _ecef(np.array(self.lat), np.array(self.lon), np.array(self.alt_m))
        d = _ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float)) - origin
        return np.asarray(d @ self._basis().T, dtype=np.float64)

    def az_zen(self, lat: FloatArray, lon: FloatArray, h: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Azimuth (rad, from north, clockwise) and zenith angle (rad) of points."""
        e = self.enu(lat, lon, h)
        return np.arctan2(e[..., 0], e[..., 1]), np.arccos(e[..., 2] / np.linalg.norm(e, axis=-1))

    def pixel(self, az: FloatArray, zen: FloatArray) -> tuple[FloatArray, FloatArray]:
        """GVCCS image pixel (x, y) of directions."""
        t = np.tan(np.minimum(zen, 1.5))
        p = np.stack([t * np.sin(az), t * np.cos(az), np.ones_like(t)], -1) @ self.coef
        return p[..., 0] + self.x_offset, p[..., 1]

    def direction(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Azimuth / zenith of image pixels (nearest grid node), in the convention of :meth:`az_zen`."""
        xi = np.clip(np.rint(np.asarray(x) - self.x_offset).astype(int), 0, self.grid.shape[1] - 1)
        yi = np.clip(np.rint(y).astype(int), 0, self.grid.shape[0] - 1)
        g = self.grid[yi, xi]
        return (g[..., 0] + np.pi + np.pi) % (2 * np.pi) - np.pi, g[..., 1]
