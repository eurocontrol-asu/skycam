"""Viewing geometry of the camera: the direction of every pixel.

Azimuth (degrees from north through east) and zenith angle of every pixel at the native size of
the frames, from the camera calibration; NaN (or anything beyond 90 degrees of zenith) where a
pixel sees no sky direction. Everything skysun needs is derived once from them: the unit
direction of every pixel (for its angle to the sun) and the pixel the sun falls on.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from skysun.sun import SunPosition

__all__ = ["CameraGeometry"]

F32 = NDArray[np.float32]


def _unit(azimuth_deg: NDArray[np.floating], zenith_deg: NDArray[np.floating]) -> F32:
    a, z = np.radians(azimuth_deg), np.radians(zenith_deg)
    return np.stack([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)], axis=-1).astype(
        np.float32
    )


class CameraGeometry:
    """Azimuth and zenith of every pixel (degrees, float32); outside the sky domain: NaN."""

    def __init__(self, azimuth_deg: NDArray[np.floating], zenith_deg: NDArray[np.floating]):
        azimuth = np.asarray(azimuth_deg, dtype=np.float32)
        zenith = np.asarray(zenith_deg, dtype=np.float32)
        if zenith.ndim != 2 or azimuth.shape != zenith.shape:
            raise ValueError(f"azimuth {azimuth.shape} and zenith {zenith.shape}: equal 2-D maps")
        outside = ~np.isfinite(azimuth) | ~np.isfinite(zenith) | (zenith > 90)
        self.azimuth_deg: F32 = np.where(outside, np.nan, azimuth).astype(np.float32)
        self.zenith_deg: F32 = np.where(outside, np.nan, zenith).astype(np.float32)
        if not self.inside.any():
            raise ValueError("The geometry has no pixel with a sky direction")

    @classmethod
    def from_file(
        cls, path: Path, azimuth_key: str = "azimuth_deg", zenith_key: str = "zenith_deg"
    ) -> CameraGeometry:
        """Both maps from one ``.npz`` (degrees)."""
        with np.load(path, allow_pickle=False) as stored:
            missing = {azimuth_key, zenith_key} - set(stored.files)
            if missing:
                raise KeyError(f"{path} lacks {sorted(missing)}; it has {stored.files}")
            return cls(stored[azimuth_key], stored[zenith_key])

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns)."""
        return self.zenith_deg.shape[0], self.zenith_deg.shape[1]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height), the order of image sizes."""
        return self.zenith_deg.shape[1], self.zenith_deg.shape[0]

    @cached_property
    def inside(self) -> NDArray[np.bool_]:
        """Pixels that see a sky direction (zenith angle up to 90 degrees)."""
        return np.asarray(np.isfinite(self.zenith_deg))

    @cached_property
    def directions(self) -> F32:
        """Unit direction (east, north, up) of every pixel; the nadir outside the sky."""
        az = np.where(self.inside, self.azimuth_deg, 0.0)
        zen = np.where(self.inside, self.zenith_deg, 180.0)
        return _unit(az, zen)

    def angle_to(self, sun: SunPosition) -> F32:
        """Angle (degrees) between every pixel's direction and the sun; 180 outside the sky."""
        vector = _unit(np.array(sun.azimuth_deg), np.array(sun.zenith_deg))
        cosine = np.asarray(self.directions @ vector, dtype=np.float32)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).astype(np.float32)
        angle[~self.inside] = 180.0
        return angle

    def sun_pixel(self, angle: F32) -> tuple[float, float]:
        """(x, y) of the pixel closest to the sun, given ``angle_to(sun)``."""
        row, col = np.unravel_index(int(np.argmin(angle)), angle.shape)
        return float(col), float(row)
