"""Viewing geometry of the camera: the azimuth and zenith angle of every pixel, and the site.

The two maps come from the camera calibration (degrees; NaN where a pixel has no direction).
Everything else the engine needs from the geometry is derived here once: the quarter-resolution
grids the smooth reference fields are read on, the fisheye scale near the zenith, the unit
direction of every pixel and the optical centre.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from skycam.domain.models import CalibrationData

__all__ = ["CameraGeometry", "Site", "angle_to_sun", "to_vectors"]

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


class Site(BaseModel):
    """Position of the camera (WGS84 degrees), for the sun's position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)


def to_vectors(az: Any, zen: Any) -> F64:
    """Degrees to unit vectors (east, north, up)."""
    a, z = np.radians(az), np.radians(zen)
    return np.asarray(np.stack([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)], axis=-1))


def angle_to_sun(az: Any, zen: Any, saz: float, szen: float) -> F64:
    """Angle (deg) between the directions (az, zen) and the sun (saz, szen)."""
    z, zs = np.radians(zen), np.radians(szen)
    c = np.cos(z) * np.cos(zs) + np.sin(z) * np.sin(zs) * np.cos(np.radians(az - saz))
    return np.asarray(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _load_map(path: Path, key: str) -> NDArray[np.floating]:
    loaded = np.load(path)
    if isinstance(loaded, np.ndarray):
        return loaded
    if key not in loaded.files:
        raise KeyError(f"{path} has no array '{key}'; available: {', '.join(loaded.files)}")
    return np.asarray(loaded[key])


class CameraGeometry:
    """Azimuth (from north towards east) and zenith angle of every pixel, degrees, float64;
    NaN where the pixel sees no sky direction."""

    def __init__(self, azimuth_deg: NDArray[np.floating], zenith_deg: NDArray[np.floating]):
        az = np.asarray(azimuth_deg, dtype=np.float64)
        zen = np.asarray(zenith_deg, dtype=np.float64)
        if az.ndim != 2 or az.shape != zen.shape:
            raise ValueError(f"azimuth {az.shape} and zenith {zen.shape} must be equal 2-D maps")
        out = ~np.isfinite(zen) | ~np.isfinite(az)
        az, zen = az.copy(), zen.copy()
        az[out] = np.nan
        zen[out] = np.nan
        self.azimuth_deg: F64 = az
        self.zenith_deg: F64 = zen

    @classmethod
    def from_files(
        cls,
        azimuth: Path,
        zenith: Path,
        *,
        azimuth_key: str = "azimuth_deg",
        zenith_key: str = "zenith_deg",
        unit: Literal["deg", "rad"] = "deg",
    ) -> CameraGeometry:
        """Load the two maps from ``.npy`` files or arrays of ``.npz`` files."""
        az, zen = _load_map(azimuth, azimuth_key), _load_map(zenith, zenith_key)
        if unit == "rad":
            az, zen = np.degrees(az), np.degrees(zen)
        return cls(az, zen)

    @classmethod
    def from_skycam(cls, calibration: CalibrationData) -> CameraGeometry:
        """Use a skycam calibration (radians; azimuth decoded in [-pi, pi]).

        Install ``skycloud[skycam]``. skycam decodes the azimuth code with a -180 deg shift;
        the geographic azimuth is the raw one (checked on 47 suns against their computed
        position), so 180 deg is added back. Directions beyond 90 deg of zenith (the code
        reserved for "no sky") are excluded. The float path differs from decoding the codes
        in degrees by a few ULPs, so renders are not bit-identical to the degree maps.
        """
        try:
            from skycam.domain.models import CalibrationData
        except ImportError as exc:
            raise ImportError("Skycam calibration support requires skycloud[skycam]") from exc
        if not isinstance(calibration, CalibrationData):
            raise TypeError("Expected a skycam CalibrationData instance")
        az = np.degrees(np.asarray(calibration.azimuth_array, dtype=np.float64)) + 180.0
        zen = np.degrees(np.asarray(calibration.zenith_array, dtype=np.float64))
        bad = (zen > 90.0) | (az > 360.0)
        az[bad] = np.nan
        zen[bad] = np.nan
        return cls(az, zen)

    @property
    def shape(self) -> tuple[int, int]:
        return self.zenith_deg.shape[0], self.zenith_deg.shape[1]

    @cached_property
    def inside(self) -> NDArray[np.bool_]:
        """Pixels that see a sky direction."""
        return np.asarray(np.isfinite(self.zenith_deg))

    @cached_property
    def az4(self) -> F64:
        """Azimuth at quarter resolution (every 4th pixel), 0 outside."""
        return np.nan_to_num(self.azimuth_deg[::4, ::4]).astype(np.float64)

    @cached_property
    def zen4(self) -> F64:
        """Zenith angle at quarter resolution, 90 outside."""
        return np.nan_to_num(self.zenith_deg[::4, ::4], nan=90.0).astype(np.float64)

    @cached_property
    def zen90(self) -> F32:
        """Zenith angle, float32, 90 outside (the zenith bands of the frame statistics)."""
        return np.nan_to_num(self.zenith_deg, nan=90.0).astype(np.float32)

    @cached_property
    def zen99(self) -> F32:
        """Zenith angle, float32, 99 outside (the sky of the gain ledger and the rim rules)."""
        return np.nan_to_num(self.zenith_deg, nan=99.0).astype(np.float32)

    @cached_property
    def px_deg(self) -> float:
        """Pixels per degree of zenith angle near the zenith (the fisheye's scale)."""
        zen = self.zenith_deg
        gz = np.hypot(*np.gradient(np.where(self.inside, zen, np.nan)))
        return float(1.0 / np.nanmedian(gz[np.nan_to_num(zen, nan=90.0) < 60]))

    @cached_property
    def vec(self) -> F32:
        """Unit direction (east, north, up) of every pixel, float32 (outside: the zenith)."""
        az, zen = np.nan_to_num(self.azimuth_deg), np.nan_to_num(self.zenith_deg)
        return to_vectors(az, zen).astype(np.float32)

    @cached_property
    def centre(self) -> F64:
        """The optical centre (x, y): the pixel of smallest zenith angle."""
        iy, ix = np.unravel_index(np.nanargmin(self.zenith_deg), self.zenith_deg.shape)
        return np.array([ix, iy], float)
