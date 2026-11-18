"""Viewing geometry of the camera: the zenith angle of every pixel.

The zenith map comes from the camera calibration. It bounds the analysis (below a zenith limit)
and tells where the sky certainly is (around the zenith).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from skycam.domain.models import CalibrationData

__all__ = ["SkyGeometry"]


@dataclass(frozen=True)
class SkyGeometry:
    """Zenith angle in degrees per pixel; pixels without a direction hold ``inf``.

    Optional, to keep what the sun leaves out of the marks: the azimuth of every pixel (degrees
    from north through east) and the camera site (latitude, longitude). See :meth:`with_sun`.
    """

    zenith_deg: NDArray[np.float32]
    azimuth_deg: NDArray[np.float32] | None = None
    site: tuple[float, float] | None = None

    def with_sun(
        self, azimuth_deg: NDArray[np.floating], latitude: float, longitude: float
    ) -> SkyGeometry:
        """The same geometry with the pixel azimuths and the site, so the sun can be placed."""
        azimuth = np.asarray(azimuth_deg, dtype=np.float32)
        if azimuth.shape != self.zenith_deg.shape:
            raise ValueError("Azimuth and zenith maps must have the same shape")
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 360):
            raise ValueError("Site latitude/longitude out of range")
        return SkyGeometry(self.zenith_deg, azimuth, (float(latitude), float(longitude)))

    @classmethod
    def from_array(
        cls, zenith: NDArray[np.floating], unit: Literal["deg", "rad"] = "deg"
    ) -> SkyGeometry:
        degrees = np.degrees(zenith) if unit == "rad" else np.asarray(zenith)
        degrees = np.where(np.isfinite(degrees), degrees, np.inf).astype(np.float32)
        if degrees.ndim != 2:
            raise ValueError("The zenith map must be a 2-D array")
        return cls(degrees)

    @classmethod
    def from_file(
        cls, path: Path, key: str = "zenith_deg", unit: Literal["deg", "rad"] = "deg"
    ) -> SkyGeometry:
        """Load from ``.npy`` or from one array of a ``.npz``."""
        loaded = np.load(path)
        if isinstance(loaded, np.ndarray):
            return cls.from_array(loaded, unit)
        if key not in loaded.files:
            raise KeyError(f"{path} has no array '{key}'; available: {', '.join(loaded.files)}")
        return cls.from_array(loaded[key], unit)

    @classmethod
    def from_skycam(cls, calibration: CalibrationData) -> SkyGeometry:
        """Use a skycam calibration's native zenith map (radians).

        Install ``skystatic[skycam]`` to use this constructor. Invalid
        directions are excluded exactly as in :meth:`from_array`.
        """
        try:
            from skycam.domain.models import CalibrationData
        except ImportError as exc:
            raise ImportError("Skycam calibration support requires skystatic[skycam]") from exc
        if not isinstance(calibration, CalibrationData):
            raise TypeError("Expected a skycam CalibrationData instance")
        zenith = np.asarray(calibration.zenith_array, dtype=np.float64)
        geometry = cls.from_array(zenith, unit="rad")
        if geometry.shape != calibration.image_size:
            raise ValueError("Zenith map shape does not match calibration image_size")
        return geometry

    @property
    def shape(self) -> tuple[int, int]:
        return self.zenith_deg.shape[0], self.zenith_deg.shape[1]

    def resized(self, size: tuple[int, int]) -> SkyGeometry:
        """Geometry sampled at pixel centres of a (width, height) image of the same view."""
        width, height = size
        rows, cols = self.shape
        if (rows, cols) == (height, width):
            return self
        if abs(rows / height - cols / width) > 0.01 * rows / height:
            raise ValueError(
                f"Zenith map {cols}x{rows} and image {width}x{height} have different aspects"
            )
        y = np.minimum(((np.arange(height) + 0.5) * rows / height).astype(int), rows - 1)
        x = np.minimum(((np.arange(width) + 0.5) * cols / width).astype(int), cols - 1)
        azimuth = None if self.azimuth_deg is None else self.azimuth_deg[np.ix_(y, x)]
        return SkyGeometry(self.zenith_deg[np.ix_(y, x)], azimuth, self.site)

    def below(self, limit_deg: float) -> NDArray[np.bool_]:
        """Pixels looking closer to the zenith than ``limit_deg``."""
        return self.zenith_deg < limit_deg
