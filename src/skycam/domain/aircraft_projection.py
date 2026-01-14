"""Aircraft and geometry projection using analytical camera model.

This module provides vectorized projection for aircraft positions and
Shapely geometries, complementing the calibration-based ProjectionService
with a fast analytical camera model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from geographiclib.geodesic import Geodesic
from pydantic import BaseModel, Field
from shapely import transform
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AircraftProjectionSettings(BaseModel, frozen=True):
    """Settings for aircraft/geometry projection.

    These values control the analytical projection grid.
    Defaults match ProjectionSettings for consistency.
    """

    cloud_height: float = Field(
        default=10000.0,
        ge=100,
        description="Reference altitude for projection plane (meters)",
    )
    square_size: float = Field(
        default=75000.0,
        ge=1000,
        description="Physical size of output grid (meters)",
    )
    resolution: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Output grid resolution (pixels)",
    )


class AircraftProjector:
    """Vectorized projection for aircraft positions and Shapely geometries.

    This projector uses an analytical camera model based on azimuth/zenith
    calculations through the WGS84 ellipsoid. Unlike ProjectionService
    (which uses calibration maps), this provides fast approximate projection.

    All altitudes are in meters (not feet).

    Example:
        >>> from skycam.domain.aircraft_projection import AircraftProjector
        >>> proj = AircraftProjector(camera_lat=48.6, camera_lon=2.35, camera_alt=90.0)
        >>> lon = np.array([2.3, 2.4])
        >>> lat = np.array([48.5, 48.6])
        >>> alt = np.array([10000.0, 10000.0])
        >>> px, py = proj.lonlat_to_pixels(lon, lat, alt)
    """

    def __init__(
        self,
        camera_lat: float,
        camera_lon: float,
        camera_alt: float,
        settings: AircraftProjectionSettings | None = None,
    ) -> None:
        """Initialize the aircraft projector.

        Args:
            camera_lat: Camera latitude in decimal degrees (WGS84).
            camera_lon: Camera longitude in decimal degrees (WGS84).
            camera_alt: Camera altitude in meters above sea level.
            settings: Projection settings. Uses defaults if None.
        """
        self.camera_lat = camera_lat
        self.camera_lon = camera_lon
        self.camera_alt = camera_alt
        self.settings = settings or AircraftProjectionSettings()

        # Derived geometry
        self._half = self.settings.square_size / 2
        self._step = self.settings.square_size / (self.settings.resolution - 1)

        # WGS84 geodesic calculator
        self._geod = Geodesic.WGS84

    # ──────────────────────────────────────────────────────────────────────────
    # Geographic (Lon/Lat) ↔ Spherical (Azimuth/Zenith)
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_azimuth_zenith(
        self,
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
        alt_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute azimuth and zenith from camera to targets (vectorized)."""
        inv_results = [
            self._geod.Inverse(self.camera_lat, self.camera_lon, la, lo)
            for la, lo in zip(lat, lon, strict=True)
        ]

        azi1 = np.array([res["azi1"] for res in inv_results], dtype=np.float64)
        s_ground = np.array([res["s12"] for res in inv_results], dtype=np.float64)

        dz = alt_m - self.camera_alt
        straight = np.sqrt(s_ground**2 + dz**2)
        elevation = np.degrees(np.arcsin(dz / straight))
        zenith = 90.0 - elevation

        return azi1, zenith

    def _azimuth_zenith_to_lonlat(
        self,
        azimuth_deg: NDArray[np.float64],
        zenith_deg: NDArray[np.float64],
        alt_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute lon/lat from azimuth/zenith and altitude (vectorized)."""
        elevation_rad = np.radians(90.0 - zenith_deg)
        dz = alt_m - self.camera_alt
        distance_on_surface = dz / np.tan(elevation_rad)

        direct_results = [
            self._geod.Direct(self.camera_lat, self.camera_lon, az, dist)
            for az, dist in zip(azimuth_deg, distance_on_surface, strict=True)
        ]

        lon = np.array([res["lon2"] for res in direct_results], dtype=np.float64)
        lat = np.array([res["lat2"] for res in direct_results], dtype=np.float64)

        return lon, lat

    # ──────────────────────────────────────────────────────────────────────────
    # Spherical (Azimuth/Zenith) ↔ Planar (X/Y meters)
    # ──────────────────────────────────────────────────────────────────────────

    def _azimuth_zenith_to_xy(
        self,
        azimuth_deg: NDArray[np.float64],
        zenith_deg: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert azimuth/zenith to projection plane (x, y) in meters."""
        az = np.radians(azimuth_deg)
        ze = np.radians(zenith_deg)

        r = self.settings.cloud_height * np.tan(ze)

        # x is East/West (cos), y is North/South (sin)
        x = r * np.cos(az)
        y = r * np.sin(az)

        # Shift to grid coordinates (center at half)
        return y + self._half, self._half - x

    def _xy_to_azimuth_zenith(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert (x, y) in meters to azimuth/zenith."""
        x_centered = self._half - y
        y_centered = x - self._half

        r = np.sqrt(x_centered**2 + y_centered**2)
        az_rad = np.arctan2(y_centered, x_centered)
        azimuth_deg = np.degrees(az_rad)

        ze_rad = np.arctan(r / self.settings.cloud_height)
        zenith_deg = np.degrees(ze_rad)

        return azimuth_deg, zenith_deg

    # ──────────────────────────────────────────────────────────────────────────
    # Planar (X/Y meters) ↔ Pixels
    # ──────────────────────────────────────────────────────────────────────────

    def _xy_to_pixels(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert (x, y) in meters to pixel coordinates."""
        return x / self._step, y / self._step

    def _pixels_to_xy(
        self,
        px: NDArray[np.float64],
        py: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert pixel coordinates to (x, y) in meters."""
        return px * self._step, py * self._step

    # ──────────────────────────────────────────────────────────────────────────
    # Public API: Full Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def lonlat_to_pixels(
        self,
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        alt_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert geographic positions to pixel coordinates.

        Uses analytical camera model projection based on azimuth/zenith
        calculations through the WGS84 ellipsoid.

        Args:
            lon: Longitude array in decimal degrees.
            lat: Latitude array in decimal degrees.
            alt_m: Altitude array in meters above sea level.

        Returns:
            Tuple of (px, py) pixel coordinate arrays. Coordinates are
            floating-point and may lie outside the valid pixel range.

        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        alt_m = np.atleast_1d(np.asarray(alt_m, dtype=np.float64))

        if not (lon.shape == lat.shape == alt_m.shape):
            msg = (
                f"Input arrays must have matching shapes: "
                f"{lon.shape}, {lat.shape}, {alt_m.shape}"
            )
            raise ValueError(msg)

        az, ze = self._calculate_azimuth_zenith(lat, lon, alt_m)
        x, y = self._azimuth_zenith_to_xy(az, ze)
        px, py = self._xy_to_pixels(x, y)

        return px, py

    def pixels_to_lonlat(
        self,
        px: NDArray[np.float64],
        py: NDArray[np.float64],
        alt_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert pixel coordinates to geographic positions.

        Args:
            px: Pixel X coordinate array.
            py: Pixel Y coordinate array.
            alt_m: Target altitude array in meters.

        Returns:
            Tuple of (lon, lat) in decimal degrees.

        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        px = np.atleast_1d(np.asarray(px, dtype=np.float64))
        py = np.atleast_1d(np.asarray(py, dtype=np.float64))
        alt_m = np.atleast_1d(np.asarray(alt_m, dtype=np.float64))

        if not (px.shape == py.shape == alt_m.shape):
            msg = (
                f"Input arrays must have matching shapes: "
                f"{px.shape}, {py.shape}, {alt_m.shape}"
            )
            raise ValueError(msg)

        x, y = self._pixels_to_xy(px, py)
        az, ze = self._xy_to_azimuth_zenith(x, y)
        lon, lat = self._azimuth_zenith_to_lonlat(az, ze, alt_m)

        return lon, lat

    # ──────────────────────────────────────────────────────────────────────────
    # Public API: Shapely Geometry Projection
    # ──────────────────────────────────────────────────────────────────────────

    def _project_geom(
        self,
        geom: BaseGeometry,
        proj_func: Callable[
            [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ],
    ) -> BaseGeometry:
        """Transform a shapely geometry using a projection function."""
        if geom.is_empty:
            return geom

        if not geom.has_z:
            msg = (
                "Coordinates must have Z (altitude in meters). "
                "2D geometry not supported."
            )
            raise ValueError(msg)

        def wrapper(coords: NDArray[np.float64]) -> NDArray[np.float64]:
            c1 = coords[:, 0]
            c2 = coords[:, 1]
            alt_m = coords[:, 2]

            c1_proj, c2_proj = proj_func(c1, c2, alt_m)

            return np.column_stack((c1_proj, c2_proj, alt_m))

        return transform(geom, wrapper, include_z=True)

    def project_geometry(self, geom: BaseGeometry) -> BaseGeometry:
        """Project a 3D Shapely geometry to pixel space.

        Input geometry coordinates are (lon, lat, alt_m).
        Output geometry coordinates are (px, py, alt_m).

        Args:
            geom: 3D Shapely geometry with Z as altitude in meters.

        Returns:
            Projected geometry in pixel coordinates.

        Raises:
            ValueError: If geometry lacks Z coordinate.
        """
        return self._project_geom(geom, self.lonlat_to_pixels)

    def project_geometry_back(self, geom: BaseGeometry) -> BaseGeometry:
        """Project a 3D Shapely geometry from pixel space to geographic.

        Input geometry coordinates are (px, py, alt_m).
        Output geometry coordinates are (lon, lat, alt_m).

        Args:
            geom: 3D Shapely geometry with Z as altitude in meters.

        Returns:
            Geographic geometry in (lon, lat, alt_m) coordinates.

        Raises:
            ValueError: If geometry lacks Z coordinate.
        """
        return self._project_geom(geom, self.pixels_to_lonlat)
