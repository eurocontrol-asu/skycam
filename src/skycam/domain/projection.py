"""Projection service for fisheye image transformation.

This module contains the core domain logic for projecting hemispherical
camera images onto a regular grid using calibration data.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from skycam.domain.exceptions import ProjectionError
from skycam.domain.models import CalibrationData, ProjectionSettings

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ProjectionService:
    """Service for projecting fisheye images to regular grids.

    This service encapsulates all the interpolation logic for transforming
    raw hemispherical camera images into projected grid coordinates.

    The projection uses:
    1. Calibration data (azimuth/zenith maps) to define the camera geometry
    2. Projection settings (resolution, cloud height, etc.) to define output

    Example:
        >>> calibration = loader.load("visible")
        >>> settings = ProjectionSettings()
        >>> service = ProjectionService(calibration, settings)
        >>> projected = service.project(raw_image)
    """

    calibration: CalibrationData
    settings: ProjectionSettings

    # Private fields for cached interpolators
    _azimuth_zenith_to_pixel_raw: LinearNDInterpolator | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _interpolation_grid: NDArray[np.float64] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _azimuth_zenith_grid: NDArray[np.float64] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Initialize interpolators after dataclass construction."""
        self._init_interpolators()

    def _init_interpolators(self) -> None:
        """Build the interpolation grids and cached interpolators.

        This sets up:
        1. The output grid based on square_size and resolution
        2. The azimuth/zenith values for each grid point
        3. The LinearNDInterpolator for azimuth/zenith → pixel mapping
        """
        resolution = self.settings.resolution
        square_size = self.settings.square_size
        cloud_height = self.settings.cloud_height
        max_zenith = self.settings.max_zenith_angle

        # Get calibration arrays (cast from object type)
        azimuth_array = np.asarray(self.calibration.azimuth_array, dtype=np.float64)
        zenith_array = np.asarray(self.calibration.zenith_array, dtype=np.float64)
        image_size = self.calibration.image_size

        # Build the output grid
        half_size = square_size / 2
        step = square_size / (resolution - 1)
        x = np.arange(-half_size, half_size + step, step)
        y = np.arange(-half_size, half_size + step, step)
        grid_xy = np.meshgrid(x, y)

        # Calculate zenith and azimuth for output grid points
        r = np.sqrt(grid_xy[0] ** 2 + grid_xy[1] ** 2)
        interpolation_zenith = np.arctan(r / cloud_height)
        interpolation_azimuth = np.arctan2(grid_xy[1], grid_xy[0])

        # CRITICAL: Legacy azimuth alignment formula
        # This specific transformation aligns computed azimuths with the
        # legacy JP2 calibration maps
        interpolation_azimuth = (interpolation_azimuth - 3 * np.pi / 2) % (
            2 * np.pi
        ) - np.pi

        # Store azimuth/zenith grid for projection
        self._azimuth_zenith_grid = np.stack(
            [interpolation_azimuth, interpolation_zenith], axis=-1
        )

        # Create restriction mask for max zenith angle
        restriction_array = np.where(
            zenith_array > max_zenith * np.pi / 180, np.nan, 1.0
        )

        # Flatten and mask calibration arrays
        flattened_azimuth = (restriction_array * azimuth_array).flatten()
        flattened_zenith = (restriction_array * zenith_array).flatten()

        # Create combined NaN mask
        mask_nan = np.isnan(flattened_azimuth) | np.isnan(flattened_zenith)

        # Filter valid values
        filtered_azimuth = flattened_azimuth[~mask_nan]
        filtered_zenith = flattened_zenith[~mask_nan]
        azimuth_zenith = np.stack([filtered_azimuth, filtered_zenith], axis=-1)

        # Create coordinate grid for raw image
        x_r = np.arange(image_size[0])
        y_r = np.arange(image_size[1])
        grid = np.meshgrid(x_r, y_r)
        coordinates = np.stack(
            [grid[0].T.flatten()[~mask_nan], grid[1].T.flatten()[~mask_nan]], axis=-1
        )

        # Build the interpolator: azimuth/zenith → pixel coordinates in raw image
        self._azimuth_zenith_to_pixel_raw = LinearNDInterpolator(
            azimuth_zenith, coordinates
        )

    def project(
        self,
        image: NDArray[np.uint8],
        as_uint8: bool = True,
    ) -> NDArray[np.uint8] | NDArray[np.float64]:
        """Project a raw fisheye image to the output grid.

        Args:
            image: Input image array (H, W, C) as uint8
            as_uint8: If True, return uint8; otherwise return float64

        Returns:
            Projected image array (resolution, resolution, C)

        Raises:
            ProjectionError: If projection fails
        """
        if self._azimuth_zenith_to_pixel_raw is None:
            raise ProjectionError("Interpolators not initialized")
        if self._azimuth_zenith_grid is None:
            raise ProjectionError("Grid not initialized")

        try:
            # Create interpolator for raw image RGB values
            x_r = np.arange(image.shape[0])
            y_r = np.arange(image.shape[1])
            pixel_to_rgb = RegularGridInterpolator(
                (x_r, y_r),
                image,
                bounds_error=False,
                fill_value=0,
            )

            # Map azimuth/zenith grid to raw pixel coordinates
            projected_grid = self._azimuth_zenith_to_pixel_raw(
                self._azimuth_zenith_grid
            )

            # Interpolate RGB values for projected grid
            projected_image = pixel_to_rgb(projected_grid)

            if as_uint8:
                result: NDArray[np.uint8] = projected_image.astype(np.uint8)
                return result
            result_f64: NDArray[np.float64] = projected_image.astype(np.float64)
            return result_f64

        except Exception as e:
            raise ProjectionError(f"Projection failed: {e}") from e

    def calculate_azimuth_zenith(
        self,
        target_lat: float,
        target_lon: float,
        target_alt: float,
        observer_lat: float,
        observer_lon: float,
        observer_alt: float,
    ) -> tuple[float, float]:
        """Calculate azimuth and zenith angles from observer to target.

        Uses the WGS84 ellipsoid for geodesic calculations.

        Args:
            target_lat: Target latitude in decimal degrees
            target_lon: Target longitude in decimal degrees
            target_alt: Target altitude in meters
            observer_lat: Observer latitude in decimal degrees
            observer_lon: Observer longitude in decimal degrees
            observer_alt: Observer altitude in meters

        Returns:
            Tuple of (azimuth_degrees, zenith_degrees)
        """
        import math

        from geographiclib.geodesic import Geodesic

        wgs84 = Geodesic.WGS84
        inverse_coords = wgs84.Inverse(
            observer_lat, observer_lon, target_lat, target_lon
        )

        azimuth = float(inverse_coords["azi1"])
        distance_on_surface = float(inverse_coords["s12"])

        delta_altitude = target_alt - observer_alt
        straight_distance = math.sqrt(distance_on_surface**2 + delta_altitude**2)
        elevation_angle = math.degrees(math.asin(delta_altitude / straight_distance))
        zenith = 90.0 - elevation_angle

        return azimuth, zenith

    def calculate_latitude_longitude(
        self,
        azimuth: float,
        zenith: float,
        target_altitude: float,
        observer_lat: float,
        observer_lon: float,
        observer_alt: float,
    ) -> tuple[float, float]:
        """Calculate target lat/lon from azimuth and zenith angles.

        Uses the WGS84 ellipsoid for geodesic calculations.

        Args:
            azimuth: Azimuth angle in degrees
            zenith: Zenith angle in degrees
            target_altitude: Target altitude in meters
            observer_lat: Observer latitude in decimal degrees
            observer_lon: Observer longitude in decimal degrees
            observer_alt: Observer altitude in meters

        Returns:
            Tuple of (latitude, longitude) in decimal degrees
        """
        import math

        from geographiclib.geodesic import Geodesic

        wgs84 = Geodesic.WGS84

        elevation_angle_rad = math.radians(90 - zenith)
        delta_altitude = target_altitude - observer_alt
        distance_on_surface = delta_altitude / math.tan(elevation_angle_rad)

        direct_result = wgs84.Direct(
            observer_lat, observer_lon, azimuth, distance_on_surface
        )

        return float(direct_result["lat2"]), float(direct_result["lon2"])
