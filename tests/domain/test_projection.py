"""Tests for projection service."""

import numpy as np
from numpy.typing import NDArray

from skycam.domain.projection import ProjectionService


class TestProjectionService:
    """Tests for ProjectionService.

    Uses session-scoped projector to avoid rebuilding interpolators.
    """

    def test_project_returns_ndarray(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() returns a numpy array."""
        result = projector_session.project(sample_image_session)
        assert isinstance(result, np.ndarray)

    def test_project_output_dtype(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() returns uint8 by default."""
        result = projector_session.project(sample_image_session)
        assert result.dtype == np.uint8

    def test_project_output_shape(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() produces correct output dimensions."""
        settings = projector_session.settings
        result = projector_session.project(sample_image_session)

        assert result.shape[0] == settings.resolution
        assert result.shape[1] == settings.resolution
        # RGB channels
        assert result.shape[2] == 3

    def test_project_float_output(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() can return float64 when requested."""
        result = projector_session.project(sample_image_session, as_uint8=False)
        assert result.dtype == np.float64


class TestGeodesicCalculations:
    """Tests for geodesic calculation methods.

    Uses session-scoped projector to avoid rebuilding interpolators.
    """

    def test_azimuth_zenith_calculation(
        self,
        projector_session: ProjectionService,
    ) -> None:
        """calculate_azimuth_zenith returns valid angles."""
        # Test with a point north of the observer
        azimuth, zenith = projector_session.calculate_azimuth_zenith(
            target_lat=49.0,  # North of observer
            target_lon=2.346,  # Same longitude
            target_alt=10000,  # At cloud height
            observer_lat=48.600518087374105,
            observer_lon=2.3467954996250784,
            observer_alt=90.0,
        )

        # Azimuth should be approximately north (0 degrees)
        assert -10 < azimuth < 10

        # Zenith should be less than 90 (above horizon)
        assert 0 < zenith < 90

    def test_latitude_longitude_calculation(
        self,
        projector_session: ProjectionService,
    ) -> None:
        """calculate_latitude_longitude returns valid coordinates."""
        lat, lon = projector_session.calculate_latitude_longitude(
            azimuth=0.0,  # North
            zenith=45.0,  # 45 degrees from vertical
            target_altitude=10000,
            observer_lat=48.600518087374105,
            observer_lon=2.3467954996250784,
            observer_alt=90.0,
        )

        # Result should be north of observer
        assert lat > 48.600518087374105

        # Longitude should be approximately the same
        assert abs(lon - 2.3467954996250784) < 0.1

    def test_round_trip_coordinates(
        self,
        projector_session: ProjectionService,
    ) -> None:
        """Azimuth/zenith → lat/lon → azimuth/zenith round trip."""
        original_az = 45.0
        original_zen = 60.0
        observer_lat = 48.600518087374105
        observer_lon = 2.3467954996250784
        observer_alt = 90.0
        target_alt = 10000.0

        # Convert to lat/lon
        lat, lon = projector_session.calculate_latitude_longitude(
            azimuth=original_az,
            zenith=original_zen,
            target_altitude=target_alt,
            observer_lat=observer_lat,
            observer_lon=observer_lon,
            observer_alt=observer_alt,
        )

        # Convert back to azimuth/zenith
        az, zen = projector_session.calculate_azimuth_zenith(
            target_lat=lat,
            target_lon=lon,
            target_alt=target_alt,
            observer_lat=observer_lat,
            observer_lon=observer_lon,
            observer_alt=observer_alt,
        )

        # Should match original within floating point tolerance
        assert abs(az - original_az) < 0.1
        assert abs(zen - original_zen) < 0.1
