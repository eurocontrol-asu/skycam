"""Tests for aircraft projection.

TDD RED Phase: All tests are expected to fail until implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import LineString, Point, Polygon

from skycam.domain.aircraft_projection import (
    AircraftProjectionSettings,
    AircraftProjector,
)


class TestAircraftProjectionSettings:
    """Tests for AircraftProjectionSettings Pydantic model."""

    def test_default_values(self) -> None:
        """Settings have sensible defaults."""
        settings = AircraftProjectionSettings()
        assert settings.cloud_height == 10000.0
        assert settings.square_size == 75000.0
        assert settings.resolution == 1024

    def test_custom_values(self) -> None:
        """Settings accept custom values."""
        settings = AircraftProjectionSettings(
            cloud_height=12000.0,
            square_size=100000.0,
            resolution=2048,
        )
        assert settings.cloud_height == 12000.0
        assert settings.square_size == 100000.0
        assert settings.resolution == 2048

    def test_frozen(self) -> None:
        """Settings are immutable."""
        from pydantic import ValidationError

        settings = AircraftProjectionSettings()
        with pytest.raises(ValidationError):
            settings.resolution = 512  # type: ignore[misc]


class TestAircraftProjectorInit:
    """Tests for AircraftProjector initialization."""

    def test_init_with_defaults(self) -> None:
        """AircraftProjector initializes with default settings."""
        proj = AircraftProjector(
            camera_lat=48.6,
            camera_lon=2.35,
            camera_alt=90.0,
        )
        assert proj.camera_lat == 48.6
        assert proj.camera_lon == 2.35
        assert proj.camera_alt == 90.0
        assert proj.settings is not None

    def test_init_with_custom_settings(self) -> None:
        """AircraftProjector accepts custom settings."""
        settings = AircraftProjectionSettings(resolution=2048)
        proj = AircraftProjector(
            camera_lat=48.6,
            camera_lon=2.35,
            camera_alt=90.0,
            settings=settings,
        )
        assert proj.settings.resolution == 2048


class TestLonLatToPixels:
    """Tests for lonlat_to_pixels vectorized conversion."""

    def test_single_point(self, aircraft_projector: AircraftProjector) -> None:
        """Convert single point to pixels."""
        lon = np.array([2.35])
        lat = np.array([48.65])
        alt = np.array([10000.0])

        px, py = aircraft_projector.lonlat_to_pixels(lon, lat, alt)

        assert px.shape == (1,)
        assert py.shape == (1,)
        assert np.isfinite(px[0])
        assert np.isfinite(py[0])

    def test_batch_points(self, aircraft_projector: AircraftProjector) -> None:
        """Convert batch of points to pixels."""
        n = 100
        lon = np.linspace(2.0, 2.7, n)
        lat = np.linspace(48.3, 48.9, n)
        alt = np.full(n, 10000.0)

        px, py = aircraft_projector.lonlat_to_pixels(lon, lat, alt)

        assert px.shape == (n,)
        assert py.shape == (n,)

    def test_camera_directly_above(self, aircraft_projector: AircraftProjector) -> None:
        """Point directly above camera projects to center."""
        lon = np.array([aircraft_projector.camera_lon])
        lat = np.array([aircraft_projector.camera_lat])
        alt = np.array([10000.0])

        px, py = aircraft_projector.lonlat_to_pixels(lon, lat, alt)

        # Center of grid should be at half (resolution * step / step)
        half = aircraft_projector.settings.square_size / 2
        step = aircraft_projector.settings.square_size / (
            aircraft_projector.settings.resolution - 1
        )
        expected_center = half / step

        assert_allclose(px[0], expected_center, atol=1.0)
        assert_allclose(py[0], expected_center, atol=1.0)

    def test_mismatched_shapes_raises(
        self, aircraft_projector: AircraftProjector
    ) -> None:
        """Mismatched input shapes raise ValueError."""
        lon = np.array([2.3, 2.4])
        lat = np.array([48.5])  # Different size!
        alt = np.array([10000.0])

        with pytest.raises(ValueError, match="shape"):
            aircraft_projector.lonlat_to_pixels(lon, lat, alt)


class TestPixelsToLonLat:
    """Tests for pixels_to_lonlat vectorized conversion."""

    def test_single_pixel(self, aircraft_projector: AircraftProjector) -> None:
        """Convert single pixel to geographic coordinates."""
        px = np.array([512.0])
        py = np.array([512.0])
        alt = np.array([10000.0])

        lon, lat = aircraft_projector.pixels_to_lonlat(px, py, alt)

        assert lon.shape == (1,)
        assert lat.shape == (1,)

    def test_center_pixel_returns_camera_position(
        self, aircraft_projector: AircraftProjector
    ) -> None:
        """Center pixel maps to camera position."""
        half = aircraft_projector.settings.square_size / 2
        step = aircraft_projector.settings.square_size / (
            aircraft_projector.settings.resolution - 1
        )
        center_px = half / step

        px = np.array([center_px])
        py = np.array([center_px])
        alt = np.array([10000.0])

        lon, lat = aircraft_projector.pixels_to_lonlat(px, py, alt)

        assert_allclose(lon[0], aircraft_projector.camera_lon, atol=0.01)
        assert_allclose(lat[0], aircraft_projector.camera_lat, atol=0.01)


class TestRoundTrip:
    """Tests for projection round-trip accuracy."""

    def test_lonlat_pixels_lonlat_roundtrip(
        self, aircraft_projector: AircraftProjector
    ) -> None:
        """lon/lat → pixels → lon/lat round-trip preserves coordinates."""
        lon_orig = np.array([2.3, 2.35, 2.4])
        lat_orig = np.array([48.55, 48.60, 48.65])
        alt = np.array([10000.0, 10000.0, 10000.0])

        # Forward
        px, py = aircraft_projector.lonlat_to_pixels(lon_orig, lat_orig, alt)

        # Back
        lon_back, lat_back = aircraft_projector.pixels_to_lonlat(px, py, alt)

        # Should match within geodesic precision
        assert_allclose(lon_back, lon_orig, atol=1e-5)
        assert_allclose(lat_back, lat_orig, atol=1e-5)

    def test_pixels_lonlat_pixels_roundtrip(
        self, aircraft_projector: AircraftProjector
    ) -> None:
        """pixels → lon/lat → pixels round-trip preserves coordinates."""
        px_orig = np.array([400.0, 512.0, 600.0])
        py_orig = np.array([400.0, 512.0, 600.0])
        alt = np.array([10000.0, 10000.0, 10000.0])

        # Forward
        lon, lat = aircraft_projector.pixels_to_lonlat(px_orig, py_orig, alt)

        # Back
        px_back, py_back = aircraft_projector.lonlat_to_pixels(lon, lat, alt)

        assert_allclose(px_back, px_orig, atol=1e-3)
        assert_allclose(py_back, py_orig, atol=1e-3)


class TestShapelyProjection:
    """Tests for Shapely geometry projection."""

    def test_project_point(self, aircraft_projector: AircraftProjector) -> None:
        """Project 3D Point geometry."""
        point = Point(2.35, 48.60, 10000.0)  # lon, lat, alt_m
        projected = aircraft_projector.project_geometry(point)

        assert projected.has_z
        assert projected.geom_type == "Point"

    def test_project_linestring(self, aircraft_projector: AircraftProjector) -> None:
        """Project 3D LineString (flight path)."""
        line = LineString(
            [
                (2.3, 48.5, 10000.0),
                (2.35, 48.55, 10500.0),
                (2.4, 48.6, 11000.0),
            ]
        )
        projected = aircraft_projector.project_geometry(line)

        assert projected.has_z
        assert projected.geom_type == "LineString"
        assert len(projected.coords) == 3

    def test_project_polygon(self, aircraft_projector: AircraftProjector) -> None:
        """Project 3D Polygon (airspace sector)."""
        polygon = Polygon(
            [
                (2.3, 48.5, 10000.0),
                (2.4, 48.5, 10000.0),
                (2.4, 48.6, 10000.0),
                (2.3, 48.6, 10000.0),
                (2.3, 48.5, 10000.0),  # Close the ring
            ]
        )
        projected = aircraft_projector.project_geometry(polygon)

        assert projected.has_z
        assert projected.geom_type == "Polygon"

    def test_project_2d_geometry_raises(
        self, aircraft_projector: AircraftProjector
    ) -> None:
        """2D geometry without altitude raises ValueError."""
        point_2d = Point(2.35, 48.60)  # No Z coordinate

        with pytest.raises(ValueError, match="must have"):
            aircraft_projector.project_geometry(point_2d)

    def test_project_back_linestring(
        self, aircraft_projector: AircraftProjector
    ) -> None:
        """Project back a LineString from pixel space."""
        # Create in pixel space
        line_pixels = LineString(
            [
                (400, 400, 10000.0),
                (512, 512, 10000.0),
                (600, 600, 10000.0),
            ]
        )

        geographic = aircraft_projector.project_geometry_back(line_pixels)

        assert geographic.has_z
        assert geographic.geom_type == "LineString"

    def test_geometry_roundtrip(self, aircraft_projector: AircraftProjector) -> None:
        """Geometry projection round-trip preserves coordinates."""
        original = LineString(
            [
                (2.3, 48.5, 10000.0),
                (2.35, 48.55, 10000.0),
                (2.4, 48.6, 10000.0),
            ]
        )

        projected = aircraft_projector.project_geometry(original)
        recovered = aircraft_projector.project_geometry_back(projected)

        # Compare coordinates
        orig_coords = np.array(original.coords)
        rec_coords = np.array(recovered.coords)

        assert_allclose(rec_coords, orig_coords, atol=1e-5)

    def test_empty_geometry(self, aircraft_projector: AircraftProjector) -> None:
        """Empty geometry returns empty geometry."""
        empty_point = Point()
        result = aircraft_projector.project_geometry(empty_point)
        assert result.is_empty
