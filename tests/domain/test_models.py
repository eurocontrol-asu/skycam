"""Tests for domain models."""

import pytest
from pydantic import ValidationError

from skycam.domain.models import (
    CameraConfig,
    Position,
    ProjectionSettings,
)


class TestPosition:
    """Tests for Position model."""

    def test_defaults_to_ectl_coordinates(self) -> None:
        """Position defaults match ECTL Bretigny."""
        pos = Position()
        assert pos.longitude == pytest.approx(2.3467954996250784)
        assert pos.latitude == pytest.approx(48.600518087374105)
        assert pos.altitude == pytest.approx(90.0)

    def test_is_frozen(self) -> None:
        """Position is immutable."""
        pos = Position()
        with pytest.raises(ValidationError):
            pos.longitude = 0.0  # type: ignore[misc]

    def test_custom_values(self) -> None:
        """Position accepts custom values."""
        pos = Position(longitude=1.0, latitude=45.0, altitude=100.0)
        assert pos.longitude == 1.0
        assert pos.latitude == 45.0
        assert pos.altitude == 100.0


class TestProjectionSettings:
    """Tests for ProjectionSettings model."""

    def test_defaults_match_legacy(self) -> None:
        """Default values match legacy implementation."""
        settings = ProjectionSettings()
        assert settings.resolution == 1024
        assert settings.cloud_height == pytest.approx(10000.0)
        assert settings.square_size == pytest.approx(75000.0)
        assert settings.max_zenith_angle == pytest.approx(80.0)

    def test_is_frozen(self) -> None:
        """ProjectionSettings is immutable."""
        settings = ProjectionSettings()
        with pytest.raises(ValidationError):
            settings.resolution = 512  # type: ignore[misc]

    def test_validation_resolution_bounds(self) -> None:
        """Resolution must be within bounds."""
        with pytest.raises(ValidationError):
            ProjectionSettings(resolution=32)  # Below minimum

        with pytest.raises(ValidationError):
            ProjectionSettings(resolution=10000)  # Above maximum


class TestCameraConfig:
    """Tests for CameraConfig model."""

    def test_defaults(self) -> None:
        """CameraConfig has sensible defaults."""
        config = CameraConfig()
        assert config.name == "Bretigny Hemispherical Camera"
        assert config.location == "ECTL Bretigny"
        assert config.camera_type == "hemispherical"

    def test_is_frozen(self) -> None:
        """CameraConfig is immutable."""
        config = CameraConfig()
        with pytest.raises(ValidationError):
            config.name = "Modified"  # type: ignore[misc]
