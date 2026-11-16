"""Tests for domain exceptions."""

from skycam.domain.exceptions import (
    CalibrationError,
    ConfigurationError,
    ProjectionError,
    SkycamError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_calibration_error_is_skycam_error(self) -> None:
        """CalibrationError inherits from SkycamError."""
        err = CalibrationError("test")
        assert isinstance(err, SkycamError)
        assert isinstance(err, Exception)

    def test_projection_error_is_skycam_error(self) -> None:
        """ProjectionError inherits from SkycamError."""
        err = ProjectionError("test")
        assert isinstance(err, SkycamError)
        assert isinstance(err, Exception)

    def test_configuration_error_is_skycam_error(self) -> None:
        """ConfigurationError inherits from SkycamError."""
        err = ConfigurationError("test")
        assert isinstance(err, SkycamError)
        assert isinstance(err, Exception)


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_calibration_error_message(self) -> None:
        """CalibrationError preserves message."""
        err = CalibrationError("Missing azimuth file")
        assert "azimuth" in str(err)

    def test_projection_error_message(self) -> None:
        """ProjectionError preserves message."""
        err = ProjectionError("Interpolation failed")
        assert "Interpolation" in str(err)

    def test_configuration_error_message(self) -> None:
        """ConfigurationError preserves message."""
        err = ConfigurationError("Invalid resolution")
        assert "resolution" in str(err)
