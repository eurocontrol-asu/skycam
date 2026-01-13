"""Domain exceptions for skycam."""


class SkycamError(Exception):
    """Base exception for all skycam errors."""

    pass


class CalibrationError(SkycamError):
    """Raised when calibration data cannot be loaded or is invalid."""

    pass


class ProjectionError(SkycamError):
    """Raised when projection calculation fails."""

    pass


class ConfigurationError(SkycamError):
    """Raised when configuration is invalid or missing."""

    pass
