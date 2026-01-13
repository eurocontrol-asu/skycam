"""Domain exceptions for skycam."""


class SkycamError(Exception):
    """Base exception for all skycam errors."""


class CalibrationError(SkycamError):
    """Raise when calibration data cannot be loaded or is invalid."""


class ProjectionError(SkycamError):
    """Raise when projection calculation fails."""


class ConfigurationError(SkycamError):
    """Raise when configuration is invalid or missing."""
