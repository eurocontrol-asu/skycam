"""Domain layer: Pure business logic with no I/O dependencies."""

from skycam.domain.aircraft_projection import (
    AircraftProjectionSettings,
    AircraftProjector,
)
from skycam.domain.exceptions import (
    CalibrationError,
    ConfigurationError,
    ProjectionError,
    SkycamError,
)
from skycam.domain.models import (
    CameraConfig,
    Position,
    ProjectionSettings,
)

__all__ = [
    "AircraftProjectionSettings",
    "AircraftProjector",
    "CalibrationError",
    "CameraConfig",
    "ConfigurationError",
    "Position",
    "ProjectionError",
    "ProjectionSettings",
    "SkycamError",
]
