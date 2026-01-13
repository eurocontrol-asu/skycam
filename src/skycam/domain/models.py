"""Domain models for skycam camera projection."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Position(BaseModel, frozen=True):
    """Geographic position of the camera installation.

    Default values correspond to ECTL Bretigny coordinates.
    """

    longitude: float = Field(
        default=2.3467954996250784,
        description="Longitude in decimal degrees (WGS84)",
    )
    latitude: float = Field(
        default=48.600518087374105,
        description="Latitude in decimal degrees (WGS84)",
    )
    altitude: float = Field(
        default=90.0,
        ge=0,
        description="Altitude above ground in meters",
    )


class CameraConfig(BaseModel, frozen=True):
    """Camera metadata and installation information."""

    name: str = Field(
        default="Bretigny Hemispherical Camera",
        description="Human-readable camera name",
    )
    location: str = Field(
        default="ECTL Bretigny",
        description="Installation location",
    )
    camera_type: Literal["hemispherical", "fisheye", "standard"] = Field(
        default="hemispherical",
        description="Type of camera lens",
    )
    installation_date: date | None = Field(
        default=None,
        description="Date of camera installation",
    )


class ProjectionSettings(BaseModel, frozen=True):
    """Settings for the projection algorithm.

    These values control the output grid and filtering.
    Default values match the legacy implementation exactly.
    """

    resolution: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Output grid resolution in pixels",
    )
    cloud_height: float = Field(
        default=10000.0,
        ge=100,
        description="Assumed cloud height in meters",
    )
    square_size: float = Field(
        default=75000.0,
        ge=1000,
        description="Physical size of output grid in meters",
    )
    max_zenith_angle: float = Field(
        default=80.0,
        ge=0,
        le=90,
        description="Maximum zenith angle in degrees for valid data",
    )


class CalibrationData(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """Container for loaded calibration arrays.

    Uses model_config to allow numpy arrays.
    """

    # Note: numpy arrays stored via 'arbitrary_types_allowed'
    # Type hints are for documentation; actual validation is minimal
    azimuth_array: object = Field(
        ...,
        description="Azimuth calibration map in radians [-π, π]",
    )
    zenith_array: object = Field(
        ...,
        description="Zenith calibration map in radians [0, π/2]",
    )
    image_size: tuple[int, int] = Field(
        ...,
        description="Size of calibration arrays (height, width)",
    )
