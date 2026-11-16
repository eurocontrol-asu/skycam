"""Application settings using Pydantic Settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from skycam.domain.models import CameraConfig, Position, ProjectionSettings


class SkycamSettings(BaseSettings):
    """Main application settings with environment/file support.

    Loads configuration from:
    1. Environment variables (prefixed with SKYCAM_)
    2. .env file (if present)
    3. Default values from domain models
    """

    model_config = SettingsConfigDict(
        env_prefix="SKYCAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Directory paths
    calibration_dir: Path = Field(
        default=Path("calibration"),
        description="Directory containing calibration JP2 files",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for input/output data",
    )

    # Camera category
    category: str = Field(
        default="visible",
        description="Camera category (visible, infrarouge)",
    )

    # Nested configuration (use default factories)
    position: Position = Field(default_factory=Position)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    projection: ProjectionSettings = Field(default_factory=ProjectionSettings)
