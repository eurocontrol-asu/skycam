"""Shared pytest fixtures for skycam tests."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from skycam.adapters.calibration import JP2CalibrationLoader
from skycam.adapters.image_io import load_jp2
from skycam.domain.models import ProjectionSettings
from skycam.domain.projection import ProjectionService


@pytest.fixture
def fixtures_path() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def calibration_path(fixtures_path: Path) -> Path:
    """Return path to calibration fixtures."""
    return fixtures_path / "calibration"


@pytest.fixture
def gold_inputs_path(fixtures_path: Path) -> Path:
    """Return path to golden input fixtures."""
    return fixtures_path / "gold_inputs"


@pytest.fixture
def gold_outputs_path(fixtures_path: Path) -> Path:
    """Return path to golden output fixtures."""
    return fixtures_path / "gold_outputs"


@pytest.fixture
def calibration_loader(calibration_path: Path) -> JP2CalibrationLoader:
    """Create a calibration loader with test fixtures."""
    return JP2CalibrationLoader(calibration_path)


@pytest.fixture
def projector(calibration_loader: JP2CalibrationLoader) -> ProjectionService:
    """Create a ProjectionService with default settings."""
    calibration = calibration_loader.load("visible")
    settings = ProjectionSettings()
    return ProjectionService(calibration=calibration, settings=settings)


@pytest.fixture
def sample_image(gold_inputs_path: Path) -> NDArray[np.uint8]:
    """Load a sample input image for testing."""
    return load_jp2(gold_inputs_path / "image_20250215080830.jp2")
