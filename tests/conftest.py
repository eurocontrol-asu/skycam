"""Shared pytest fixtures for skycam tests.

This module provides both session-scoped fixtures for fast test execution
and function-scoped fixtures for isolation where needed.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from skycam.adapters.calibration import JP2CalibrationLoader
from skycam.adapters.image_io import load_jp2
from skycam.domain.models import CalibrationData, ProjectionSettings
from skycam.domain.projection import ProjectionService

# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped fixtures (computed once per test session for speed)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def fixtures_path_session() -> Path:
    """Return path to test fixtures directory (session-scoped)."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def calibration_path_session(fixtures_path_session: Path) -> Path:
    """Return path to calibration fixtures (session-scoped)."""
    return fixtures_path_session / "calibration"


@pytest.fixture(scope="session")
def calibration_data_session(calibration_path_session: Path) -> CalibrationData:
    """Load calibration data once per session (expensive JP2 loading)."""
    loader = JP2CalibrationLoader(calibration_path_session)
    return loader.load("visible")


@pytest.fixture(scope="session")
def projector_session(
    calibration_data_session: CalibrationData,
    calibration_path_session: Path,
) -> ProjectionService:
    """Session-cached projector with interpolators built once.

    Use this fixture for tests that actually need projection to work.
    The interpolator build happens once, then cached to disk.
    """
    settings = ProjectionSettings()
    return ProjectionService(
        calibration=calibration_data_session,
        settings=settings,
        calibration_path=calibration_path_session,  # Enable disk caching
        lazy_init=False,  # Build immediately, cached for session
    )


@pytest.fixture(scope="session")
def sample_image_session(fixtures_path_session: Path) -> NDArray[np.uint8]:
    """Load a sample input image once per session."""
    return load_jp2(fixtures_path_session / "gold_inputs" / "image_20250215080830.jp2")


# ─────────────────────────────────────────────────────────────────────────────
# Function-scoped fixtures (for test isolation, backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────


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
def projector(
    calibration_data_session: CalibrationData,
    calibration_path: Path,
) -> ProjectionService:
    """Create a ProjectionService for testing.

    Uses session-cached calibration data but creates fresh service instance.
    Builds interpolators immediately (backward compatible behavior).
    """
    settings = ProjectionSettings()
    return ProjectionService(
        calibration=calibration_data_session,
        settings=settings,
        calibration_path=calibration_path,  # Enable disk caching
        lazy_init=False,
    )


@pytest.fixture
def sample_image(gold_inputs_path: Path) -> NDArray[np.uint8]:
    """Load a sample input image for testing."""
    return load_jp2(gold_inputs_path / "image_20250215080830.jp2")
