"""Tests for calibration adapter."""

from pathlib import Path

import numpy as np
import pytest

from skycam.adapters.calibration import JP2CalibrationLoader
from skycam.domain.exceptions import CalibrationError
from skycam.domain.models import CalibrationData


class TestJP2CalibrationLoader:
    """Tests for JP2CalibrationLoader."""

    def test_loader_returns_calibration_data(
        self,
        calibration_loader: JP2CalibrationLoader,
    ) -> None:
        """load() returns CalibrationData."""
        data = calibration_loader.load("visible")
        assert isinstance(data, CalibrationData)

    def test_calibration_has_azimuth_array(
        self,
        calibration_loader: JP2CalibrationLoader,
    ) -> None:
        """CalibrationData contains azimuth array."""
        data = calibration_loader.load("visible")
        azimuth = np.asarray(data.azimuth_array)
        assert isinstance(azimuth, np.ndarray)
        assert azimuth.dtype == np.float64

    def test_calibration_has_zenith_array(
        self,
        calibration_loader: JP2CalibrationLoader,
    ) -> None:
        """CalibrationData contains zenith array."""
        data = calibration_loader.load("visible")
        zenith = np.asarray(data.zenith_array)
        assert isinstance(zenith, np.ndarray)
        assert zenith.dtype == np.float64

    def test_azimuth_range(
        self,
        calibration_loader: JP2CalibrationLoader,
    ) -> None:
        """Azimuth values are in valid range [-π, π]."""
        data = calibration_loader.load("visible")
        azimuth = np.asarray(data.azimuth_array)
        assert azimuth.min() >= -np.pi
        assert azimuth.max() <= np.pi

    def test_zenith_range(
        self,
        calibration_loader: JP2CalibrationLoader,
    ) -> None:
        """Zenith values are in valid range [0, ~π/2].

        Note: Edge pixels may slightly exceed π/2 due to
        calibration data encoding quirks.
        """
        data = calibration_loader.load("visible")
        zenith = np.asarray(data.zenith_array)
        assert zenith.min() >= 0
        # Allow 5% tolerance for edge cases
        assert zenith.max() <= np.pi / 2 * 1.05

    def test_image_size_matches_arrays(
        self,
        calibration_loader: JP2CalibrationLoader,
    ) -> None:
        """image_size matches array dimensions."""
        data = calibration_loader.load("visible")
        azimuth = np.asarray(data.azimuth_array)
        assert data.image_size == azimuth.shape

    def test_missing_file_raises_error(
        self,
        fixtures_path: Path,
    ) -> None:
        """Missing calibration file raises CalibrationError."""
        loader = JP2CalibrationLoader(fixtures_path / "nonexistent")
        with pytest.raises(CalibrationError, match="not found"):
            loader.load("visible")
