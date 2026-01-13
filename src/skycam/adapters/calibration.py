"""Calibration data loading adapter for JP2 files."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from skycam.domain.exceptions import CalibrationError
from skycam.domain.models import CalibrationData


class JP2CalibrationLoader:
    """Loads JP2 calibration files and converts to radians.

    The calibration files contain azimuth and zenith maps encoded as
    uint16 values in the range [0, 64000]. These are converted to
    radians using the legacy formulas.
    """

    def __init__(self, calibration_dir: Path) -> None:
        """Initialize loader with calibration directory path.

        Args:
            calibration_dir: Directory containing azimuth_*.jp2 and zenith_*.jp2 files
        """
        self.calibration_dir = Path(calibration_dir)

    def load(self, category: str = "visible") -> CalibrationData:
        """Load calibration data for the specified camera category.

        Args:
            category: Camera category (e.g., "visible", "infrarouge")

        Returns:
            CalibrationData with azimuth and zenith arrays in radians

        Raises:
            CalibrationError: If calibration files cannot be loaded
        """
        azimuth_path = self.calibration_dir / f"azimuth_{category}.jp2"
        zenith_path = self.calibration_dir / f"zenith_{category}.jp2"

        azimuth_array = self._load_and_convert_azimuth(azimuth_path)
        zenith_array = self._load_and_convert_zenith(zenith_path)

        if azimuth_array.shape != zenith_array.shape:
            msg = (
                f"Calibration array shape mismatch: "
                f"azimuth {azimuth_array.shape} vs zenith {zenith_array.shape}"
            )
            raise CalibrationError(msg)

        return CalibrationData(
            azimuth_array=azimuth_array,
            zenith_array=zenith_array,
            image_size=(azimuth_array.shape[0], azimuth_array.shape[1]),
        )

    def _load_jp2(self, path: Path) -> NDArray[np.uint16]:
        """Load JP2 file as uint16 array.

        Args:
            path: Path to JP2 file

        Returns:
            Raw pixel values as uint16 array

        Raises:
            CalibrationError: If file cannot be loaded
        """
        if not path.exists():
            msg = f"Calibration file not found: {path}"
            raise CalibrationError(msg)

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            msg = f"Failed to load calibration file: {path}"
            raise CalibrationError(msg)

        return img.astype(np.uint16)

    def _load_and_convert_azimuth(self, path: Path) -> NDArray[np.float64]:
        """Load and convert azimuth calibration to radians.

        Formula: 360 * π/180 * raw / 64000 - π
        Result range: [-π, π]

        Args:
            path: Path to azimuth JP2 file

        Returns:
            Azimuth values in radians
        """
        raw = self._load_jp2(path)
        # Convert to radians: full circle (360°) mapped to [0, 64000] → [-π, π]
        return 360 * np.pi / 180 * raw.astype(np.float64) / 64000 - np.pi

    def _load_and_convert_zenith(self, path: Path) -> NDArray[np.float64]:
        """Load and convert zenith calibration to radians.

        Formula: 90 * π/180 * raw / 64000
        Result range: [0, π/2]

        Args:
            path: Path to zenith JP2 file

        Returns:
            Zenith values in radians
        """
        raw = self._load_jp2(path)
        # Convert to radians: quarter circle (90°) mapped to [0, 64000] → [0, π/2]
        return 90 * np.pi / 180 * raw.astype(np.float64) / 64000
