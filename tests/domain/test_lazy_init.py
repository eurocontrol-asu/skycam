"""Tests for ProjectionService lazy initialization.

These tests verify that:
1. ProjectionService can be created without immediate interpolator build
2. Interpolators are built on-demand when project() is called
3. ensure_initialized() can be called explicitly

Uses session-scoped calibration from conftest.py for performance.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from skycam.domain.models import CalibrationData, ProjectionSettings
from skycam.domain.projection import ProjectionService


class TestLazyInitialization:
    """Tests for lazy initialization behavior.

    Uses session-scoped calibration_data_session from conftest.py.
    """

    def test_lazy_init_skips_interpolator_build(
        self,
        calibration_data_session: CalibrationData,
    ) -> None:
        """ProjectionService with lazy_init=True skips interpolator build."""
        settings = ProjectionSettings(resolution=64)

        # Create with lazy init - should NOT build interpolators (instant)
        service = ProjectionService(
            calibration=calibration_data_session,
            settings=settings,
            lazy_init=True,
        )

        # Interpolator should be None (not built)
        assert service._azimuth_zenith_to_pixel_raw is None

    @pytest.mark.slow
    def test_eager_init_builds_interpolator(
        self,
        calibration_data_session: CalibrationData,
    ) -> None:
        """ProjectionService with lazy_init=False builds interpolators immediately."""
        settings = ProjectionSettings(resolution=64)

        service = ProjectionService(
            calibration=calibration_data_session,
            settings=settings,
            lazy_init=False,
        )

        # Interpolator should be built
        assert service._azimuth_zenith_to_pixel_raw is not None

    @pytest.mark.slow
    def test_default_is_eager_init(
        self,
        calibration_data_session: CalibrationData,
    ) -> None:
        """Default behavior is eager initialization (backward compatibility)."""
        settings = ProjectionSettings(resolution=64)

        # Default behavior - should build interpolators
        service = ProjectionService(
            calibration=calibration_data_session,
            settings=settings,
        )

        # Interpolator should be built (backward compatible)
        assert service._azimuth_zenith_to_pixel_raw is not None

    @pytest.mark.slow
    def test_ensure_initialized_builds_interpolator(
        self,
        calibration_data_session: CalibrationData,
    ) -> None:
        """ensure_initialized() builds interpolators on demand."""
        settings = ProjectionSettings(resolution=64)

        service = ProjectionService(
            calibration=calibration_data_session,
            settings=settings,
            lazy_init=True,
        )

        # Before: not initialized
        assert service._azimuth_zenith_to_pixel_raw is None

        # Call ensure_initialized
        service.ensure_initialized()

        # After: initialized
        assert service._azimuth_zenith_to_pixel_raw is not None

    @pytest.mark.slow
    def test_project_triggers_lazy_init(
        self,
        calibration_data_session: CalibrationData,
    ) -> None:
        """project() automatically initializes if needed."""
        settings = ProjectionSettings(resolution=64)

        service = ProjectionService(
            calibration=calibration_data_session,
            settings=settings,
            lazy_init=True,
        )

        # Before: not initialized
        assert service._azimuth_zenith_to_pixel_raw is None

        # Create a test image matching expected input size
        h, w = calibration_data_session.image_size
        test_image: NDArray[np.uint8] = np.zeros((h, w, 3), dtype=np.uint8)

        # Project should trigger initialization
        service.project(test_image)

        # After: initialized
        assert service._azimuth_zenith_to_pixel_raw is not None
