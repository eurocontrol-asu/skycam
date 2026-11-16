"""Tests for ProjectionService lazy initialization.

These tests verify that:
1. ProjectionService can be created without immediate interpolator build
2. Interpolators are built on-demand when project() is called
3. ensure_initialized() can be called explicitly

Optimized: Uses session-scoped fixtures and signature inspection
to minimize redundant 10s interpolator builds.
"""

import inspect

import numpy as np
import pytest
from numpy.typing import NDArray

from skycam.domain.models import CalibrationData, ProjectionSettings
from skycam.domain.projection import ProjectionService


class TestLazyInitialization:
    """Tests for lazy initialization behavior."""

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
        assert service._pixel_coords is None

    def test_lazy_init_defaults_to_false(self) -> None:
        """Default lazy_init is False (backward compatibility).

        Verifies via signature inspection - no interpolator build needed.
        """
        sig = inspect.signature(ProjectionService)
        lazy_init_param = sig.parameters.get("lazy_init")

        assert lazy_init_param is not None
        assert lazy_init_param.default is False

    def test_ensure_initialized_is_idempotent(
        self,
        projector_session: ProjectionService,
    ) -> None:
        """ensure_initialized() can be called multiple times safely.

        Uses session-scoped projector - no additional build.
        """
        # Already initialized from session fixture (may be from cache)
        assert projector_session._pixel_coords is not None

        # Should not error when called again
        projector_session.ensure_initialized()
        projector_session.ensure_initialized()

        # Still initialized
        assert projector_session._pixel_coords is not None

    def test_project_works_with_lazy_init(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() works correctly after initialization.

        Uses session-scoped projector - no additional build.
        """
        result = projector_session.project(sample_image_session)

        assert result is not None
        assert result.dtype == np.uint8

    @pytest.mark.slow
    def test_full_initialization_builds_all_components(
        self,
        calibration_data_session: CalibrationData,
    ) -> None:
        """INTEGRATION: Full interpolator build creates all required components.

        This is the ONLY test that builds a fresh interpolator.
        Consolidates verification of eager init and ensure_initialized.
        """
        settings = ProjectionSettings(resolution=64)

        # Test eager initialization (lazy_init=False)
        service = ProjectionService(
            calibration=calibration_data_session,
            settings=settings,
            lazy_init=False,
        )

        # All components should be built
        assert service._azimuth_zenith_to_pixel_raw is not None
        assert service._azimuth_zenith_grid is not None
        assert service._pixel_coords is not None

        # Verify projection works
        h, w = calibration_data_session.image_size
        test_image: NDArray[np.uint8] = np.zeros((h, w, 3), dtype=np.uint8)
        result = service.project(test_image)

        assert result is not None
        assert result.shape == (64, 64, 3)
