"""Tests for projection service."""

import numpy as np
from numpy.typing import NDArray

from skycam.domain.projection import ProjectionService


class TestProjectionService:
    """Tests for ProjectionService.

    Uses session-scoped projector to avoid rebuilding interpolators.
    """

    def test_project_returns_ndarray(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() returns a numpy array."""
        result = projector_session.project(sample_image_session)
        assert isinstance(result, np.ndarray)

    def test_project_output_dtype(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() returns uint8 by default."""
        result = projector_session.project(sample_image_session)
        assert result.dtype == np.uint8

    def test_project_output_shape(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() produces correct output dimensions."""
        settings = projector_session.settings
        result = projector_session.project(sample_image_session)

        assert result.shape[0] == settings.resolution
        assert result.shape[1] == settings.resolution
        # RGB channels
        assert result.shape[2] == 3

    def test_project_float_output(
        self,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """project() can return float64 when requested."""
        result = projector_session.project(sample_image_session, as_uint8=False)
        assert result.dtype == np.float64
