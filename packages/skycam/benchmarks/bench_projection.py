"""Projection performance benchmarks.

Run benchmarks with:
    uv run pytest benchmarks/ --benchmark-only

Compare against baseline:
    uv run pytest benchmarks/ --benchmark-compare
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from skycam.domain.models import CalibrationData, ProjectionSettings
from skycam.domain.projection import ProjectionService


@pytest.mark.benchmark(group="projection")
class TestProjectionBenchmarks:
    """Benchmarks for projection operations."""

    def test_projection_throughput(
        self,
        benchmark: pytest.fixture,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """Measure projection throughput (images/second).

        This benchmarks the core projection operation using pre-built
        interpolators (realistic production scenario).
        """
        result = benchmark(projector_session.project, sample_image_session)
        assert result.dtype == np.uint8

    def test_projection_float_output(
        self,
        benchmark: pytest.fixture,
        projector_session: ProjectionService,
        sample_image_session: NDArray[np.uint8],
    ) -> None:
        """Measure float64 projection (higher precision path)."""
        result = benchmark(
            projector_session.project,
            sample_image_session,
            as_uint8=False,
        )
        assert result.dtype == np.float64


@pytest.mark.benchmark(group="initialization")
class TestInitializationBenchmarks:
    """Benchmarks for service initialization (interpolator build)."""

    def test_interpolator_build_time(
        self,
        benchmark: pytest.fixture,
        calibration_data_session: CalibrationData,
    ) -> None:
        """Measure time to build interpolators from calibration data.

        This is the expensive operation that happens once per service
        instantiation (unless lazy_init=True is used).
        """
        settings = ProjectionSettings(resolution=512)

        def create_service() -> ProjectionService:
            return ProjectionService(
                calibration=calibration_data_session,
                settings=settings,
                lazy_init=False,
            )

        result = benchmark(create_service)
        assert result._azimuth_zenith_to_pixel_raw is not None
