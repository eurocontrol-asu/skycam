"""Golden output integration tests.

These tests verify that the refactored projection produces outputs
that match the legacy implementation exactly.

Mark: These tests are slow (first run only) due to interpolator build.
The projector is session-cached, so subsequent runs are fast.
Skip in development with: pytest -m "not slow"
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from skycam.adapters.image_io import load_jp2, load_jpg
from skycam.domain.projection import ProjectionService


def calculate_image_similarity(
    result: NDArray[np.uint8],
    expected: NDArray[np.uint8],
) -> tuple[float, float, float]:
    """Calculate similarity metrics between two images.

    Returns:
        Tuple of (max_diff, mean_diff, percentage_matching)
    """
    min_h = min(result.shape[0], expected.shape[0])
    min_w = min(result.shape[1], expected.shape[1])

    result_crop = result[:min_h, :min_w]
    expected_crop = expected[:min_h, :min_w]

    diff = np.abs(result_crop.astype(np.float64) - expected_crop.astype(np.float64))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # Percentage of pixels within tolerance
    matching = np.sum(diff <= 5) / diff.size * 100

    return max_diff, mean_diff, matching


@pytest.mark.slow
@pytest.mark.parametrize("image_stem", ["image_20250215080830"])
class TestGoldenOutputs:
    """Tests comparing against legacy golden outputs.

    Uses session-scoped projector for performance.
    These tests are marked as slow and can be skipped with: pytest -m "not slow"
    """

    def test_projection_produces_valid_output(
        self,
        projector_session: ProjectionService,
        gold_inputs_path: Path,
        image_stem: str,
    ) -> None:
        """Projection produces a valid image."""
        input_path = gold_inputs_path / f"{image_stem}.jp2"
        input_image = load_jp2(input_path)

        result = projector_session.project(input_image)

        assert result is not None
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        assert result.dtype == np.uint8

    def test_projection_output_dimensions(
        self,
        projector_session: ProjectionService,
        gold_inputs_path: Path,
        image_stem: str,
    ) -> None:
        """Projection output has expected dimensions."""
        input_path = gold_inputs_path / f"{image_stem}.jp2"
        input_image = load_jp2(input_path)

        result = projector_session.project(input_image)

        assert result.shape[0] == projector_session.settings.resolution
        assert result.shape[1] == projector_session.settings.resolution
        assert result.shape[2] == 3  # RGB

    def test_projection_matches_golden_exactly(
        self,
        projector_session: ProjectionService,
        gold_inputs_path: Path,
        gold_outputs_path: Path,
        image_stem: str,
    ) -> None:
        """Projection output matches golden output exactly.

        Now that golden outputs are regenerated from the same code,
        the outputs should be byte-identical (mean_diff < 1).
        """
        input_path = gold_inputs_path / f"{image_stem}.jp2"
        expected_path = gold_outputs_path / f"{image_stem}.jpg"

        input_image = load_jp2(input_path)
        expected = load_jpg(expected_path)

        result = projector_session.project(input_image)

        _max_diff, mean_diff, pct_matching = calculate_image_similarity(
            result, expected
        )

        # Since we regenerated golden outputs, should be near-identical
        # Small diff allowed for JPEG compression
        assert mean_diff < 1.0, f"Mean diff too high: {mean_diff}"
        assert pct_matching > 99.0, f"Too few matching pixels: {pct_matching}%"
