"""Tests for Numba-accelerated interpolation functions.

Note:
    First test run incurs JIT compilation (~2-3s).
    Subsequent runs are fast.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from skycam.domain.interpolation import bilinear_sample, bilinear_sample_grayscale


class TestBilinearSample:
    """Tests for bilinear_sample (3-channel images)."""

    @pytest.fixture
    def sample_image(self) -> NDArray[np.uint8]:
        """Create a simple 4x4 RGB test image with known values."""
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        # Set corners to distinct colors for easy verification
        image[0, 0] = [100, 0, 0]  # Top-left: red
        image[0, 3] = [0, 100, 0]  # Top-right: green
        image[3, 0] = [0, 0, 100]  # Bottom-left: blue
        image[3, 3] = [100, 100, 100]  # Bottom-right: gray
        # Fill a 2x2 region with solid white for interpolation testing
        image[1:3, 1:3] = [200, 200, 200]
        return image

    def test_exact_pixel_coordinates(self, sample_image: NDArray[np.uint8]) -> None:
        """Sampling at exact integer coordinates returns pixel values."""
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
        result = bilinear_sample(sample_image, coords)

        np.testing.assert_array_almost_equal(result[0], [100.0, 0.0, 0.0])

    def test_interpolation_between_pixels(
        self, sample_image: NDArray[np.uint8]
    ) -> None:
        """Sampling between pixels returns interpolated values."""
        # Sample in the middle of the white 2x2 region
        coords = np.array([[1.5, 1.5]], dtype=np.float64)
        result = bilinear_sample(sample_image, coords)

        # Should get approximately white (200, 200, 200)
        np.testing.assert_array_almost_equal(result[0], [200.0, 200.0, 200.0])

    def test_out_of_bounds_returns_zeros(self, sample_image: NDArray[np.uint8]) -> None:
        """Coordinates outside image bounds return zeros."""
        coords = np.array(
            [
                [-1.0, 0.0],  # Above image
                [0.0, -1.0],  # Left of image
                [5.0, 0.0],  # Below image
                [0.0, 5.0],  # Right of image
            ],
            dtype=np.float64,
        )
        result = bilinear_sample(sample_image, coords)

        # All out-of-bounds should be zeros
        np.testing.assert_array_equal(result, np.zeros((4, 3)))

    def test_boundary_edge_cases(self, sample_image: NDArray[np.uint8]) -> None:
        """Test behavior at exact boundary coordinates."""
        h, w = sample_image.shape[:2]
        coords = np.array(
            [
                [0.0, 0.0],  # Valid: top-left corner
                [h - 1.0, w - 1.0],  # Invalid: at boundary (needs +1 for interp)
            ],
            dtype=np.float64,
        )
        result = bilinear_sample(sample_image, coords)

        # First should have values, second should be zero (at boundary)
        assert result[0].sum() > 0
        np.testing.assert_array_equal(result[1], [0.0, 0.0, 0.0])

    def test_multiple_coordinates_parallel(
        self, sample_image: NDArray[np.uint8]
    ) -> None:
        """Test parallel processing with many coordinates."""
        # Generate grid of coordinates
        rows = np.linspace(0.5, 2.0, 10)
        cols = np.linspace(0.5, 2.0, 10)
        coords = np.array(
            [[r, c] for r in rows for c in cols],
            dtype=np.float64,
        )
        result = bilinear_sample(sample_image, coords)

        assert result.shape == (100, 3)
        # All points should have non-zero values (inside valid region)
        assert np.all(result.sum(axis=1) > 0)


class TestBilinearSampleGrayscale:
    """Tests for bilinear_sample_grayscale (single-channel images)."""

    @pytest.fixture
    def grayscale_image(self) -> NDArray[np.uint8]:
        """Create a simple 4x4 grayscale test image."""
        image = np.zeros((4, 4), dtype=np.uint8)
        # Create a gradient pattern
        image[0, 0] = 0
        image[0, 3] = 100
        image[3, 0] = 100
        image[3, 3] = 200
        # Fill center with known value
        image[1:3, 1:3] = 150
        return image

    def test_exact_pixel_coordinates(self, grayscale_image: NDArray[np.uint8]) -> None:
        """Sampling at exact integer coordinates returns pixel values."""
        coords = np.array([[1.0, 1.0]], dtype=np.float64)
        result = bilinear_sample_grayscale(grayscale_image, coords)

        np.testing.assert_almost_equal(result[0], 150.0)

    def test_interpolation_between_pixels(
        self, grayscale_image: NDArray[np.uint8]
    ) -> None:
        """Sampling between pixels returns interpolated values."""
        coords = np.array([[1.5, 1.5]], dtype=np.float64)
        result = bilinear_sample_grayscale(grayscale_image, coords)

        np.testing.assert_almost_equal(result[0], 150.0)

    def test_out_of_bounds_returns_zeros(
        self, grayscale_image: NDArray[np.uint8]
    ) -> None:
        """Coordinates outside image bounds return zeros."""
        coords = np.array(
            [
                [-1.0, 0.0],
                [0.0, -1.0],
                [5.0, 0.0],
                [0.0, 5.0],
            ],
            dtype=np.float64,
        )
        result = bilinear_sample_grayscale(grayscale_image, coords)

        np.testing.assert_array_equal(result, np.zeros(4))

    def test_returns_1d_array(self, grayscale_image: NDArray[np.uint8]) -> None:
        """Result should be 1D array matching number of input coordinates."""
        coords = np.array([[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]], dtype=np.float64)
        result = bilinear_sample_grayscale(grayscale_image, coords)

        assert result.shape == (3,)
        assert result.dtype == np.float64
