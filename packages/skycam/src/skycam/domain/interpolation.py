"""Numba-accelerated interpolation functions.

This module provides JIT-compiled interpolation for fast image sampling,
replacing scipy's RegularGridInterpolator for significant speedup.

Note:
    First call incurs JIT compilation overhead (~2-3s).
    Subsequent calls are fast (<50ms for typical projections).
    Use cache=True to persist compiled code between sessions.
"""

import numpy as np
from numba import jit, prange
from numpy.typing import NDArray


@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def bilinear_sample(
    image: NDArray[np.uint8],
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sample image at floating-point coordinates using bilinear interpolation.

    This is a Numba-accelerated replacement for scipy RegularGridInterpolator.
    Provides ~10x speedup for typical projection operations.

    Args:
        image: (H, W, C) uint8 input image
        coords: (N, 2) float64 coordinates as (row, col) pairs

    Returns:
        (N, C) float64 sampled values
    """
    n_points = coords.shape[0]
    n_channels = image.shape[2]
    result = np.zeros((n_points, n_channels), dtype=np.float64)

    h, w = image.shape[0], image.shape[1]

    for i in prange(n_points):
        r = coords[i, 0]
        c = coords[i, 1]

        # Bounds check - fill with 0 if out of bounds
        if r < 0 or r >= h - 1 or c < 0 or c >= w - 1:
            continue

        # Integer pixel coordinates
        r0 = int(r)
        c0 = int(c)
        r1 = r0 + 1
        c1 = c0 + 1

        # Bilinear weights
        dr = r - r0
        dc = c - c0

        # Interpolate each channel
        for ch in range(n_channels):
            v00 = image[r0, c0, ch]
            v01 = image[r0, c1, ch]
            v10 = image[r1, c0, ch]
            v11 = image[r1, c1, ch]

            result[i, ch] = (
                v00 * (1 - dr) * (1 - dc)
                + v01 * (1 - dr) * dc
                + v10 * dr * (1 - dc)
                + v11 * dr * dc
            )

    return result


@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def bilinear_sample_grayscale(
    image: NDArray[np.uint8],
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sample grayscale image at floating-point coordinates.

    Optimized version for single-channel images.

    Args:
        image: (H, W) uint8 input image
        coords: (N, 2) float64 coordinates as (row, col) pairs

    Returns:
        (N,) float64 sampled values
    """
    n_points = coords.shape[0]
    result = np.zeros(n_points, dtype=np.float64)

    h, w = image.shape[0], image.shape[1]

    for i in prange(n_points):
        r = coords[i, 0]
        c = coords[i, 1]

        if r < 0 or r >= h - 1 or c < 0 or c >= w - 1:
            continue

        r0 = int(r)
        c0 = int(c)
        r1 = r0 + 1
        c1 = c0 + 1

        dr = r - r0
        dc = c - c0

        v00 = image[r0, c0]
        v01 = image[r0, c1]
        v10 = image[r1, c0]
        v11 = image[r1, c1]

        result[i] = (
            v00 * (1 - dr) * (1 - dc)
            + v01 * (1 - dr) * dc
            + v10 * dr * (1 - dc)
            + v11 * dr * dc
        )

    return result
