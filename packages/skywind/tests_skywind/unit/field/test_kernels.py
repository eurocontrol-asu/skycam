from __future__ import annotations

import numpy as np
import pytest

from skywind.field.kernels import correlation, correlation_and_grads, flow_coordinates


@pytest.mark.parametrize("kernel", ["gaussian", "matern32"])
def test_correlation_is_one_at_zero_and_decreasing(kernel: str) -> None:
    X = np.array([[0.0, 0, 10000, 0], [10.0, 0, 10000, 0], [40.0, 0, 10000, 0]])
    k = correlation(X[:1], X, (20.0, 300.0, 10.0), kernel)  # type: ignore[arg-type]
    assert k[0, 0] == pytest.approx(1.0)
    assert k[0, 0] > k[0, 1] > k[0, 2] > 0


@pytest.mark.parametrize("kernel", ["gaussian", "matern32"])
def test_length_gradients(kernel: str) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4)) * [30, 30, 300, 600]
    lengths = (25.0, 200.0, 7.0)
    _, grads = correlation_and_grads(X, X, lengths, kernel)  # type: ignore[arg-type]
    for j in range(3):
        up, down = list(lengths), list(lengths)
        up[j] *= np.exp(1e-6)
        down[j] *= np.exp(-1e-6)
        num = (correlation(X, X, tuple(up), kernel) - correlation(X, X, tuple(down), kernel)) / 2e-6  # type: ignore[arg-type]
        np.testing.assert_allclose(grads[j], num, atol=1e-6)


def test_flow_coordinates_move_with_wind_to_reference_time() -> None:
    X = np.array([[0.0, 0.0, 10000.0, 0.0]])
    out = flow_coordinates(X, np.array([[10.0, -5.0]]), 600.0)
    np.testing.assert_allclose(out[0, :2], [6.0, -3.0])
