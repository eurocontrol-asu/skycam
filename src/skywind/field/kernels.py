"""Separable space-time correlation functions (horizontal distance x altitude x time) and their length-scale gradients.

Coordinates are (east km, north km, pressure altitude m, epoch s). In the flow-following variant the horizontal
coordinates are first displaced with the background (ERA5) wind to a reference time ``t0``: residual structures are then
assumed to move with the flow instead of staying fixed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from skywind.geo import FloatArray

__all__ = ["Kernel", "Lengths", "correlation", "correlation_and_grads", "flow_coordinates"]

type Kernel = Literal["gaussian", "matern32"]
type Lengths = tuple[float, float, float]
_SQ3 = np.sqrt(3.0)


def flow_coordinates(X: FloatArray, bg: FloatArray, t0: float) -> FloatArray:
    """Horizontal position moved with the background wind (m/s) from the point time to ``t0``."""
    out = X.copy()
    dt = t0 - X[:, 3]
    out[:, 0] += bg[:, 0] * dt / 1000.0
    out[:, 1] += bg[:, 1] * dt / 1000.0
    return out


def _scaled(XA: FloatArray, XB: FloatArray, lengths: Lengths) -> tuple[FloatArray, FloatArray, FloatArray]:
    lh, lz, lt = lengths
    dh2 = ((XA[:, None, 0] - XB[None, :, 0]) ** 2 + (XA[:, None, 1] - XB[None, :, 1]) ** 2) / lh**2
    dz2 = (XA[:, None, 2] - XB[None, :, 2]) ** 2 / lz**2
    dt2 = (XA[:, None, 3] - XB[None, :, 3]) ** 2 / (lt * 60.0) ** 2
    return dh2, dz2, dt2


def _matern(r2: FloatArray) -> FloatArray:
    r = _SQ3 * np.sqrt(r2)
    return (1 + r) * np.exp(-r)


def _matern_dlog(r2: FloatArray) -> FloatArray:
    """d k / d log(length) for the Matern 3/2 correlation."""
    return 3 * r2 * np.exp(-_SQ3 * np.sqrt(r2))


def correlation(XA: FloatArray, XB: FloatArray, lengths: Lengths, kernel: Kernel = "gaussian") -> FloatArray:
    """Correlation matrix (len(XA), len(XB)); lengths = (km, m, min)."""
    d = _scaled(XA, XB, lengths)
    if kernel == "gaussian":
        return np.exp(-0.5 * (d[0] + d[1] + d[2]))
    return _matern(d[0]) * _matern(d[1]) * _matern(d[2])


def correlation_and_grads(
    XA: FloatArray, XB: FloatArray, lengths: Lengths, kernel: Kernel = "gaussian"
) -> tuple[FloatArray, list[FloatArray]]:
    """Correlation and its derivatives with respect to log(length_h), log(length_z), log(length_t)."""
    d = _scaled(XA, XB, lengths)
    if kernel == "gaussian":
        k = np.exp(-0.5 * (d[0] + d[1] + d[2]))
        return k, [k * d[0], k * d[1], k * d[2]]
    parts = [_matern(x) for x in d]
    grads = [_matern_dlog(x) for x in d]
    k = parts[0] * parts[1] * parts[2]
    return k, [grads[0] * parts[1] * parts[2], parts[0] * grads[1] * parts[2], parts[0] * parts[1] * grads[2]]
