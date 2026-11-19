"""Where the sensor does not see the sky: the value taken from the measured sky around it.

One mechanism for every such place (``unseen_sky="fill"``): the sun's saturated core and the
other unmeasured pixels near the sun, and the variable static of the day (skystatic class 2:
drops, dirt on the dome). It runs after every measurement, on the final value and confidence:
the pixels it fills are never read by the cues, the level fit, the certification or any
statistic (they stay outside the valid sky there), and only measured pixels are its sources.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["fill_holes"]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]


def _ring_fill(
    sub: B, valid: B, alpha: F32, conf: F32, widths: tuple[int, ...], min_px: int
) -> tuple[float, float] | None:
    """Median value and confidence of the measured ring just outside ``sub`` (the first width
    with at least ``min_px`` measured pixels); None when none has enough."""
    for r in widths:
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ring = (cv2.dilate(sub.astype(np.uint8), disc) > 0) & ~sub & valid
        if int(ring.sum()) >= min_px:
            return float(np.median(alpha[ring])), float(np.median(conf[ring]))
    return None


def fill_holes(
    alpha: F32,
    conf: F32,
    valid: B,
    hole: B,
    widths: tuple[int, ...],
    min_px: int,
    core: B | None = None,
) -> dict[str, Any]:
    """In place: each 8-connected component of ``hole`` takes the median value and the median
    confidence of the MEASURED pixels (``valid``: sky that is neither static, saturated nor the
    sun's core) of the ring just outside it, ``widths[0]`` px wide, widened to the next width
    when that ring holds fewer than ``min_px`` measured pixels (a ring partly behind the
    skyline or the installation uses its measured part). A component whose widest ring holds
    too few measured pixels keeps no value. One constant per component: faithful for small
    marks, a coarse estimate for large ones (README). ``core`` (the sun's saturated pixels)
    only feeds the diagnostics. Returns pixel counts."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    pad = max(widths) + 1
    h, w = hole.shape
    filled = left = 0
    core_value: float | None = None
    for i in range(1, n):
        x, y, ww, hh, a = (int(v) for v in st[i])
        y0, y1 = max(y - pad, 0), min(y + hh + pad, h)
        x0, x1 = max(x - pad, 0), min(x + ww + pad, w)
        sub = lab[y0:y1, x0:x1] == i
        crop = (slice(y0, y1), slice(x0, x1))
        got = _ring_fill(sub, valid[crop], alpha[crop], conf[crop], widths, min_px)
        if got is None:
            left += a
            continue
        alpha[crop][sub], conf[crop][sub] = got
        filled += a
        if core is not None and core_value is None and (core[crop] & sub).any():
            core_value = got[0]
    return {
        "holes": n - 1,
        "filled_px": filled,
        "unfilled_px": left,
        "core_value": None if core_value is None else round(core_value, 4),
    }
