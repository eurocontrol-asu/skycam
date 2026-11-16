"""From per-day static masks to installation and variable static.

The installation (trees, buildings, masts, antennas) is static on nearly every reference day,
provided those days span a period longer than the life of anything on the optics. What a day
adds to it is that day's variable static: dirt, drops, anything that stays put for a day but
not for the whole reference period.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

__all__ = ["CLASS_NAMES", "class_map", "installation_mask", "variable_mask"]

CLASS_NAMES = {0: "sky", 1: "installation", 2: "variable_static", 3: "outside"}


def installation_mask(
    day_masks: Sequence[NDArray[np.bool_]], min_fraction: float
) -> NDArray[np.bool_]:
    """Pixels static on at least ``min_fraction`` of the reference days."""
    if not day_masks:
        raise ValueError("At least one reference day is required")
    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must be in (0, 1]")
    votes = np.sum(np.stack(day_masks), axis=0)
    needed = max(1, int(np.ceil(min_fraction * len(day_masks))))
    return np.asarray(votes >= needed)


def variable_mask(
    day_mask: NDArray[np.bool_], installation: NDArray[np.bool_], tolerance: int
) -> NDArray[np.bool_]:
    """Static on the day but not part of the installation, with ``tolerance`` pixels of slack.

    The slack absorbs the one or two pixels by which an edge moves between days.
    """
    slack = binary_dilation(installation, iterations=tolerance) if tolerance > 0 else installation
    return day_mask & ~slack


def class_map(
    domain: NDArray[np.bool_],
    installation: NDArray[np.bool_],
    variable: NDArray[np.bool_] | None = None,
) -> NDArray[np.uint8]:
    """One label per pixel, see ``CLASS_NAMES``."""
    classes = np.full(domain.shape, 3, dtype=np.uint8)
    classes[domain] = 0
    classes[domain & installation] = 1
    if variable is not None:
        classes[domain & variable] = 2
    return classes
