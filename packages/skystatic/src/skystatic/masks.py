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
from scipy.ndimage import binary_dilation, find_objects, label

__all__ = [
    "CLASS_NAMES",
    "class_map",
    "enclosed_sky",
    "hysteresis",
    "installation_mask",
    "variable_mask",
]

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

    The slack absorbs the one or two pixels by which an edge moves between days: those pixels
    are dropped from a static piece that lies mostly within the slack (edge jitter). A piece that
    lies mostly beyond it (a mark touching the installation or crossing a ring line) is kept
    whole, its slack pixels included, so that no ring of sky is left inside the mark.
    """
    candidate = day_mask & ~installation
    if tolerance <= 0:
        return candidate
    slack = binary_dilation(installation, iterations=tolerance)
    components, count = label(candidate, structure=np.ones((3, 3)))
    total = np.bincount(components.ravel(), minlength=count + 1)
    beyond = np.bincount(components[~slack].ravel(), minlength=count + 1)
    whole = beyond >= total / 2
    whole[0] = False
    return np.asarray((candidate & ~slack) | whole[components])


def hysteresis(seeds: NDArray[np.bool_], weak: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Seeds plus the weak pixels connected to them through weak pixels (8-connectivity).

    Two thresholds and connectivity, as in an edge detector: a faint pixel alone cannot be
    trusted, a faint pixel attached to a sure one can.
    """
    components, count = label(seeds | weak, structure=np.ones((3, 3)))
    if count == 0:
        return seeds.copy()
    touched = np.zeros(count + 1, dtype=bool)
    touched[np.unique(components[seeds])] = True
    touched[0] = False
    return np.asarray(seeds | touched[components])


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


def enclosed_sky(
    classes: NDArray[np.uint8],
    core: NDArray[np.bool_],
    transparent: NDArray[np.bool_] | None = None,
) -> NDArray[np.uint8]:
    """Sky that is not connected to the open sky is not sky: every class-0 component other than
    the one holding most of ``core`` (the sky around the zenith) takes the class that surrounds it
    (installation or variable static, whichever borders it most). ``transparent`` pixels (the thin
    ring lines of the optics, which hide no sky behind them) connect the sky across them."""
    see_through = np.zeros(classes.shape, bool) if transparent is None else transparent
    sky = classes == 0
    components, count = label(sky | see_through, structure=np.ones((3, 3)))
    components[~sky] = 0
    if count <= 1:
        return classes
    main = int(np.bincount(components[core & sky], minlength=count + 1)[1:].argmax()) + 1
    out = classes.copy()
    for index, box in enumerate(find_objects(components), start=1):
        if index == main or box is None:
            continue
        window = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in box)
        own = components[window] == index
        border = classes[window][binary_dilation(own, structure=np.ones((3, 3), dtype=bool)) & ~own]
        fill = 2 if (border == 2).sum() > (border == 1).sum() else 1
        out[window][own] = fill
    return out
