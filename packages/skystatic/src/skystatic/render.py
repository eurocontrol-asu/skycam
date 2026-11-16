"""Pictures of the result, for a human to check the masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion

__all__ = ["CLASS_COLOURS", "overlay"]

CLASS_COLOURS: dict[int, tuple[int, int, int]] = {1: (255, 30, 60), 2: (255, 230, 0)}


def overlay(
    image: NDArray[np.float32], classes: NDArray[np.uint8], gain: float | None = None
) -> NDArray[np.uint8]:
    """Averaged image with the installation tinted, the variable static filled, outside dimmed.

    ``gain`` brightens the picture; by default the sky is brought to mid-grey, so that dark
    exposures stay readable.
    """
    if gain is None:
        sky = image[classes == 0]
        gain = 110.0 / max(float(np.median(sky)) if sky.size else 110.0, 1.0)
    picture = np.clip(image * gain, 0, 255)
    installation = classes == 1
    tint = np.array(CLASS_COLOURS[1], dtype=np.float32)
    picture[installation] = picture[installation] * 0.5 + tint * 0.5
    variable = classes == 2
    picture[variable] = CLASS_COLOURS[2]
    picture[classes == 3] *= 0.3
    outline = installation & ~binary_erosion(installation)
    picture[outline] = CLASS_COLOURS[1]
    return np.asarray(picture, dtype=np.uint8)
