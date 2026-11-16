"""Sky versus static in an averaged image.

In an average over many frames the sky is smooth and every fixed object keeps sharp edges. The
sky is the connected smooth region that holds most of the pixels around the zenith; all the
rest of the analysed domain is static.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, label, median_filter, sobel, zoom

__all__ = ["edge_strength", "sky_region", "static_mask", "static_mask_all_parts"]

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _local_median(values: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    """Median over a ``window`` x ``window`` neighbourhood, computed at reduced resolution."""
    step = max(1, window // 9)
    reduced = median_filter(values[::step, ::step], size=max(3, window // step) | 1)
    factors = (values.shape[0] / reduced.shape[0], values.shape[1] / reduced.shape[1])
    full = zoom(reduced, factors, order=1)
    return np.asarray(full[: values.shape[0], : values.shape[1]], dtype=np.float32)


def edge_strength(
    image: NDArray[np.float32],
    core: NDArray[np.bool_],
    floor: float = 0.1,
    sigma: float = 1.0,
    background: int = 36,
) -> NDArray[np.float32]:
    """How much sharper a pixel is than its surroundings.

    The gradient is taken on the log image, after dividing by the median level in ``core``
    (the sky around the zenith), so it does not depend on exposure; ``floor`` bounds the
    amplification of the darkest pixels, in units of that sky level. The largest value over
    the colour channels is kept.

    The median gradient over a ``background`` x ``background`` window is then subtracted. An
    edge is a local excess: the steady brightening of the sky towards the horizon, strong on
    short exposures, and the residue of slow clouds are gradients too, but spread out.
    ``background=0`` returns the raw gradient.
    """
    level = float(np.median(image[core] @ _LUMA))
    if level <= 0:
        raise ValueError("The sky around the zenith is black; no daytime content to analyse")
    strength = np.zeros(image.shape[:2], dtype=np.float32)
    for channel in range(image.shape[2]):
        smooth = gaussian_filter(np.log(image[..., channel] / level + floor), sigma)
        strength = np.maximum(strength, np.hypot(sobel(smooth, 0), sobel(smooth, 1)))
    if background > 0:
        strength = np.maximum(strength - _local_median(strength, background), 0)
    return strength


def sky_region(smooth: NDArray[np.bool_], core: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Connected component of ``smooth`` that covers the most pixels of ``core``."""
    regions, count = label(smooth)
    if count == 0:
        return np.zeros(smooth.shape, dtype=bool)
    sizes = np.bincount(regions[core].ravel(), minlength=count + 1)
    sizes[0] = 0
    if sizes.max() == 0:
        return np.zeros(smooth.shape, dtype=bool)
    return np.asarray(regions == sizes.argmax())


def static_mask(
    image: NDArray[np.float32],
    domain: NDArray[np.bool_],
    core: NDArray[np.bool_],
    edge: float,
    background: int = 36,
) -> NDArray[np.bool_]:
    """Pixels of ``domain`` that are not the smooth sky around the zenith."""
    smooth = domain & (edge_strength(image, core, background=background) < edge)
    return domain & ~sky_region(smooth, core)


def static_mask_all_parts(
    parts: NDArray[np.float32],
    domain: NDArray[np.bool_],
    core: NDArray[np.bool_],
    edge: float,
    background: int = 36,
) -> NDArray[np.bool_]:
    """Static in every part of the day.

    A slow cloud bank can survive the average of one part; it does not sit at the same place in
    all of them, whereas a fixed object does.
    """
    masks = [static_mask(part, domain, core, edge, background) for part in parts]
    return np.asarray(np.logical_and.reduce(masks))
