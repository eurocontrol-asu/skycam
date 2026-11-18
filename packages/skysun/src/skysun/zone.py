"""The sun zone: the glow around the computed position of the sun, measured on each frame.

For every annulus of ``step_deg`` about the sun, the darker quarter of its sky pixels gives the
sky level there (clouds near the sun, brighter than the sky beside them, do not shape it). The
glow is that level above the sky's own level (the annuli 30-40 degrees from the sun) as a share
of the headroom up to saturation; the saturated share of the annulus is taken when larger. The
profile is forced non-increasing outward, smoothed, and faded to 0 across the reference annuli.
It is multiplied by whether the sun shines (read from the saturated share of the zone's core):
the zone is absent when the sun is hidden. Saturated pixels connected to the sun's disc (its
glare and star spikes) read 1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, label

from skysun.params import ZoneParams

__all__ = ["CoreState", "ZoneProfile", "core_state", "radial_profile", "sun_saturation", "zone_map"]

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
B = NDArray[np.bool_]

_LEVELS = 766  # sum of three 8-bit channels: 0..765


@dataclass(frozen=True)
class ZoneProfile:
    """The radial profile of one frame (annulus centres in degrees from the sun)."""

    centres: F64
    level: F64  # luminance (mean of RGB) at the quantile, NaN where too few pixels
    saturated: F64  # share of saturated pixels, NaN where too few pixels
    counts: NDArray[np.int64]
    saturated_counts: NDArray[np.int64]  # saturated pixels per annulus (all annuli)
    sky_level: float  # the sky's own level (reference annuli), NaN if nothing measured
    value: F64  # the zone value per annulus, after monotony, smoothing and fade

    def radius_at(self, level: float) -> float:
        """Angle (deg) where the value falls below ``level``; 0 if it never reaches it."""
        above = self.value >= level
        if not above.any():
            return 0.0
        below = np.flatnonzero(~above)
        return float(self.centres[below[0]]) if below.size else float(self.centres[-1])

    def glow(self, between: tuple[float, float]) -> float:
        """Mean zone value of the annuli whose centre lies between the two angles (deg)."""
        inner = (self.centres >= between[0]) & (self.centres < between[1])
        return float(self.value[inner].mean()) if inner.any() else 0.0

    def with_value(self, value: F64) -> ZoneProfile:
        """The same measurement with another value profile (the temporal median of a run)."""
        return replace(self, value=np.asarray(value, dtype=np.float64))

    def core_saturation(self, core_deg: float) -> float:
        """Saturated share of the measured pixels within ``core_deg`` of the sun (0 if none)."""
        inner = self.centres < core_deg
        total = int(self.counts[inner].sum())
        return float(self.saturated_counts[inner].sum() / total) if total else 0.0

    def saturation_radius(self, threshold: float = 0.01) -> float:
        """Outer edge (deg) of the last annulus, counted from the sun without a gap, whose
        saturated share exceeds ``threshold``; 0 if the first does not."""
        shares = np.nan_to_num(self.saturated)
        step = float(self.centres[1] - self.centres[0]) if self.centres.size > 1 else 0.0
        radius = 0.0
        for centre, share in zip(self.centres, shares, strict=True):
            if share <= threshold:
                break
            radius = float(centre) + step / 2
        return radius


@dataclass(frozen=True)
class CoreState:
    """The zone's core (field pixels within ``core_deg`` of the sun, static or not): the share
    that is saturated and the share that is static sky."""

    saturated: float
    sky: float


def core_state(
    rgb: NDArray[np.uint8], angle: F32, inside: B, sky: B, params: ZoneParams
) -> CoreState:
    """Read the core over every pixel of the field near the sun: a mast or a branch in front of
    the disc does not hide the saturated pixels around it (review S1)."""
    core = inside & (angle < params.core_deg)
    total = int(core.sum())
    if total == 0:
        return CoreState(0.0, 0.0)
    saturated = int((rgb[core].max(axis=1) >= params.saturation).sum())
    return CoreState(saturated / total, int((core & sky).sum()) / total)


def _quantiles(histogram: NDArray[np.int64], quantile: float) -> F64:
    """Per row of a (bins, levels) histogram: the quantile with linear interpolation (as
    ``numpy.percentile``); NaN for empty rows."""
    counts = histogram.sum(axis=1)
    cumulative = np.cumsum(histogram, axis=1)
    out = np.full(histogram.shape[0], np.nan)
    for row in np.flatnonzero(counts):
        position = quantile * (counts[row] - 1)
        low = int(np.floor(position))
        high = min(low + 1, int(counts[row]) - 1)
        v0 = int(np.searchsorted(cumulative[row], low, side="right"))
        v1 = int(np.searchsorted(cumulative[row], high, side="right"))
        out[row] = v0 + (position - low) * (v1 - v0)
    return out


def _fill(centres: F64, values: F64) -> F64:
    """Missing annuli interpolated from their neighbours (ends held)."""
    known = np.isfinite(values)
    return np.interp(centres, centres[known], values[known])


def _fade(centres: F64, fade: tuple[float, float]) -> F64:
    """1 inside ``fade[0]``, 0 beyond ``fade[1]``, a raised cosine between."""
    t = np.clip((centres - fade[0]) / (fade[1] - fade[0]), 0.0, 1.0)
    return np.asarray(0.5 * (1.0 + np.cos(np.pi * t)))


def radial_profile(
    rgb: NDArray[np.uint8], angle: F32, measured: B, params: ZoneParams
) -> ZoneProfile:
    """The zone profile of one frame: ``angle`` to the sun (deg) and the pixels ``measured``
    (sky, inside the field) of an (H, W, 3) frame."""
    bins = round(params.reach_deg / params.step_deg)
    centres = ((np.arange(bins) + 0.5) * params.step_deg).astype(np.float64)
    use = measured & (angle < params.reach_deg)
    index = np.minimum((angle[use] / params.step_deg).astype(np.int64), bins - 1)
    pixels = rgb[use]
    total = pixels.astype(np.int64).sum(axis=1)
    saturated = pixels.max(axis=1) >= params.saturation
    counts = np.bincount(index, minlength=bins)
    histogram = np.bincount(index * _LEVELS + total, minlength=bins * _LEVELS).reshape(
        bins, _LEVELS
    )
    enough = counts >= params.min_pixels
    level = np.where(enough, _quantiles(histogram, params.quantile) / 3, np.nan)
    saturated_counts = np.bincount(index, saturated, minlength=bins).astype(np.int64)
    share = np.where(enough, saturated_counts / np.maximum(counts, 1), np.nan)
    sky_level, value = _profile_value(centres, level, share, params)
    return ZoneProfile(centres, level, share, counts, saturated_counts, sky_level, value)


def _profile_value(centres: F64, level: F64, share: F64, params: ZoneParams) -> tuple[float, F64]:
    known = np.isfinite(level)
    if known.sum() < 4:  # nothing to measure a glow on (prototype: more than 3 annuli)
        return float("nan"), np.zeros(centres.size)
    reference = known & (centres >= params.reference_deg[0]) & (centres <= params.reference_deg[1])
    sky = float(np.median(level[reference] if reference.sum() >= 5 else level[known]))
    glow = np.clip((level - sky) / max(params.saturation - sky, 1.0), 0.0, 1.0)
    value = np.maximum(_fill(centres, glow), _fill(centres, share))
    value = np.maximum.accumulate(value[::-1])[::-1]  # non-increasing away from the sun
    if params.smoothing_bins > 0:
        value = gaussian_filter1d(value, params.smoothing_bins, mode="nearest")
    return sky, np.clip(value * _fade(centres, params.fade_deg), 0.0, 1.0)


def sun_saturation(rgb: NDArray[np.uint8], angle: F32, inside: B, params: ZoneParams) -> B:
    """Saturated pixels connected (8 neighbours) to a saturated pixel within ``sun_link_deg``
    of the sun: the sun's disc, its glare and its star spikes."""
    saturated = (rgb.max(axis=-1) >= params.saturation) & inside
    seeds = saturated & (angle < params.sun_link_deg)
    if not seeds.any():
        return np.zeros(angle.shape, dtype=bool)
    components, count = label(saturated, structure=np.ones((3, 3), dtype=bool))
    keep = np.zeros(count + 1, dtype=bool)
    keep[components[seeds]] = True
    keep[0] = False
    return np.asarray(keep[components])


def zone_map(
    profile: ZoneProfile, angle: F32, sun_saturated: B, inside: B, visibility: float = 1.0
) -> F32:
    """The zone value of every pixel: the profile at its angle to the sun times whether the sun
    shines (``visibility``, 0..1: the zone is absent when the sun is hidden), 1 on the sun's
    saturated pixels, 0 outside the field."""
    value = np.interp(angle, profile.centres, profile.value, right=0.0).astype(np.float32)
    value *= np.float32(np.clip(visibility, 0.0, 1.0))
    value[sun_saturated] = 1.0
    value[~inside] = 0.0
    return value
