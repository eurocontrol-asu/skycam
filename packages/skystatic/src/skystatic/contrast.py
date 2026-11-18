"""Marks that stay at the same place in the image while the sky moves behind them.

Dirt, drops and smears on the optics, water trails, thin rods and cables: in an average they are
local departures from the smooth sky around them, bright or dark, and the same dirt is dark against
a bright morning sky and bright when the sun lights it. The day is therefore cut into short windows
(an hour by default) and averaged in each, and a mark is what is present at the same place in most
of them, whatever its sign: a cloud, the sun's track, its glare or a reflection that travels with
it does not come back to the same pixel hour after hour. A mark exceeds ``k`` noise units in more
than half of the hours (``fixed_marks``) and is grown to half its mean significance (never below
``k``). Fainter marks, under ``k`` in most single hours, are found by hysteresis on the day and its
parts (``faint_marks``): weak but steady pixels attached to a mark, or a stronger faint mark alone.

Local contrast = log(luminance) - log(smooth sky), where the smooth sky is a quadratic surface
fitted around each pixel under a Gaussian window (normalised convolution), on sky pixels only: a
plane or a constant would leave a false band along the treeline, where the sky brightens towards
the horizon.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    correlate1d,
    find_objects,
    gaussian_filter,
    label,
    sobel,
)

from skystatic.masks import hysteresis

__all__ = [
    "FaintEvidence",
    "FaintInputs",
    "MarkEvidence",
    "SmoothSky",
    "band_pass",
    "faint_inputs",
    "faint_marks",
    "fixed_marks",
    "gradient_significance",
    "local_contrast",
    "log_contrast",
    "noise_by_zenith",
    "outline_votes",
    "window_significance",
]

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# Monomials x^a y^b of the quadratic surface, x and y in units of the window's sigma.
_BASIS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
# Pixels whose window holds less than this share of sky weight get no background (no contrast).
_MIN_SUPPORT = 0.05


def _filter(
    values: NDArray[np.float64], kx: NDArray[np.float64], ky: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Separable correlation: ``kx`` along rows (x), then ``ky`` along columns (y)."""
    rows = correlate1d(values, kx, axis=1, mode="constant")
    return np.asarray(correlate1d(rows, ky, axis=0, mode="constant"), dtype=np.float64)


class SmoothSky:
    """Quadratic sky surface fitted around every pixel, for one set of sky ``weights``.

    The moments of the weights are inverted once, then any image is fitted at the cost of six
    separable filters.
    """

    def __init__(self, weights: NDArray[np.bool_], sigma: float) -> None:
        radius = max(1, int(np.ceil(3 * sigma)))
        t = np.arange(-radius, radius + 1, dtype=np.float64) / sigma
        gauss = np.exp(-(t**2) / 2)
        gauss /= gauss.sum()
        self._kernels = [gauss * t**power for power in range(5)]
        w = weights.astype(np.float64)
        moments: dict[tuple[int, int], NDArray[np.float64]] = {}
        for a1, b1 in _BASIS:
            for a2, b2 in _BASIS:
                key = (a1 + a2, b1 + b2)
                if key not in moments:
                    moments[key] = _filter(w, self._kernels[key[0]], self._kernels[key[1]])
        matrix = np.stack(
            [np.stack([moments[(a1 + a2, b1 + b2)] for a2, b2 in _BASIS], -1) for a1, b1 in _BASIS],
            -2,
        )
        self.supported = moments[(0, 0)] >= _MIN_SUPPORT
        self._row = np.zeros((*weights.shape, len(_BASIS)), dtype=np.float64)
        ridge = 1e-9 * np.eye(len(_BASIS))
        self._row[self.supported] = np.linalg.inv(matrix[self.supported] + ridge)[:, 0, :]
        self._weights = w

    def fit(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        """Value of the fitted surface at each pixel (0 where unsupported)."""
        weighted = values.astype(np.float64) * self._weights
        out = np.zeros(values.shape, dtype=np.float64)
        for j, (a, b) in enumerate(_BASIS):
            out += self._row[..., j] * _filter(weighted, self._kernels[a], self._kernels[b])
        return out.astype(np.float32)


def local_contrast(image: NDArray[np.float32], sky: SmoothSky) -> NDArray[np.float32]:
    """log(luminance) minus its smooth-sky fit; 0 where the fit has no support."""
    luminance = np.log(np.maximum(image @ _LUMA, 0.5)).astype(np.float32)
    return np.where(sky.supported, luminance - sky.fit(luminance), 0).astype(np.float32)


def _by_zenith(
    values: NDArray[np.float32],
    where: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    band: float,
    statistic: Callable[[NDArray[np.float32]], float],
) -> NDArray[np.float32]:
    """``statistic`` of ``values`` over ``where`` in zenith bands of ``band`` degrees, interpolated
    between band centres to every pixel."""
    edges = np.arange(0.0, float(zenith[where].max()) + band, band)
    centres, levels = [], []
    for low, high in pairwise(edges):
        selected = values[where & (zenith >= low) & (zenith < high)]
        if selected.size < 100:  # too few sky pixels in the band for a robust estimate
            continue
        centres.append((low + high) / 2)
        levels.append(statistic(selected))
    if not levels:
        raise ValueError("No sky to measure the noise of the day")
    finite = np.where(np.isfinite(zenith), zenith, 0)
    return np.interp(finite, centres, levels).astype(np.float32)


def _robust_std(values: NDArray[np.float32]) -> float:
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def noise_by_zenith(
    contrast: NDArray[np.float32],
    where: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    band: float,
) -> NDArray[np.float32]:
    """Robust standard deviation (1.4826 MAD) of ``contrast`` over ``where``, per zenith band.

    The sky's own texture grows towards the horizon; it is measured in bands of ``band``
    degrees and interpolated between band centres.
    """
    return np.maximum(_by_zenith(contrast, where, zenith, band, _robust_std), 1e-6)


def gradient_significance(
    values: NDArray[np.float32],
    where: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    band: float,
) -> NDArray[np.float32]:
    """Gradient of a log image (Sobel after a 1-pixel blur) above the sky's own gradient at that
    zenith angle, in units of its robust spread: an outline stands out, the sky's texture not."""
    smooth = gaussian_filter(values, 1.0)
    gradient = np.hypot(sobel(smooth, 0), sobel(smooth, 1)).astype(np.float32)
    median = _by_zenith(gradient, where, zenith, band, lambda v: float(np.median(v)))
    spread = noise_by_zenith(gradient, where, zenith, band)
    return np.asarray((gradient - median) / spread, dtype=np.float32)


def log_contrast(values: NDArray[np.float32], sky: SmoothSky) -> NDArray[np.float32]:
    """A log image minus its smooth-sky fit; 0 where the fit has no support."""
    return np.where(sky.supported, values - sky.fit(values), 0).astype(np.float32)


def _grow(
    seeds: NDArray[np.bool_],
    strength: NDArray[np.float32],
    sky: NDArray[np.bool_],
    pad: int,
    floor: float,
) -> NDArray[np.bool_]:
    """Each seed component extended, over ``sky`` pixels, to where ``strength`` stays above half its
    peak and above ``floor`` (never into the noise), holes included (the flat centre of a soft
    blob)."""
    out = seeds.copy()
    components, _ = label(seeds, structure=np.ones((3, 3)))
    for index, box in enumerate(find_objects(components), start=1):
        if box is None:
            continue
        window = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in box)
        own = components[window] == index
        threshold = max(float(strength[window][own].max()) / 2, floor)
        candidate = ((strength[window] >= threshold) & sky[window]) | own
        regions, _ = label(candidate, structure=np.ones((3, 3)))
        kept = np.isin(regions, np.unique(regions[own]))
        out[window] |= binary_fill_holes(kept)
    return out


def window_significance(
    windows: NDArray[np.float32],
    surface: SmoothSky,
    where: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    band: float,
) -> NDArray[np.float32]:
    """|contrast| / noise of each window, (windows, height, width).

    ``windows`` holds, per window, the log of the mean luminance and the mean of the log
    luminance; the larger significance of the two is kept (either polarity).
    """
    out = np.zeros((windows.shape[0], *where.shape), dtype=np.float32)
    for index, pair in enumerate(windows):
        for values in pair:
            contrast = log_contrast(values, surface)
            noise = noise_by_zenith(contrast, where, zenith, band)
            np.maximum(out[index], np.abs(contrast) / noise, out=out[index])
    return out


def _signed_windows(
    windows: NDArray[np.float32],
    surface: SmoothSky,
    where: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    band: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per window, the signed significance and the signed contrast of the pixel (of the window's
    two log images, the one with the larger significance), (windows, height, width) each."""
    z = np.zeros((windows.shape[0], *where.shape), dtype=np.float32)
    c = np.zeros((windows.shape[0], *where.shape), dtype=np.float32)
    for index, pair in enumerate(windows):
        for values in pair:
            contrast = log_contrast(values, surface)
            significance = contrast / noise_by_zenith(contrast, where, zenith, band)
            larger = np.abs(significance) > np.abs(z[index])
            z[index] = np.where(larger, significance, z[index])
            c[index] = np.where(larger, contrast, c[index])
    return z, c


def _counted_hits(
    z: NDArray[np.float32], c: NDArray[np.float32], k: float, opposite: float
) -> NDArray[np.bool_]:
    """Windows that vote for a mark: significant (|z| >= ``k``) with the sign of the pixel's
    strongest window, or of the other sign at no less than ``opposite`` times its amplitude."""
    amplitude = np.abs(c)
    strongest = np.argmax(amplitude, axis=0)
    peak = np.take_along_axis(amplitude, strongest[None], 0)[0]
    sign = np.sign(np.take_along_axis(c, strongest[None], 0)[0])
    same = np.sign(c) == sign
    return np.asarray((np.abs(z) >= k) & (same | (amplitude >= opposite * peak)))


def outline_votes(
    windows: NDArray[np.float32],
    where: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    k: float,
    band: float,
    usable: NDArray[np.bool_] | None = None,
) -> NDArray[np.int_]:
    """Number of windows in which a pixel lies on an outline (gradient significance >= ``k``).

    A thin rod or the edge of a bright wall can have almost no contrast against the horizon sky
    for most of the day (dark in the morning, sunlit later), but its outline stays at the same
    pixels from hour to hour. Only outlines that touch what is already static are kept (see
    ``fixed_marks``).
    """
    votes = np.zeros(where.shape, dtype=np.int_)
    for index, pair in enumerate(windows):
        hit = np.zeros(where.shape, dtype=bool)
        for values in pair:
            hit |= gradient_significance(values, where, zenith, band) >= k
        if usable is not None:
            hit &= usable[index]
        votes += hit
    return votes


def _standing(outlines: NDArray[np.bool_], sky: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Outline components that touch something already static (or the edge of the analysed
    sky): a mast or a wall stands on the installation; a speck in the open sky does not."""
    components, count = label(outlines, structure=np.ones((3, 3)))
    if count == 0:
        return outlines
    touching = np.unique(components[binary_dilation(~sky) & outlines])
    return np.asarray(np.isin(components, touching[touching > 0]))


@dataclass(frozen=True)
class MarkEvidence:
    """Marks of a day and, per pixel, in how many judged windows it was significant (the
    evidence an installation vote can pool across reference days)."""

    marks: NDArray[np.bool_]
    hits: NDArray[np.uint8]
    windows: int
    chance: float  # share of sky pixels significant in a window: what noise gives
    weak: NDArray[np.bool_] | None = None  # faint but fixed all day (see ``faint_marks``)
    # voting windows of both signs (a mark that changes sign with the lighting, ``fixed_marks``)
    flips: NDArray[np.bool_] | None = None


def band_pass(
    values: NDArray[np.float32], weights: NDArray[np.bool_], inner: float, outer: float
) -> NDArray[np.float32]:
    """Gaussian mean at ``inner`` minus Gaussian mean at ``outer`` (sigmas, pixels), both taken
    over ``weights`` only (normalised convolution); 0 outside ``weights``.

    It keeps the scale of a drop or a thin line and removes the smooth sky and the slow cloud
    residue; static objects left out of ``weights`` leave no lobe beside them.
    """
    w = weights.astype(np.float32)
    v = np.where(weights, values, 0).astype(np.float32)

    def mean(sigma: float) -> NDArray[np.float32]:
        return np.asarray(
            gaussian_filter(v, sigma) / np.maximum(gaussian_filter(w, sigma), 1e-3),
            dtype=np.float32,
        )

    return np.where(weights, mean(inner) - mean(outer), 0).astype(np.float32)


def _part_means(
    windows: NDArray[np.float32], counts: NDArray[np.float64], parts: int
) -> list[NDArray[np.float32]]:
    """Mean log luminance of the whole day, then of ``parts`` consecutive groups of windows."""
    groups = [np.arange(len(counts)), *np.array_split(np.arange(len(counts)), parts)]
    return [
        (np.tensordot(counts[g], windows[g, 1].astype(np.float64), 1) / counts[g].sum()).astype(
            np.float32
        )
        for g in groups
    ]


@dataclass(frozen=True)
class FaintInputs:
    """Local contrast of the day's mean log luminance and of each part of the day, over the sky
    that is not static yet (``support``); what ``faint_marks`` decides on."""

    day: NDArray[np.float32]
    parts: tuple[NDArray[np.float32], ...]
    support: NDArray[np.bool_]
    seeds: NDArray[np.bool_]
    clear: NDArray[np.bool_] | None = None  # (parts, h, w): the sun stayed far all the part


def faint_inputs(
    windows: NDArray[np.float32],
    counts: Sequence[int],
    sky: NDArray[np.bool_],
    seeds: NDArray[np.bool_],
    sigma: float,
    parts: int = 3,
    usable: NDArray[np.bool_] | None = None,
    min_windows: int = 3,
) -> FaintInputs | None:
    """Contrast (``local_contrast``, against the sky minus ``seeds``) of the day and of ``parts``
    consecutive parts of it, and per part the pixels the sun never approached (all its windows
    ``usable``). None when a part would average fewer than ``min_windows`` windows."""
    if windows.shape[0] < min_windows * parts or parts < 2:
        return None
    means = _part_means(windows, np.asarray(counts, dtype=np.float64), parts)
    support = sky & ~seeds
    surface = SmoothSky(support, sigma)
    support = support & surface.supported
    contrasts = [log_contrast(m, surface) for m in means]
    groups = np.array_split(np.arange(windows.shape[0]), parts)
    clear = None if usable is None else np.stack([usable[g].all(axis=0) for g in groups])
    return FaintInputs(contrasts[0], tuple(contrasts[1:]), support, seeds.copy(), clear)


@dataclass(frozen=True)
class FaintEvidence:
    """Faint fixed structure of a day (see ``faint_marks``)."""

    marks: NDArray[np.bool_]  # seeds grown through the weak pixels, plus stand-alone faint marks
    weak: NDArray[np.bool_]  # faint and of the same sign in every part of the day


def faint_marks(
    inputs: FaintInputs,
    zenith: NDArray[np.float32],
    *,
    scales: tuple[float, float],
    weak_level: float,
    weak_k: float,
    alone_level: float,
    alone_k: float,
    central_zenith: float,
    band: float = 5.0,
    exclude: NDArray[np.bool_] | None = None,
) -> FaintEvidence:
    """Faint BRIGHT marks by hysteresis: two thresholds and connectivity.

    The contrasts of ``inputs`` are band-passed at the scale of a drop or a thin line
    (``scales``, ``band_pass``) over the sky that is not static yet, minus ``exclude`` (the
    ring lines of the optics: they are masked, and nothing is decided under them). The noise is
    TEMPORAL: the robust spread of the differences between parts, per zenith band, in which
    everything fixed cancels (a dense drop field would otherwise count as noise) and every cloud
    or glare that moves remains.

    - Weak: band-passed day contrast of at least ``weak_level`` (log units) and ``weak_k`` times
      the day's temporal noise, with the same sign in every part; or the same on two of the three
      parts, each of them at ``weak_k`` times its own noise.
    - Stand-alone: on a span of the day (the whole day, or any two of its three parts), at least
      ``alone_level`` and ``alone_k`` times the span's temporal noise, and in every part of the
      span the same sign at ``weak_k`` times that part's noise. A mark that stays for two thirds
      of the day, or is hidden by glare or cloud for one third of it, is static; the daily mask
      holds it all day. A span of two parts judges only pixels the sun never approached in them
      (``inputs.clear``): what is lit only near the sun is not judged.
    - A bright weak pixel is kept when it is connected through bright weak pixels to a bright
      seed (a mark of the day brighter than its sky, or a stand-alone mark): a trail grows from
      its head, a drop's faint skirt from its core.

    Only the bright sign: the band-pass of a bright drop is negative all around it (its side
    lobe), and between the drops of a field the lobes add up, so a dark pixel of the band-passed
    mean is not evidence of a dark mark (measured in round c8: dark faint marks had 18-44 % of
    their pieces darker than the open sky in the unfiltered day mean, bright ones 61-98 %). A
    dark mark (dirt against a bright sky) must be a mark hour by hour (``fixed_marks``).

    Both the growth and the stand-alone marks are confined within ``central_zenith`` of the
    zenith, where the sky is open and nothing slow stays put for hours; nearer the horizon haze
    and cloud banks do. The weak map of the whole day (all parts, either sign) is returned over
    the whole sky: the installation pools it over the reference days (thin, dark mast tops at
    the horizon).
    """
    support = inputs.support if exclude is None else inputs.support & ~exclude
    day = band_pass(inputs.day, support, *scales)
    part = [band_pass(p, support, *scales) for p in inputs.parts]
    pairs = [(i, j) for i in range(len(part)) for j in range(i + 1, len(part))]
    differences = [(part[i] - part[j]) / np.float32(np.sqrt(2)) for i, j in pairs]
    part_noise = np.mean([noise_by_zenith(d, support, zenith, band) for d in differences], axis=0)
    central = np.where(np.isfinite(zenith), zenith, np.inf) <= central_zenith
    # Spans of the day on which a mark must hold: the whole day (all parts), and every two of
    # its parts (a mark present, or visible, for most of the day but not all of it).
    everywhere = np.ones(support.shape, dtype=bool)
    spans = [(day, part, everywhere)] + (
        [
            (
                (part[i] + part[j]) / 2,
                [part[i], part[j]],
                everywhere if inputs.clear is None else inputs.clear[i] & inputs.clear[j],
            )
            for i, j in pairs
        ]
        if len(part) > 2
        else []
    )
    weak_all = np.zeros(support.shape, dtype=bool)
    weak = np.zeros(support.shape, dtype=bool)
    alone = np.zeros(support.shape, dtype=bool)
    for index, (mean, members, judged) in enumerate(spans):
        noise = part_noise / np.float32(np.sqrt(len(members)))
        sign = np.sign(mean)
        steady = np.all([np.sign(p) == sign for p in members], axis=0) & (sign != 0)
        steady &= support & judged
        each = np.all([np.abs(p) >= weak_k * part_noise for p in members], axis=0)
        faint = steady & (np.abs(mean) >= np.maximum(weak_level, weak_k * noise))
        if index > 0:  # on two parts, the sign test alone is a coin toss: each part must show it
            faint &= each
        strong = (
            steady & each & (np.abs(mean) >= np.maximum(alone_level, alone_k * noise)) & central
        )
        if index == 0:
            weak_all = faint
        weak |= faint & (sign > 0)
        alone |= strong & (sign > 0)
    seeds = (inputs.seeds & (inputs.day > 0)) | alone
    return FaintEvidence(hysteresis(seeds, weak & central), weak_all)


def fixed_marks(
    windows: NDArray[np.float32],
    sky: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    sigma: float,
    k: float,
    band: float = 5.0,
    usable: NDArray[np.bool_] | None = None,
    opposite: float = 0.0,
) -> MarkEvidence:
    """Marks present at the same place in most of the day's windows (module docstring).

    ``windows`` (windows, 2, height, width): log mean and mean log luminance of each short window
    of the day (hourly by default). ``sky``: pixels that may serve as smooth sky and hold marks
    (the domain minus what the edge test calls static all day); the marks returned are within it.
    Two passes: the marks found in the first are left out of the sky fitted in the second, so that
    a strong spot does not pull the surface around it and leave a false halo.

    A pixel is a mark when it exceeds ``k`` times the noise (either sign) in more than half of ALL
    the day's windows, as a spot (its own contrast) or on an outline that stays put and touches
    what is already static (thin masts). ``usable`` (windows, height, width): windows that may vote
    for a pixel (the sun stayed far from it); a pixel judged in fewer than half of the day's
    windows is left undecided. The per-pixel counts are returned for the installation vote, which
    pools them over all the reference days, and ``flips``: the pixels significant in windows of
    both signs, the other sign at no less than ``opposite`` times the strongest window's
    amplitude (dirt lit from the front then from behind changes sign at comparable amplitude).
    """
    count = windows.shape[0]
    empty = np.zeros(sky.shape, dtype=bool)
    if count < 3:
        return MarkEvidence(empty, np.zeros(sky.shape, np.uint8), count, 0.0)
    judged = np.full(sky.shape, count, dtype=np.int_) if usable is None else usable.sum(axis=0)
    majority = count // 2 + 1
    pad = int(np.ceil(2 * sigma))  # a mark wider than the window is sky gradient, not a mark
    decided = (judged >= max(3, (count + 1) // 2)) & sky
    support = sky
    marks = empty
    hits = np.zeros(sky.shape, dtype=np.uint8)
    flips = np.zeros(sky.shape, dtype=bool)
    chance = 0.0
    for _ in range(2):
        surface = SmoothSky(support, sigma)
        z, c = _signed_windows(windows, surface, support, zenith, band)
        if usable is not None:
            z = np.where(usable, z, 0)
            c = np.where(usable, c, 0)
        pixel = np.abs(z)
        strong = ((pixel >= k).sum(axis=0) >= majority) & decided & surface.supported
        strength = (pixel.sum(axis=0) / np.maximum(judged, 1)).astype(np.float32)
        marks = _grow(strong, strength, sky, pad, k)
        hits = np.minimum((pixel >= k).sum(axis=0), 255).astype(np.uint8)
        counted = _counted_hits(z, c, k, opposite)
        flips = (counted & (z > 0)).any(axis=0) & (counted & (z < 0)).any(axis=0)
        chance = float(np.mean([(w[support] >= k).mean() for w in pixel]))
        support = sky & ~marks
    outline_hits = outline_votes(windows, support, zenith, k, band, usable)
    marks = marks | _standing((outline_hits >= majority) & decided, sky)
    hits = np.maximum(hits, np.minimum(outline_hits, 255).astype(np.uint8))
    return MarkEvidence(np.asarray(marks), hits, count, chance, flips=np.asarray(flips))
