"""Permanent reflections of the optics: faint rings around a point near the zenith.

Some cameras show thin bright or dark circles (reflections inside the dome or the lens). They stay
at the same place all day and for years, with a contrast of a few tenths of a percent, below the
noise of a single pixel of a day's average. They are found where that noise averages out: in the
mean local contrast of the reference days, averaged around circles. The common centre is the
point that makes that radial profile sharpest. The reflections are not exactly concentric: each
bright ring is a circle with its own centre (within a few pixels of the common one) and radius,
the circle along which the mean contrast peaks (``ring_circles``). The circles do not change
from day to day; the light along them is the sun's (reflected in the optics), so the masks write
them as sky by default and describe them in ``optics.json`` (``StaticConfig.ring_class``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.ndimage import maximum_filter, median
from scipy.optimize import least_squares

__all__ = [
    "RingFit",
    "circle_bands",
    "concentric_rings",
    "ring_circles",
    "ring_mask",
    "without_rings",
]

# A radius bin is used when it holds at least this many sky pixels (robust median).
_MIN_PIXELS = 20
# A ring is a circle: its sign must hold in more than half of these azimuth sectors (a dirt spot
# that happens to sit on one radius shows in one sector only).
_SECTORS = 8
# Own circle of a ring: its maximum is measured in these azimuth sectors (15 degrees each) and must
# be found in at least a third of them. A ring that merges with a neighbouring one over part of its
# circle (measured: one ring of the site camera over about a third) still shows its own maximum in
# the other sectors; a spot or a short arc shows in one or two.
_OWN_SECTORS = 24


class RingFit(BaseModel):
    """Fitted centre (x, y pixels of the working image) and masked rings."""

    centre: tuple[float, float]
    noise: float
    # (inner radius, outer radius, peak contrast): pixels, log units
    rings: list[tuple[float, float, float]]
    eye: float = 0.0  # radius of the bright central disc ("eye") masked with the rings, 0 if none
    # Own centre (x, y) of each ring, in the order of ``rings``; empty: every ring about ``centre``
    centres: list[tuple[float, float]] = []

    def circles(self) -> list[tuple[tuple[float, float], float]]:
        """(own centre, radius) of every bright line (lines mode)."""
        return [
            (self.centres[i] if self.centres else self.centre, inner)
            for i, (inner, _, _) in enumerate(self.rings)
        ]


def _profile(
    contrast: NDArray[np.float32], sky: NDArray[np.bool_], centre: tuple[float, float]
) -> NDArray[np.float64]:
    """Median contrast per integer radius around ``centre`` (NaN where too few sky pixels)."""
    rows, cols = np.indices(contrast.shape)
    radius = np.rint(np.hypot(cols - centre[0], rows - centre[1])).astype(int)[sky]
    values = contrast[sky]
    order = np.argsort(radius, kind="stable")
    radius, values = radius[order], values[order]
    edges = np.searchsorted(radius, np.arange(int(radius.max()) + 2))
    out = np.full(int(radius.max()) + 1, np.nan)
    for r in range(out.size):
        if edges[r + 1] - edges[r] >= _MIN_PIXELS:
            out[r] = np.median(values[edges[r] : edges[r + 1]])
    return out


def _sharpness(profile: NDArray[np.float64]) -> float:
    finite = profile[np.isfinite(profile)]
    return float(np.std(finite)) if finite.size else 0.0


def _robust_std(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0


def _rings(profile: NDArray[np.float64], level: float) -> list[tuple[float, float, float]]:
    """Radii above ``level``, each extended to half its peak with the same sign."""
    values = np.nan_to_num(profile)
    taken = np.zeros(values.size, dtype=bool)
    rings: list[tuple[float, float, float]] = []
    for peak in np.argsort(-np.abs(values)):
        if abs(values[peak]) < level:
            break
        if taken[peak]:
            continue
        half, sign = abs(values[peak]) / 2, np.sign(values[peak])
        inner = outer = int(peak)
        while inner > 0 and sign * values[inner - 1] >= half and not taken[inner - 1]:
            inner -= 1
        while outer < values.size - 1 and sign * values[outer + 1] >= half and not taken[outer + 1]:
            outer += 1
        taken[inner : outer + 1] = True
        rings.append((float(inner), float(outer), float(values[peak])))
    return sorted(rings)


def _around(
    contrast: NDArray[np.float32],
    sky: NDArray[np.bool_],
    radius: NDArray[np.int_],
    sector: NDArray[np.int_],
    ring: tuple[float, float, float],
) -> int:
    """Number of azimuth sectors in which the ring shows with its sign."""
    inner, outer, peak = ring
    band = sky & (radius >= round(inner)) & (radius <= round(outer))
    count = 0
    for index in range(_SECTORS):
        values = contrast[band & (sector == index)]
        if values.size >= _MIN_PIXELS and np.sign(np.median(values)) == np.sign(peak):
            count += 1
    return count


def _fit_centre(
    contrast: NDArray[np.float32], sky: NDArray[np.bool_], around: tuple[float, float], search: int
) -> tuple[float, float]:
    """Centre that makes the radial profile sharpest: coarse grid, then half-pixel steps."""
    best = (-1.0, around)
    for step, span in ((max(1, search // 10), search), (0.5, max(1, search // 10))):
        start = best[1]
        offsets = np.arange(-span, span + step / 2, step)
        for dy in offsets:
            for dx in offsets:
                centre = (start[0] + float(dx), start[1] + float(dy))
                score = _sharpness(_profile(contrast, sky, centre))
                if score > best[0]:
                    best = (score, centre)
    return best[1]


def _bright_lines(profile: NDArray[np.float64], level: float) -> list[tuple[float, float, float]]:
    """Local maxima of the profile above ``level`` (bright lines), as (radius, radius, value);
    the radius is refined to a fraction of a pixel by a parabola through the three bins."""
    values = np.nan_to_num(profile)
    peaks = (values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:]) & (values[1:-1] >= level)
    lines = []
    for index in np.nonzero(peaks)[0] + 1:
        before, at, after = values[index - 1], values[index], values[index + 1]
        curvature = before - 2 * at + after
        shift = 0.5 * (before - after) / curvature if curvature < 0 else 0.0
        radius = float(index + np.clip(shift, -0.5, 0.5))
        lines.append((radius, radius, float(at)))
    return lines


def concentric_rings(
    contrast: NDArray[np.float32],
    sky: NDArray[np.bool_],
    around: tuple[float, float],
    search: int,
    level: float,
    halfwidth: float = 0.0,
    k: float = 2.0,
    own: float = 0.0,
) -> tuple[NDArray[np.bool_], RingFit]:
    """Mask of the permanent rings and their fit.

    ``contrast``: mean local contrast of the reference days (log units); ``sky``: pixels where
    rings are looked for (the sky near the zenith, without the installation); ``around``: start
    of the centre search (the zenith pixel), searched over +-``search`` pixels.

    Two ways to mask, the user's choice:
    - ``halfwidth`` > 0 (lines): only the bright lines, each as its own circle (``ring_circles``,
      its centre within +-``own`` pixels of the common one), masked over +-``halfwidth`` pixels
      of it, and the bright "eye" at the centre;
    - ``halfwidth`` = 0 (level): every line, bright or dark, down to the contrast ``level`` (and
      ``k`` times the noise), masked over the radii where it stays above half its peak.
    In level mode a ring must also show with its sign in more than half of the azimuth sectors.
    """
    centre = _fit_centre(contrast, sky, around, search)
    profile = _profile(contrast, sky, centre)
    noise = _robust_std(profile)
    rows, cols = np.indices(contrast.shape)
    distance = np.hypot(cols - centre[0], rows - centre[1])
    radius = np.rint(distance).astype(int)
    eye = _eye(contrast, sky, radius, k * noise)
    mask = distance <= eye
    if halfwidth > 0:
        circles = ring_circles(contrast, sky, centre, own, k, halfwidth, eye)
        mask |= circle_bands([(c, r) for c, r, _ in circles], contrast.shape, halfwidth)
        return mask, RingFit(
            centre=(float(centre[0]), float(centre[1])),
            noise=noise,
            rings=[(r, r, peak) for _, r, peak in circles],
            eye=eye,
            centres=[c for c, _, _ in circles],
        )
    sector = (
        (np.arctan2(rows - centre[1], cols - centre[0]) + np.pi) / (2 * np.pi) * _SECTORS
    ).astype(int) % _SECTORS
    rings = []
    for ring in _rings(profile, max(level, k * noise)):
        if _around(contrast, sky, radius, sector, ring) > _SECTORS / 2:
            mask |= (distance >= ring[0] - 0.5) & (distance <= ring[1] + 0.5)
            rings.append(ring)
    return mask, RingFit(
        centre=(float(centre[0]), float(centre[1])), noise=noise, rings=rings, eye=eye
    )


def _circle_means(
    contrast: NDArray[np.float32],
    sky: NDArray[np.bool_],
    centre: tuple[float, float],
    offsets: NDArray[np.float64],
    step: float,
    reach: float,
) -> NDArray[np.float64]:
    """Mean contrast per ``step`` of radius about every centre ``centre`` + (dx, dy), dx and dy in
    ``offsets``: (len(offsets), len(offsets), radius bins); NaN where a bin holds too few pixels."""
    rows, cols = np.nonzero(sky)
    keep = np.hypot(cols - centre[0], rows - centre[1]) < reach
    rows, cols = rows[keep], cols[keep]
    values = contrast[rows, cols].astype(np.float64)
    bins = int(reach / step)
    out = np.full((offsets.size, offsets.size, bins), np.nan)
    for i, dy in enumerate(offsets):
        for j, dx in enumerate(offsets):
            b = (np.hypot(cols - centre[0] - dx, rows - centre[1] - dy) / step).astype(int)
            ok = b < bins
            n = np.bincount(b[ok], minlength=bins)
            total = np.bincount(b[ok], values[ok], minlength=bins)
            out[i, j] = np.where(n >= 2 * _MIN_PIXELS, total / np.maximum(n, 1), np.nan)
    return out


def _circle_residual(q: NDArray[np.float64], xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance of each point to the circle (radius, x, y) = ``q``, minus its radius."""
    return np.asarray(np.hypot(xy[:, 0] - q[1], xy[:, 1] - q[2]) - q[0])


def _own_circle(
    contrast: NDArray[np.float32],
    sky: NDArray[np.bool_],
    others: NDArray[np.bool_],
    centre: tuple[float, float],
    radius: float,
    window: float,
    step: float = 0.5,
) -> tuple[tuple[float, float], float, int]:
    """A ring's own circle: in each of ``_OWN_SECTORS`` sectors about the current centre, the
    radius of the maximum of the median contrast within +-``window`` of the current radius (not
    at the window's edge, brighter than the sky); a least-squares circle through those points
    (robust to a few outliers); three passes. Returns the centre, the radius and the number of
    sectors that showed a maximum. A sector in which another ring (``others``) enters the window is
    not used: there the two rings merge."""
    found = 0
    for _ in range(3):
        top = max(0, int(centre[1] - radius - window - 1))
        left = max(0, int(centre[0] - radius - window - 1))
        rows, cols = np.indices(
            (
                min(contrast.shape[0], int(centre[1] + radius + window + 2)) - top,
                min(contrast.shape[1], int(centre[0] + radius + window + 2)) - left,
            )
        )
        rows, cols = rows + top, cols + left
        distance = np.hypot(cols - centre[0], rows - centre[1])
        use = sky[rows, cols] & (np.abs(distance - radius) < window)
        bins = round(2 * window / step)
        b = np.clip(((distance - radius + window) / step).astype(int), 0, bins - 1)
        angle = np.arctan2(rows - centre[1], cols - centre[0])
        sector = ((angle + np.pi) / (2 * np.pi) * _OWN_SECTORS).astype(int) % _OWN_SECTORS
        window_band = np.abs(distance - radius) < window
        shared = np.bincount(
            sector[window_band & others[rows, cols]].ravel(), minlength=_OWN_SECTORS
        )
        labels = np.where(use, sector * bins + b + 1, 0)
        count = np.bincount(labels.ravel(), minlength=_OWN_SECTORS * bins + 1)
        present = np.nonzero(count[1:] >= 4)[0] + 1  # a median of at least 4 pixels
        medians = np.full(_OWN_SECTORS * bins + 1, np.nan)
        if present.size:
            medians[present] = median(contrast[rows, cols], labels=labels, index=present)
        points = []
        for s in range(_OWN_SECTORS):
            if shared[s]:
                continue
            prof = medians[1 + s * bins : 1 + (s + 1) * bins]
            if np.isnan(prof).sum() > bins // 3:
                continue
            values = np.nan_to_num(prof, nan=-np.inf)
            i = int(np.argmax(values))
            if (
                i == 0
                or i == bins - 1
                or values[i] <= 0
                or not np.isfinite(values[i - 1 : i + 2]).all()
            ):
                continue
            before, at, after = values[i - 1], values[i], values[i + 1]
            curvature = before - 2 * at + after
            shift = (
                float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))
                if curvature < 0
                else 0.0
            )
            r = radius - window + (i + 0.5 + shift) * step
            theta = (s + 0.5) / _OWN_SECTORS * 2 * np.pi - np.pi
            points.append((centre[0] + r * np.cos(theta), centre[1] + r * np.sin(theta)))
        found = len(points)
        if found < _OWN_SECTORS / 3:
            return centre, radius, found
        fit = least_squares(
            _circle_residual,
            [radius, centre[0], centre[1]],
            loss="soft_l1",
            f_scale=0.5,
            args=(np.array(points),),
        )
        radius, centre = float(fit.x[0]), (float(fit.x[1]), float(fit.x[2]))
    return centre, radius, found


def _circle_seeds(
    contrast: NDArray[np.float32],
    sky: NDArray[np.bool_],
    centre: tuple[float, float],
    own: float,
    k: float,
    halfwidth: float,
    eye: float,
    step: float,
) -> tuple[list[tuple[tuple[float, float], float, float]], list[float]]:
    """Starting circles of ``ring_circles`` (centre, radius, mean contrast), and the lines of the
    common radial profile (the concentric rule)."""
    offsets = np.arange(-own, own + step / 2, step)
    profile = _profile(contrast, sky, centre)
    line_level = k * _robust_std(profile)
    grid = np.indices(sky.shape)
    grid_rows: NDArray[np.int_] = grid[0]
    grid_cols: NDArray[np.int_] = grid[1]
    radius_px = np.rint(np.hypot(grid_cols - centre[0], grid_rows - centre[1])).astype(int)
    sector8 = (
        (np.arctan2(grid_rows - centre[1], grid_cols - centre[0]) + np.pi) / (2 * np.pi) * _SECTORS
    ).astype(int) % _SECTORS
    # the lines of the common profile (the concentric rule): a line that cannot be measured on
    # its own circle keeps the common centre
    concentric = {
        r: peak
        for r, _, peak in _bright_lines(profile, line_level)
        if r > eye + 2 * halfwidth
        and _around(contrast, sky, radius_px, sector8, (r, r, peak)) > _SECTORS / 2
    }
    lines = list(concentric)
    rows, cols = np.nonzero(sky)
    reach = float(np.hypot(cols - centre[0], rows - centre[1]).max())
    means = _circle_means(contrast, sky, centre, offsets, step, reach)
    middle = means[offsets.size // 2, offsets.size // 2]
    finite = middle[np.isfinite(middle)]
    noise = float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
    filled = np.nan_to_num(means, nan=-np.inf)
    span = max(1, round(1 / step))  # +-1 px of centre, +-1.5 px of radius
    top = maximum_filter(filled, size=(2 * span + 1, 2 * span + 1, 3 * span + 1), mode="nearest")
    seeds: list[tuple[tuple[float, float], float, float]] = []
    for i, j, b in np.argwhere((filled == top) & (filled >= k * noise)):
        r = (b + 0.5) * step
        inside = span <= i < offsets.size - span and span <= j < offsets.size - span
        if not inside or r <= eye + 2 * halfwidth:
            continue
        seeds.append(((centre[0] + offsets[j], centre[1] + offsets[i]), r, float(filled[i, j, b])))
    for r, peak in concentric.items():  # a line of the common profile no maximum stands for
        if all(
            abs(r - q) >= halfwidth or np.hypot(c[0] - centre[0], c[1] - centre[1]) > 1
            for c, q, _ in seeds
        ):
            seeds.append((centre, r, peak))
    return seeds, lines


def ring_circles(
    contrast: NDArray[np.float32],
    sky: NDArray[np.bool_],
    centre: tuple[float, float],
    own: float,
    k: float,
    halfwidth: float,
    eye: float = 0.0,
    step: float = 0.5,
) -> list[tuple[tuple[float, float], float, float]]:
    """Bright rings, each as its own circle: (centre, radius, peak contrast).

    The mean contrast is taken over circles of every radius about every centre within +-``own``
    pixels of ``centre`` (``step`` pixels apart); a ring is a local maximum of it, over centre and
    radius, above ``k`` times the noise of the radial profile about ``centre``, outside the eye
    (+``2 * halfwidth``); one near a line of the radial profile about ``centre`` starts from that
    line. Strongest first, each is then measured on its own (``_own_circle``: the maximum within
    +-(``halfwidth`` + ``step``) in each of 24 sectors, sectors crossed by a ring already found
    left out, and a circle through those maxima) and kept when a third of the sectors show it.
    A line of the common radial profile (local maximum above ``k`` times its noise, its sign in
    more than half of 8 sectors: the concentric rule) that cannot be measured on its own circle
    keeps the common centre. Two circles closer than ``halfwidth`` in radius and centre are one
    ring.
    """
    seeds, lines = _circle_seeds(contrast, sky, centre, own, k, halfwidth, eye, step)
    grid = np.indices(sky.shape)
    grid_rows: NDArray[np.int_] = grid[0]
    grid_cols: NDArray[np.int_] = grid[1]
    seeds.sort(key=lambda s: -s[2])
    rows, cols = grid_rows, grid_cols
    circles: list[tuple[tuple[float, float], float, float]] = []
    fallback: set[int] = set()  # lines kept about the common centre
    others = np.zeros(sky.shape, dtype=bool)
    # strongest first; a ring is measured apart from those already found
    for seed, r, peak in seeds:
        own_centre, own_radius, shown = _own_circle(
            contrast, sky, others, seed, r, halfwidth + step
        )
        if shown < _OWN_SECTORS / 3:
            line = [q for q in lines if abs(q - r) < halfwidth]
            if np.hypot(seed[0] - centre[0], seed[1] - centre[1]) > 1 or not line:
                continue
            own_centre, own_radius = centre, line[0]
            is_fallback = True
        else:
            is_fallback = False
        if any(
            abs(own_radius - q) < halfwidth
            and np.hypot(own_centre[0] - c[0], own_centre[1] - c[1]) < halfwidth
            for c, q, _ in circles
        ):
            continue
        if is_fallback:
            fallback.add(len(circles))
        circles.append((own_centre, own_radius, peak))
        others |= (
            np.abs(np.hypot(cols - own_centre[0], rows - own_centre[1]) - own_radius) <= halfwidth
        )
    theta = np.linspace(-np.pi, np.pi, 72, endpoint=False)

    def covered(line: float) -> bool:
        """A circle measured on its own runs within ``halfwidth`` of this line over most of it."""
        return any(
            np.mean(
                np.abs(
                    q
                    + (c[0] - centre[0]) * np.cos(theta)
                    + (c[1] - centre[1]) * np.sin(theta)
                    - line
                )
                < halfwidth
            )
            > 0.5
            for index, (c, q, _) in enumerate(circles)
            if index not in fallback
        )

    kept = [c for index, c in enumerate(circles) if index not in fallback or not covered(c[1])]
    return sorted(kept, key=lambda c: c[1])


def _eye(
    contrast: NDArray[np.float32],
    sky: NDArray[np.bool_],
    radius: NDArray[np.int_],
    level: float,
) -> float:
    """Radius of the bright disc at the centre of the rings: the profile (mean per radius, the
    bins being too small for a median) stays above ``level`` from radius 1 outwards; 0 if the
    first bins are not above it."""
    eye = 0
    for r in range(1, int(radius.max())):
        values = contrast[sky & (radius == r)]
        if values.size == 0 or float(values.mean()) < level:
            break
        eye = r
    return float(eye) if eye >= 2 else 0.0


def circle_bands(
    circles: list[tuple[tuple[float, float], float]],
    shape: tuple[int, int],
    halfwidth: float,
) -> NDArray[np.bool_]:
    """The ring circles (own centre, radius) masked +-``halfwidth``. Two circles whose bands come
    closer than the masking width (``2 * halfwidth``) anywhere are one band: everything from the
    inner edge of the inner one to the outer edge of the outer one, so that no sliver of sky
    between two neighbouring rings, lit by both, is left."""
    rows, cols = np.indices(shape)
    distances = [np.hypot(cols - c[0], rows - c[1]) for c, _ in circles]
    mask = np.zeros(shape, dtype=bool)
    for d, (_, r) in zip(distances, circles, strict=True):
        mask |= np.abs(d - r) <= halfwidth
    for i, (ci, ri) in enumerate(circles):
        for j, (cj, rj) in enumerate(circles):
            # closest approach of circle j outside circle i: radius difference minus centre offset
            if i == j or rj < ri:
                continue
            if rj - ri - float(np.hypot(cj[0] - ci[0], cj[1] - ci[1])) < 4 * halfwidth:
                mask |= (distances[i] >= ri - halfwidth) & (distances[j] <= rj + halfwidth)
    return mask


def ring_mask(fit: RingFit, shape: tuple[int, int], halfwidth: float) -> NDArray[np.bool_]:
    """The rings of a fit drawn again (for a frozen installation): bright lines +-``halfwidth``
    about their own centres (``circle_bands``; or between inner and outer radius in level mode)
    and the eye."""
    rows, cols = np.indices(shape)
    distance = np.hypot(cols - fit.centre[0], rows - fit.centre[1])
    mask = distance <= fit.eye
    lines = [
        circle
        for (inner, outer, _), circle in zip(fit.rings, fit.circles(), strict=True)
        if inner == outer
    ]
    mask |= circle_bands(lines, shape, halfwidth)
    for inner, outer, _ in fit.rings:
        if inner != outer:
            mask |= (distance >= inner - 0.5) & (distance <= outer + 0.5)
    return np.asarray(mask)


def without_rings(
    values: NDArray[np.float32],
    where: NDArray[np.bool_],
    centre: tuple[float, float],
    radii: list[float],
    reach: float,
    sectors: int = 16,
    step: float = 0.5,
    centres: list[tuple[float, float]] | None = None,
) -> NDArray[np.float32]:
    """``values`` minus the rings' own light: within ``reach`` pixels of each ring (and nowhere
    else), the median over ``where`` of each arc (``step`` pixels of radius, one of ``sectors``
    sectors of azimuth) is subtracted. A ring is the same along its circle (the sectors absorb a
    centre error of a pixel); a drop or a trail crossing it fills a small part of each arc and
    stays. ``centres``: the own centre of each ring (default: all about ``centre``); rings are
    then removed one after the other, each with arcs about its own centre."""
    if not radii:
        return values
    rows, cols = np.indices(values.shape)
    groups = (
        [(centre, list(radii))]
        if centres is None
        else [(c, [r]) for c, r in zip(centres, radii, strict=True)]
    )
    out = values.astype(np.float32)
    for (cx, cy), group in groups:
        distance = np.hypot(cols - cx, rows - cy)
        near = np.zeros(values.shape, dtype=bool)
        for r in group:
            near |= np.abs(distance - r) <= reach
        angle = (np.arctan2(rows - cy, cols - cx) + np.pi) / (2 * np.pi) * sectors
        arcs = (distance / step).astype(np.int64) * sectors + angle.astype(np.int64) % sectors + 1
        arcs = np.where(near, arcs, 0)
        used = np.where(where, arcs, 0)
        present = np.unique(used[used > 0])
        medians = np.zeros(int(arcs.max()) + 1, dtype=np.float64)
        if present.size:
            medians[present] = median(out, labels=used, index=present)
        out = (out - medians[arcs]).astype(np.float32)
    return out
