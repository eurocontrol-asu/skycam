"""The reflections of the optics that the sun lights: the rings and the central eye.

skystatic fits the rings once (each a circle with its own centre) and the eye (a small disc at
their centre). Their position never changes; their brightness depends on where the sun is and
whether it shines. Here each ring is cut into sectors about its own centre. On a frame, a
sector's light is the mean of its band minus the mean of the sky beside it (its flanks), as a
share of the headroom between that sky and saturation: the value scale of the sun zone. The
reflection model predicts that value from the sun alone: a table of the value against the
sector's direction relative to the sun's direction (both seen from the ring's centre in the
image) and the sun's elevation, calibrated on clear archive frames, multiplied by whether the
sun shines. A cloud edge crossing a ring cannot light it in the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skystatic import OpticsRings
from skysun.params import ReflectionParams

__all__ = ["RingLayout", "SectorMeasure", "Structure"]

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
B = NDArray[np.bool_]


@dataclass(frozen=True)
class Structure:
    """One ring (a circle) or the eye (a disc, ``radius`` its radius), native pixels."""

    name: str
    centre: tuple[float, float]
    radius: float
    disc: bool
    band: I64  # flat indices of the measured band
    band_sector: I64
    flank: I64  # flat indices of the flanks
    flank_sector: I64
    paint: I64  # flat indices of the painted band (with its taper)
    paint_angle: F32  # image angle (deg) of each painted pixel about the centre
    paint_weight: F32  # radial taper in [0, 1]

    def sun_angle(self, sun_xy: tuple[float, float]) -> float:
        """Image angle (deg, 0..360) of the sun's pixel seen from the centre."""
        dx, dy = sun_xy[0] - self.centre[0], sun_xy[1] - self.centre[1]
        return float(np.degrees(np.arctan2(dy, dx)) % 360)


@dataclass(frozen=True)
class SectorMeasure:
    """Per structure and absolute sector: mean luminance of the band and of the flanks, and
    whether the sector could be measured (``usable``)."""

    band: F64  # (structures, sectors)
    flank: F64
    usable: B

    def value(self, saturation: float) -> F64:
        """Added brightness of the band over the headroom above its flanks (NaN unusable)."""
        out = (self.band - self.flank) / np.maximum(saturation - self.flank, 1.0)
        return np.where(self.usable, out, np.nan)

    def contrast(self) -> F64:
        """Band over flanks minus 1 (NaN unusable)."""
        return np.where(self.usable, self.band / np.maximum(self.flank, 1e-6) - 1.0, np.nan)


def _taper(distance: F64, full: float, zero: float) -> F64:
    """1 up to ``full``, 0 from ``zero``, a raised cosine between."""
    if zero <= full:
        return np.asarray((distance <= full).astype(np.float64))
    t = np.clip((distance - full) / (zero - full), 0.0, 1.0)
    return np.asarray(0.5 * (1.0 + np.cos(np.pi * t)))


class RingLayout:
    """The structures of the optics on the native grid, built once per camera."""

    def __init__(
        self, optics: OpticsRings, shape: tuple[int, int], inside: B, params: ReflectionParams
    ):
        if optics.size_wh != (shape[1], shape[0]):
            optics = optics.scaled((shape[1], shape[0]))
        self.shape = shape
        self.params = params
        rings = [(c.centre, c.radius) for c in optics.circles]
        structures = [
            self._ring(f"ring{i}", centre, radius, inside, rings)
            for i, (centre, radius) in enumerate(rings)
        ]
        if params.eye and optics.fit.eye > 0:
            structures.append(self._eye(optics.fit.centre, optics.fit.eye, inside))
        self.structures = structures

    @property
    def sectors(self) -> int:
        return self.params.sectors

    def _window(self, centre: tuple[float, float], reach: float) -> tuple[I64, F64, F64]:
        """Flat indices, distance to ``centre`` and image angle of the pixels within reach."""
        rows, cols = self.shape
        x0, x1 = max(0, int(centre[0] - reach)), min(cols, int(centre[0] + reach) + 2)
        y0, y1 = max(0, int(centre[1] - reach)), min(rows, int(centre[1] + reach) + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - centre[0], yy - centre[1]
        flat = (yy * cols + xx).ravel()
        return (
            flat.astype(np.int64),
            np.hypot(dx, dy).ravel(),
            (np.degrees(np.arctan2(dy, dx)) % 360).ravel(),
        )

    def _sector(self, angle: F64) -> I64:
        return np.minimum((angle / (360 / self.sectors)).astype(np.int64), self.sectors - 1)

    def _in_other_band(
        self,
        flat: I64,
        own: tuple[tuple[float, float], float],
        rings: list[tuple[tuple[float, float], float]],
    ) -> B:
        cols = self.shape[1]
        yy, xx = np.divmod(flat, cols)
        out = np.zeros(flat.shape, dtype=bool)
        for centre, radius in rings:
            if (centre, radius) == own:
                continue
            distance = np.hypot(xx - centre[0], yy - centre[1])
            out |= np.abs(distance - radius) <= self.params.band_px + self.params.taper_px
        return out

    def _ring(
        self,
        name: str,
        centre: tuple[float, float],
        radius: float,
        inside: B,
        rings: list[tuple[tuple[float, float], float]],
    ) -> Structure:
        p = self.params
        flat, distance, angle = self._window(centre, radius + p.flank_px[1] + 1)
        offset = np.abs(distance - radius)
        ok = inside.ravel()[flat]
        band = ok & (offset <= p.band_px)
        flank = ok & (offset >= p.flank_px[0]) & (offset <= p.flank_px[1])
        flank &= ~self._in_other_band(flat, (centre, radius), rings)
        paint = ok & (offset <= p.band_px + p.taper_px)
        weight = _taper(offset[paint], p.band_px - p.taper_px, p.band_px + p.taper_px)
        return Structure(
            name,
            centre,
            radius,
            False,
            flat[band],
            self._sector(angle[band]),
            flat[flank],
            self._sector(angle[flank]),
            flat[paint],
            angle[paint].astype(np.float32),
            weight.astype(np.float32),
        )

    def _eye(self, centre: tuple[float, float], radius: float, inside: B) -> Structure:
        p = self.params
        flat, distance, angle = self._window(centre, radius + p.eye_flank_px[1] + 1)
        ok = inside.ravel()[flat]
        band = ok & (distance <= radius)
        flank = (
            ok & (distance >= radius + p.eye_flank_px[0]) & (distance <= radius + p.eye_flank_px[1])
        )
        paint = ok & (distance <= radius + p.taper_px)
        weight = _taper(distance[paint], radius - p.taper_px, radius + p.taper_px)
        return Structure(
            "eye",
            centre,
            radius,
            True,
            flat[band],
            self._sector(angle[band]),
            flat[flank],
            self._sector(angle[flank]),
            flat[paint],
            angle[paint].astype(np.float32),
            weight.astype(np.float32),
        )

    def measure(self, luminance: F32, usable: B) -> SectorMeasure:
        """Mean band and flank luminance per structure and sector of one frame; ``usable``
        pixels only (sky, not saturated). A sector needs ``min_share`` of its band and of its
        flanks usable."""
        n, s = len(self.structures), self.sectors
        band = np.full((n, s), np.nan)
        flank = np.full((n, s), np.nan)
        good = np.zeros((n, s), dtype=bool)
        lum, ok = luminance.ravel(), usable.ravel()
        for i, st in enumerate(self.structures):
            means = []
            shares = []
            for idx, sector in ((st.band, st.band_sector), (st.flank, st.flank_sector)):
                keep = ok[idx]
                total = np.bincount(sector, minlength=s)
                count = np.bincount(sector[keep], minlength=s)
                sums = np.bincount(sector[keep], lum[idx][keep], minlength=s)
                means.append(sums / np.maximum(count, 1))
                shares.append(np.where(total > 0, count / np.maximum(total, 1), 0.0))
            band[i], flank[i] = means
            good[i] = (
                (shares[0] >= self.params.min_share)
                & (shares[1] >= self.params.min_share)
                & (means[1] > 5.0)  # a sky darker than this is night or shadow (prototype)
            )
        return SectorMeasure(band, flank, good)

    def paint(self, values: list[F32]) -> F32:
        """The reflection layer: each structure's per-pixel value (``values[i]`` at its painted
        pixels) times its taper; the maximum where two overlap."""
        out = np.zeros(self.shape[0] * self.shape[1], dtype=np.float32)
        for st, value in zip(self.structures, values, strict=True):
            np.maximum.at(out, st.paint, value * st.paint_weight)
        return out.reshape(self.shape)
