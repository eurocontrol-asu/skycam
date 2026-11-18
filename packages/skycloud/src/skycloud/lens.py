"""Lens reflections: the ghosts flagged in ``<ts>_lens.png`` and the rings side layer.

Re-implemented from the GVCCS labelling kit (``sun_artefacts.lens_correction``, version
"lens v2": ghosts on the sun - optical centre line within its measured span, or chromatic below
zenith 70, isolated in clear sky; rings about a searched centre, kept where every quadrant
agrees). The cloud map does NOT correct the value under a ghost (it hides the sky): it only
exports the ghost pixels as a flag and the ring field as a side layer. The kit's value
correction (``inpaint_shift``) and hand-declared ghost discs are therefore not ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

__all__ = ["LensResult", "lens_flags"]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]
U8 = NDArray[np.uint8]

# sun_artefacts.LENS (lens v2), line_px = sun_artefacts.DEFAULTS["ghost_line_px"] = 90
LENS: dict[str, Any] = {
    "dip": 0.012,
    "mag_hp": 0.025,
    "gam_min": 3.0,
    "line_px": 90,
    "line_t": (-0.7, 1.2),
    "line_min_px": 4,
    "line_max_px": 1500,
    "line_dip": 0.025,
    "line_max_elong": 4.0,
    "chroma_mean": 0.02,
    "chroma_peak": 0.06,
    "chroma_max_px": 400,
    "chroma_max_elong": 3.0,
    "mag_peak": 0.07,
    "mag_white": 0.02,
    "mag_min_px": 12,
    "mag_max_px": 40000,
    "zen_max": 70.0,
    "iso_ring_px": 45,
    "iso_gap_px": 12,
    "iso_share": 0.08,
    "iso_solidity": 0.75,
    "mag_iso_share": 0.25,
    "touch_share": 0.5,
    "margin_px": 1,
    "ring_bin_px": 2.0,
    "ring_hp_px": 30.0,
    "ring_max": 0.005,
    "ring_min_q": 2,
    "ring_gam_min": 20.0,
    "ring_stride": 4,
}


@dataclass(frozen=True)
class LensResult:
    """The ghost pixels (bool), the ring field (float32, 0 where no coherent ring) and the
    number of radial bins holding a ring."""

    ghosts: B
    rings: F32
    ring_bins: int


def _nblur(a: Any, w: Any, s: float) -> Any:
    return cv2.GaussianBlur(a * w, (0, 0), s) / np.maximum(cv2.GaussianBlur(w, (0, 0), s), 1e-4)


def _disc(k: int) -> U8:
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))


def _isolation(lab: Any, st: Any, i: int, cloudy: F32) -> tuple[float, float, float]:
    """Share of cloud in a ring around blob ``i``, its solidity and the cloud it touches."""
    q = LENS
    x0, y0, ww, hh = st[i, :4]
    r = int(q["iso_ring_px"])
    sl = (slice(max(0, y0 - r), y0 + hh + r), slice(max(0, x0 - r), x0 + ww + r))
    bl = (lab[sl] == i).astype(np.uint8)
    near = cv2.dilate(bl, _disc(int(q["iso_gap_px"]))).astype(bool)
    ring = cv2.dilate(bl, _disc(r)).astype(bool) & ~near
    c = cloudy[sl]
    cv = c[ring]
    share = float(np.nanmean(cv)) if np.isfinite(cv).any() else 1.0
    cl = np.nan_to_num(c, nan=0.0) > 0.5
    _, cl_lab = cv2.connectedComponents(cl.astype(np.uint8), connectivity=8)
    ids = np.unique(cl_lab[near & cl])
    ids = ids[ids > 0]
    touch = float((np.isin(cl_lab, ids) & ~near).sum() / max(bl.sum(), 1)) if ids.size else 0.0
    cnts, _ = cv2.findContours(bl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cnts))) if cnts else 0.0
    solidity = float(bl.sum() / max(hull, 1.0)) if bl.sum() > 8 else 1.0
    return share, solidity, touch


def _elongation(lab: Any, st: Any, i: int) -> float:
    x0, y0, ww, hh = st[i, :4]
    ys, xs = np.nonzero(lab[y0 : y0 + hh, x0 : x0 + ww] == i)
    if len(xs) < 3:
        return 1.0
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(float))))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-3)))


@dataclass(frozen=True)
class _Ctx:
    """What the ghost tests share: the sun line and the cloud field."""

    ox: float
    oy: float
    ux: float
    uy: float
    n2: float
    cl: F32

    def on_line(self, x: float, y: float) -> bool:
        t0, t1 = LENS["line_t"]
        dist = abs((x - self.ox) * self.uy - (y - self.oy) * self.ux) / np.sqrt(self.n2)
        t = ((x - self.ox) * self.ux + (y - self.oy) * self.uy) / self.n2
        return bool(dist <= LENS["line_px"] and t0 <= t <= t1)

    def clear_around(self, lab: Any, st: Any, i: int, share_max: float) -> bool:
        share, sol, touch = _isolation(lab, st, i, self.cl)
        return share <= share_max and sol >= LENS["iso_solidity"] and touch <= LENS["touch_share"]


def _white_blob_ok(ctx: _Ctx, lab: Any, st: Any, i: int, stats: tuple[Any, ...], cen: Any) -> bool:
    q = LENS
    deep, zm, cm, cp = stats
    a = st[i, cv2.CC_STAT_AREA]
    if a < q["line_min_px"] or a > q["line_max_px"] or deep[i] > -q["line_dip"]:
        return False
    chroma = cm[i] >= q["chroma_mean"] and cp[i] >= q["chroma_peak"]
    line = ctx.on_line(*cen[i])
    if not line and not (chroma and a <= q["chroma_max_px"]):
        return False
    if zm[i] > q["zen_max"] and not (line and chroma):
        return False
    if _elongation(lab, st, i) > (q["line_max_elong"] if line else q["chroma_max_elong"]):
        return False
    return ctx.clear_around(lab, st, i, q["iso_share"])


def _white_blobs(ctx: _Ctx, cand: B, hps: F32, hpm: F32, zen: F32) -> tuple[B, Any, Any]:
    """Whitening blobs on the sun line, or chromatic."""
    n, lab_, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    lab: Any = lab_
    keep = np.zeros(n, bool)
    if n > 1:
        idx = np.arange(n)
        stats = (
            ndimage.minimum(hps, lab, idx),
            ndimage.mean(zen, lab, idx),
            ndimage.mean(hpm, lab, idx),
            ndimage.maximum(hpm, lab, idx),
        )
        for i in range(1, n):
            keep[i] = _white_blob_ok(ctx, lab, st, i, stats, cen)
    return keep[lab], lab, st[:, cv2.CC_STAT_AREA]


def _tinted_blobs(ctx: _Ctx, cand: B, hps: F32, hpm: F32, zen: F32) -> B:
    """Tinted blobs (a pink / violet ghost and its streaks), whitening the sky."""
    q = LENS
    n, lab_, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    lab: Any = lab_
    keep = np.zeros(n, bool)
    if n > 1:
        idx = np.arange(n)
        peak, white = ndimage.maximum(hpm, lab, idx), ndimage.mean(hps, lab, idx)
        zm = ndimage.mean(zen, lab, idx)
        area = st[:, cv2.CC_STAT_AREA]
        ok = (
            (area >= q["mag_min_px"])
            & (area <= q["mag_max_px"])
            & (peak >= q["mag_peak"])
            & (white <= -q["mag_white"])
        )
        for i in np.flatnonzero(ok[1:]) + 1:
            if zm[i] > q["zen_max"] and not ctx.on_line(*cen[i]):
                continue
            keep[i] = ctx.clear_around(lab, st, int(i), q["mag_iso_share"])
    return np.asarray(keep[lab])


def _ghosts(
    img: U8, d: F32, valid: B, gam: F32, zen: F32, centre: Any, sun_xy: Any, cloudy: B
) -> B:
    """The lens ghosts (kit: sun_artefacts.lens_ghosts), their own footprint."""
    q = LENS
    f = img.astype(np.float32)
    w = valid.astype(np.float32)
    mrel = ((f[..., 0] + f[..., 2]) / 2 - f[..., 1]) / np.maximum(f.mean(-1), 1.0)
    hpm = cv2.GaussianBlur(mrel, (0, 0), 2.0) - _nblur(mrel, w, 40)  # tint over the surroundings
    hps = cv2.GaussianBlur(d - _nblur(d, w, 10), (0, 0), 1.5)  # whitening (< 0)
    cl = np.where(valid, cloudy, np.nan).astype(np.float32)
    ox, oy = centre
    ux, uy = sun_xy[0] - ox, sun_xy[1] - oy
    ctx = _Ctx(ox, oy, ux, uy, max(float(ux * ux + uy * uy), 1e-6), cl)
    near_sun = gam > q["gam_min"]
    out, white_lab, white_area = _white_blobs(
        ctx, valid & near_sun & (hps < -q["dip"]), hps, hpm, zen
    )
    tinted = _tinted_blobs(ctx, valid & near_sun & (hpm > q["mag_hp"]), hps, hpm, zen)
    if tinted.any():  # a tinted ghost's footprint: its tint and the whitening it carries
        ids = np.unique(white_lab[tinted])
        ids = ids[(ids > 0) & (white_area[ids] <= q["mag_max_px"])]
        tinted |= np.isin(white_lab, ids)
    out = out | tinted
    k = int(q["margin_px"])
    if k and out.any():
        out = cv2.dilate(out.astype(np.uint8), _disc(k)).astype(bool)
    return np.asarray(out & valid)


def _ring_profile(dd: Any, xx: Any, yy: Any, c: Any, nb: int, bin_px: float, h: int) -> Any:
    """Per quadrant around ``c``: the mean of d per radius bin, high-passed."""
    r = np.hypot(xx - c[0], yy - c[1])
    ri = np.minimum((r / bin_px).astype(int), nb - 1)
    qd = ((np.arctan2(yy - c[1], xx - c[0]) + np.pi) / (np.pi / 2)).astype(int) % 4
    key = qd * nb + ri
    s = np.bincount(key, dd, 4 * nb).reshape(4, nb)
    n = np.bincount(key, None, 4 * nb).reshape(4, nb)
    prof = np.where(n >= 40, s / np.maximum(n, 1), np.nan)
    k = np.ones(h) / h
    hp = np.full_like(prof, np.nan)
    for i in range(4):
        ok = np.isfinite(prof[i])
        num = np.convolve(np.where(ok, prof[i], 0.0), k, "same")
        den = np.convolve(ok.astype(float), k, "same")
        hp[i] = np.where(ok & (den > 0.6), prof[i] - num / np.maximum(den, 1e-6), np.nan)
    return hp


def _coherent(hp: Any, min_q: int) -> Any:
    """Per radius: the ring amplitude where every quadrant with data agrees in sign."""
    fin = np.isfinite(hp)
    pos, neg = np.where(fin, hp > 0, True).all(0), np.where(fin, hp < 0, True).all(0)
    same = (fin.sum(0) >= min_q) & (pos | neg)
    amp = np.where(fin, np.abs(hp), np.inf).min(0)
    return np.where(same, np.where(pos, 1.0, -1.0) * np.where(np.isfinite(amp), amp, 0.0), 0.0)


def _rings(d: F32, use: B, valid: B, centre: Any, sun_xy: Any) -> tuple[F32, int]:
    """The lens rings (kit: sun_artefacts.lens_rings) and their number of radial bins."""
    q = LENS
    k = int(q["ring_stride"])
    ys, xs = np.nonzero(use[::k, ::k])
    out = np.zeros(d.shape, np.float32)
    if len(ys) < 2000:
        return out, 0
    xx, yy = xs.astype(np.float64) * k, ys.astype(np.float64) * k
    dd = d[::k, ::k][ys, xs].astype(np.float64)
    b = float(q["ring_bin_px"])
    nb, h = int(1200 / b), int(q["ring_hp_px"] / b) | 1
    o = np.asarray(centre, float)
    v = np.asarray(sun_xy, float) - o if sun_xy is not None else np.zeros(2)
    cands = [o + t * v for t in np.arange(-0.6, 1.31, 0.1)] if np.hypot(*v) > 1 else [o]
    cands += [o + np.array([a, c]) for a in (-80, -40, 0, 40, 80) for c in (-80, -40, 0, 40, 80)]

    def score(c: Any) -> tuple[float, Any, Any]:
        r = _coherent(_ring_profile(dd, xx, yy, c, nb, b, h), int(q["ring_min_q"]))
        return float(np.sum(r**2)), c, r

    best = max((score(c) for c in cands), key=lambda s: s[0])
    for a in (-20, 0, 20):
        for c in (-20, 0, 20):
            if a or c:
                best = max(best, score(best[1] + np.array([a, c])), key=lambda s: s[0])
    _, c, r = best
    r = np.clip(r, -q["ring_max"], q["ring_max"]).astype(np.float32)
    yy_, xx_ = np.nonzero(valid)
    out[yy_, xx_] = r[np.minimum((np.hypot(xx_ - c[0], yy_ - c[1]) / b).astype(int), nb - 1)]
    return out, int((r != 0).sum())


def lens_flags(
    img: U8,
    d: F32,
    valid: B,
    gam: F32,
    vec: F32,
    centre: Any,
    sun_px: tuple[int, int],
    visible: bool,
    cloudy: B,
) -> LensResult:
    """Ghosts (only with a visible sun) and rings (always, coherence-gated) of one frame.

    ``d``: the colour index's departure from the clear sky (what the kit calls the gap);
    ``gam``: the angle of every pixel to the sun; ``vec``: the pixel directions (their zenith
    gates the ghosts); ``cloudy``: the pixels read as cloud (excluded from the ring profile)."""
    ghosts = np.zeros(d.shape, bool)
    if visible:
        zen = np.degrees(np.arccos(np.clip(vec[..., 2], -1.0, 1.0))).astype(np.float32)
        ghosts = _ghosts(img, d, valid, gam, zen, centre, sun_px, cloudy)
    use = valid & ~cloudy & ~ghosts & (gam > LENS["ring_gam_min"])
    rings, ring_bins = _rings(d, use, valid, centre, sun_px)
    return LensResult(ghosts, rings, ring_bins)
