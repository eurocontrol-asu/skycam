"""Cloud structure the value can be certified by (method stage "structure certification").

At low sun the clear-sky level is unknown, so the confidence of the diffuse sky is low; but a
sharp line, a mid-scale blob standing above the frame's noise or a cloud body bounded by an edge
all round is cloud whatever that level. This module finds those structures: the brightness
shape of the frame (clear-shaped or overcast-shaped), compact cloud bodies certified by their
contour, the mid-scale cue read with the +-30 s neighbours, and the certification layer
(``_cert/``: 0.9-1.0 certified with its signal-to-noise, 0.8 field, 0.5 weak line).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from skycloud.cues import LFLOOR, mad, midscale, nconv, ped_g
from skycloud.geometry import CameraGeometry, angle_to_sun
from skycloud.images import pedestal

__all__ = ["certification", "cloud_objects", "lobe_filtered", "mid_neighbours", "shape_clear"]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]
U8 = NDArray[np.uint8]


def shape_clear(
    geometry: CameraGeometry,
    lum: Any,
    lc: Any,
    lo: Any,
    valid: B,
    sun: tuple[float, float],
    level: dict[str, Any],
    shape_table: dict[str, Any],
) -> tuple[float, float | None]:
    """c7 (R6-1): how CLEAR-SHAPED the frame's brightness is, whatever its level: ratio =
    robust sd of L - Lc over that of L - Lo (zenith < 75, > 15 deg from the sun, quarter
    resolution), mapped to 1 at or below the archive clear frames' 90th percentile R0 and to 0
    at or above the archive overcast frames' 10th percentile R1 (ref c7_shape). c10: below
    +10 deg the ratio is the temporally smoothed grid ratio, blended in over +8 to +10 deg.
    Returns (s_clear, ratio)."""
    saz, szen = sun
    elev = 90.0 - szen
    g4 = angle_to_sun(geometry.az4, geometry.zen4, saz, szen)
    m = valid[::4, ::4] & (geometry.zen4 < 75) & (g4 > 15)
    if m.sum() < 500:
        return 0.0, None
    ratio = mad((lum - lc)[::4, ::4][m]) / max(mad((lum - lo)[::4, ::4][m]), 1e-4)
    if "ratio_s" in level and elev < 10.0:
        ratio = ratio + float(np.clip((10.0 - elev) / 2.0, 0.0, 1.0)) * (level["ratio_s"] - ratio)
    r0 = float(np.interp(elev, shape_table["elev"], shape_table["R0"]))
    r1 = float(np.interp(elev, shape_table["elev"], shape_table["R1"]))
    return float(np.clip((r1 - ratio) / max(r1 - r0, 1e-3), 0.0, 1.0)), float(ratio)


def cloud_objects(
    lg: Any, alpha: Any, valid: B, alpha_clear: float, object_f0: float, info: dict[str, Any]
) -> F32:
    """c8 C / c9 (R7-1): COMPACT CLOUD BODIES certified by their contour. At half resolution:
    edges where the gradient of the smoothed log brightness exceeds 2.55 x its median over the
    sky; candidate bodies = connected regions whose value is above 2 x alpha_clear and smaller
    than half the sky; a body is certified when at least object_f0 (0.8 for a region larger
    than a quarter of the sky) of its observable boundary ring lies on an edge and its value
    stands above the ring by alpha_clear; certified bodies grow into the adjacent cloud pixels.
    Returns a 0/1 map at full resolution."""
    h, w = lg.shape
    hs = (w // 2, h // 2)
    v2 = cv2.resize(valid.astype(np.uint8), hs, interpolation=cv2.INTER_NEAREST) > 0
    l2 = cv2.resize(np.where(valid, lg, 0).astype(np.float32), hs, interpolation=cv2.INTER_AREA)
    m2 = v2.astype(np.float32)
    ls = cv2.GaussianBlur(l2 * m2, (0, 0), 1.5) / np.maximum(
        cv2.GaussianBlur(m2, (0, 0), 1.5), 1e-3
    )
    gx = cv2.Sobel(ls, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ls, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    inner = cv2.erode(v2.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inner.sum() < 1000:
        return np.zeros((h, w), np.float32)
    edge = (gm > 2.55 * float(np.median(gm[inner]))) & inner
    edge_d = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a2 = cv2.resize(np.nan_to_num(alpha).astype(np.float32), hs, interpolation=cv2.INTER_AREA)
    cand = (a2 > 2.0 * alpha_clear) & v2
    out, kept, fr = _certified_bodies(cand, a2, v2, edge_d, alpha_clear, object_f0)
    if kept:
        grown = cv2.dilate((out > 0).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        out = np.where(grown & cand, 1.0, out).astype(np.float32)
    info["objects"] = {"candidates": len(fr), "certified": kept, "f": fr[:40]}
    return np.asarray(cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST))


def _certified_bodies(
    cand: B, a2: F32, v2: B, edge_d: B, alpha_clear: float, object_f0: float
) -> tuple[F32, int, list[float]]:
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    out = np.zeros(cand.shape, np.float32)
    amax = 0.25 * v2.sum()
    kept, fr = 0, []
    inner_ring = cv2.erode(v2.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    for k in range(1, n):
        area = st[k, cv2.CC_STAT_AREA]
        big = area > amax
        if area < 50 or area > 2 * amax:
            continue
        x0, y0, ww, hh = st[k, 0], st[k, 1], st[k, 2], st[k, 3]
        sl = (slice(max(y0 - 4, 0), y0 + hh + 4), slice(max(x0 - 4, 0), x0 + ww + 4))
        comp = lab[sl] == k
        # only the OBSERVABLE contour counts: the ring more than 5 px inside the sky
        ring = (
            (cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
            & ~comp
            & inner_ring[sl]
        )
        if ring.sum() < 20:
            continue
        f = float(edge_d[sl][ring].mean())
        contrast = float(a2[sl][comp].mean() - a2[sl][ring].mean())
        fr.append(round(f, 2))
        if f >= (0.8 if big else object_f0) and contrast >= alpha_clear:
            out[sl][comp] = 1.0
            kept += 1
    return out, kept, fr


def _neighbour_bandpass(
    imn: U8, valid: B, px_deg: float, sky8: B, g8: Any, r_step: float
) -> tuple[F32, B, B]:
    """A neighbour's mid-scale band-pass and where its sky did not change (1/8 resolution
    normalised brightness change below R_STEP), upsampled."""
    h, w = valid.shape
    pn = pedestal(imn)
    gn_full = imn[..., 1].astype(np.float32)
    lgn = np.log(np.maximum(gn_full - ped_g(pn), LFLOOR * ped_g(pn)))
    bpn, _ = midscale(lgn, valid, px_deg)
    gn = _g8(gn_full, sky8)
    same8 = sky8 & (np.abs(gn - g8) / np.maximum((gn + g8) / 2.0, 1e-3) < r_step)
    same = cv2.resize(same8.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return bpn, same, same8


def _g8(g: F32, sky8: B) -> Any:
    """The 1/8-resolution normalised brightness (8 x 8 mean, Gaussian 1 cell, / sky median)."""
    h, w = g.shape
    s = cv2.GaussianBlur(
        cv2.resize(g.astype(np.float32), (w // 8, h // 8), interpolation=cv2.INTER_AREA),
        (0, 0),
        1.0,
    )
    return s / max(float(np.median(s[sky8])), 1e-3) if sky8.any() else s


def _same_sign_min(a: Any, b: Any) -> Any:
    return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def mid_neighbours(
    neighbours: dict[int, Callable[[], U8 | None]],
    bp: F32,
    valid: B,
    cov: F32,
    g: F32,
    px_deg: float,
    r_step: float,
) -> tuple[F32, dict[str, Any]]:
    """c7: the mid-scale band-pass read with the +-30 s neighbours (``neighbours``: offset ->
    loader of the neighbour image, None when absent or excluded) and the frame's own noise at
    that scale. On sky unchanged between the frames the cue reads the median of the three
    frames (the smaller same-sign one of two), elsewhere the frame's own."""
    h, w = valid.shape
    sky8 = (
        cv2.resize(valid.astype(np.float32), (w // 8, h // 8), interpolation=cv2.INTER_AREA) > 0.99
    )
    g8 = _g8(g, sky8)
    ok = valid & (cov > 0.9)
    nb: dict[int, tuple[F32, B]] = {}
    noises = []
    for s in (-30, 30):
        imn = neighbours[s]()
        if imn is None:
            continue
        bpn, same, _ = _neighbour_bandpass(imn, valid, px_deg, sky8, g8, r_step)
        m = ok & same
        if m.sum() > 5000:
            d = (bp - bpn)[m][::5]
            noises.append(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)))
        nb[s] = (bpn, same)
    note: dict[str, Any] = {"neighbours": len(nb)}
    if noises:
        note["noise"] = round(float(np.median(noises)), 5)
    if not nb:
        return bp, note
    out = bp
    if -30 in nb and 30 in nb:
        (bm, sm), (bq, sq) = nb[-30], nb[30]
        med = np.median(np.stack([bm, bp, bq]), axis=0)
        out = np.where(
            sm & sq,
            med,
            np.where(sm, _same_sign_min(bp, bm), np.where(sq, _same_sign_min(bp, bq), bp)),
        )
    else:
        bn, sn = next(iter(nb.values()))
        out = np.where(sn, _same_sign_min(bp, bn), bp)
    return out.astype(np.float32), note


def lobe_filtered(a_m: F32, bpm: F32, px_deg: float) -> Any:
    """c15 (E, R11-3): a pixel certified by the mid-scale cue alone must have the sign of the
    strongest structure within 1.5 deg (no band-pass side lobes beside a line)."""
    r15 = max(round(1.5 * px_deg), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r15 + 1, 2 * r15 + 1))
    pos = cv2.dilate(np.maximum(bpm, 0.0).astype(np.float32), ker)
    neg = cv2.dilate(np.maximum(-bpm, 0.0).astype(np.float32), ker)
    same = np.where(bpm >= 0, pos >= neg, neg > pos)
    return np.where(same, a_m, 0.0)


def certification(
    stru: Any,
    a_t: Any,
    a_m: Any,
    alpha: Any,
    alpha_m: Any,
    valid: B,
    px_deg: float,
    alpha_clear: float,
    field_dens: float,
) -> tuple[F32, B, B]:
    """c9-c16: the certification layer. Certified (0.9-1.0 with the structure's signal-to-noise)
    where a structure (texture, lobe-checked mid-scale cue or contour-bounded body) reaches 0.5,
    the value says cloud (at least twice the read-clear level) and the pixel's OWN measured
    value (before the fill) says cloud too; weak line evidence (0.5) where the texture alone
    reaches 0.5 with a value above the read-clear level; field (0.8) where the mid-scale cue is
    lit on at least ``field_dens`` of the 1.5-deg neighbourhood. Returns (layer, field mask,
    certified at full weight before the field step)."""
    two = 2.0 * alpha_clear
    a0 = np.nan_to_num(alpha)
    cert = ((stru >= 0.5) & (a0 >= two) & valid).astype(np.float32)
    own = alpha_m >= two
    cert = np.where(own, cert, 0.0).astype(np.float32)
    cert = np.where(cert > 0, 0.9 + 0.1 * np.clip((stru - 0.5) / 0.5, 0.0, 1.0), 0.0).astype(
        np.float32
    )
    weak = (a_t >= 0.5) & (a0 >= alpha_clear) & valid & (cert == 0)
    weak &= alpha_m >= alpha_clear
    cert = np.where(weak, 0.5, cert).astype(np.float32)
    strong = cert >= 0.9
    lit = ((a_m >= 0.5) & valid).astype(np.float32)
    dens = nconv(lit, valid, 1.5 * px_deg)[0]
    fld = (dens >= field_dens) & (a0 >= two) & valid & (cert < 1.0)
    fld &= alpha_m >= two
    cert = np.where(fld, 0.8, cert).astype(np.float32)
    return cert, fld, strong
