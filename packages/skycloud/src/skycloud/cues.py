"""The two cues and the image filters they are read through (method stage "cues").

* colour index N = (B - R) / (B + R) on the raw image (gain-invariant: measured);
* brightness L = log((G - pedestal) / pedestal), in units of the camera pedestal (the camera
  changes its gain, not its exposure time, and the pedestal follows the gain: L is gain-free;
  the G pedestal is read from the stable R corner).

Both are read after an edge-preserving smoothing (masked guided filter, guide = G). The other
filters here are the local structure of the brightness (texture, mid-scale band-pass), the
local clear baseline and shrinkage of the colour opacity, the low-information fill and the
fusion of the single-cue opacities.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

__all__ = [
    "LFLOOR",
    "PED_BR",
    "PED_GR",
    "clear_baseline",
    "colour_opacity",
    "cues",
    "fill_low_info",
    "fuse",
    "guided",
    "level_basis",
    "mad",
    "midscale",
    "nconv",
    "noise_sigma",
    "ped_g",
    "texture",
]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]

LFLOOR = 0.02  # floor of (G - pedestal) / pedestal before the log
PED_GR = (
    0.545  # G / R of the pedestal (median of the corners over the archive, well-measured gains)
)
PED_BR = 0.882  # B / R of the pedestal (archive frames whose R corner is >= 60 counts: high gain)


def ped_g(ped: Any) -> Any:
    """The G pedestal from the R corner and the pedestal's fixed colour: the G corner itself
    flips by +-1 count from frame to frame (JPEG quantisation of the dark corners)."""
    return PED_GR * np.asarray(ped)[..., 0]


def cues(r: Any, g: Any, b: Any, ped: Any) -> tuple[Any, Any]:
    """N (raw colour index) and L (log brightness in pedestal units)."""
    n = (b - r) / np.maximum(b + r, 1.0)
    pg = ped_g(ped)
    lum = np.log(np.maximum(g - pg, LFLOOR * pg) / pg)
    return n, lum


def mad(x: Any) -> float:
    """Robust sd (1.4826 x median absolute deviation); NaN for an empty array."""
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def level_basis(g: Any, z: Any) -> NDArray[np.float64]:
    """Basis of the frame's clear-sky level of N: a constant, two terms decaying with the angle
    to the sun (aureole, broad solar-side whitening) and one with the elevation (horizon haze)."""
    g, z = np.asarray(g, np.float64), np.asarray(z, np.float64)
    return np.stack(
        [np.ones_like(g), np.exp(-g / 25.0), np.exp(-g / 70.0), np.exp(-(90.0 - z) / 8.0)], -1
    )


def _box(x: Any, k: int) -> Any:
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)


def guided(guide: F32, srcs: list[F32], w: F32, r: int, eps: float) -> list[Any]:
    """Masked guided filter (He et al.): each source smoothed with ``guide`` over the pixels
    where ``w`` = 1; a source that IS the guide (same object) takes the guide's own form."""
    k = 2 * r + 1
    nw = _box(w, k) + 1e-6
    m_i = _box(guide * w, k) / nw
    v_i = np.maximum(_box(guide * guide * w, k) / nw - m_i * m_i, 0)
    out = []
    for p in srcs:
        if p is guide:
            a = v_i / (v_i + eps)
            b = m_i * (1 - a)
        else:
            mp = _box(p * w, k) / nw
            a = (_box(guide * p * w, k) / nw - m_i * mp) / (v_i + eps)
            b = mp - a * m_i
        ma, mb = _box(a * w, k) / nw, _box(b * w, k) / nw
        out.append(ma * guide + mb)
    return out


def noise_sigma(g: F32, valid: B) -> float:
    """Pixel noise of G on the valid sky: robust sd of G minus its 3 x 3 mean (sampled)."""
    h = (g - cv2.blur(g, (3, 3)))[::3, ::3][valid[::3, ::3]]
    return float(1.4826 * np.median(np.abs(h - np.median(h))) * 1.06) if h.size > 100 else 1.0


def texture(lg: Any, w: Any, s1: float = 2.0, s2: float = 8.0, s3: float = 12.0) -> Any:
    """Local structure of the log brightness: RMS over ~s3 px of a band-pass (Gaussian s1
    minus Gaussian s2), over the pixels of weight ``w``."""

    def nb(a: Any, s: float) -> Any:
        return cv2.GaussianBlur(a * w, (0, 0), s) / np.maximum(cv2.GaussianBlur(w, (0, 0), s), 1e-4)

    bp = nb(lg, s1) - nb(lg, s2)
    return np.sqrt(np.maximum(nb(bp * bp, s3), 0.0))


def midscale(lg: Any, valid: B, px_deg: float) -> tuple[Any, Any]:
    """c6: band-pass of the log brightness, Gaussian(1 deg) - Gaussian(3 deg), masked to the
    valid sky (at half resolution, as on the archive), and the sky coverage of its window."""
    h, w = lg.shape
    l2 = cv2.resize(lg.astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(valid.astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA) > 0.99
    mm = m2.astype(np.float32)

    def nb(s: float) -> Any:
        num = cv2.GaussianBlur(np.where(m2, l2, 0).astype(np.float32), (0, 0), s)
        return num / np.maximum(cv2.GaussianBlur(mm, (0, 0), s), 1e-3)

    s = px_deg / 2.0
    bp = nb(s) - nb(3 * s)
    cov = cv2.GaussianBlur(mm, (0, 0), 3 * s)
    return (
        cv2.resize(bp, (w, h), interpolation=cv2.INTER_LINEAR),
        cv2.resize(cov, (w, h), interpolation=cv2.INTER_LINEAR),
    )


def nconv(x: Any, m: B, sigma: float) -> tuple[Any, Any]:
    """Normalised Gaussian convolution of ``x`` over the pixels where ``m``, at 1/8 resolution."""
    h, wd = x.shape
    small = (max(wd // 8, 1), max(h // 8, 1))
    mm = cv2.resize(m.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    xx = cv2.resize(np.where(m, x, 0.0).astype(np.float32), small, interpolation=cv2.INTER_AREA)
    s = sigma / 8.0
    out = cv2.GaussianBlur(xx, (0, 0), s) / np.maximum(cv2.GaussianBlur(mm, (0, 0), s), 1e-4)
    return cv2.resize(out, (wd, h), interpolation=cv2.INTER_LINEAR), mm


def fill_low_info(alpha: Any, w: Any, valid: B, eps: float) -> Any:
    """Where the cues carry little information (precision ``w`` small) the value leans on the
    informative sky around (normalised convolution weighted by precision), only to the extent
    the surroundings are MORE informative than the pixel: fill weight eps (1 - w / W_around)."""
    h, wd = alpha.shape
    small = (wd // 8, h // 8)
    ww = np.minimum(w, 400.0) * valid
    ws = cv2.resize(ww.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    as_ = cv2.resize((alpha * ww).astype(np.float32), small, interpolation=cv2.INTER_AREA)
    num = np.zeros(ws.shape, np.float32)
    den = np.full(ws.shape, 1e-6, np.float32)
    cnt = np.full(ws.shape, 1e-6, np.float32)
    for s, a in ((5.0, 1.0), (16.0, 0.03), (50.0, 0.001)):
        num += a * cv2.GaussianBlur(as_, (0, 0), s)
        den += a * cv2.GaussianBlur(ws, (0, 0), s)
        cnt += a * cv2.GaussianBlur(vs, (0, 0), s)
    around = cv2.resize(num / den, (wd, h), interpolation=cv2.INTER_LINEAR)
    w_around = cv2.resize(den / cnt, (wd, h), interpolation=cv2.INTER_LINEAR)
    lam = eps * np.clip(1.0 - w / np.maximum(w_around, 1e-6), 0.0, 1.0)
    return (w * alpha + lam * around) / (w + lam + 1e-6)


def _block_noise(bm: Any) -> float | None:
    """Noise sd of the 8 px block means: robust sd of adjacent-cell differences / sqrt 2."""
    okb = bm < 5.0
    dx = (bm[:, 1:] - bm[:, :-1])[okb[:, 1:] & okb[:, :-1]]
    dy = (bm[1:, :] - bm[:-1, :])[okb[1:, :] & okb[:-1, :]]
    dd = np.concatenate([dx, dy])
    if dd.size <= 200:
        return None
    return float(1.4826 * float(np.median(np.abs(dd - np.median(dd)))) / np.sqrt(2.0))


def clear_baseline(ab: Any, cand: B, px_deg: float, ref_deg: float, base_q: int) -> Any:
    """c1: the LOCAL clear-sky level of the colour opacity, NEGATIVE ONLY: lower quartile of
    the 8 x 8 px block means of ``ab`` over the smooth pixels ``cand``, in a window of radius
    2 x the tables' smoothing scale, read relative to the frame's block noise (c7: the quartile
    of pure noise is -0.674 sd), smoothed at that scale and capped at 0."""
    h, wd = ab.shape
    small = (max(wd // 8, 1), max(h // 8, 1))
    mm = cv2.resize(cand.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    bm = cv2.resize(np.where(cand, ab, 0.0).astype(np.float32), small, interpolation=cv2.INTER_AREA)
    bm = bm / np.maximum(mm, 1e-4)
    bm = np.where(mm > 0.5, np.minimum(bm, 0.0), 10.0)  # cells without smooth sky never pull
    sc = ref_deg * px_deg / 8.0
    k = 2 * round(2 * sc) + 1
    q = ndimage.percentile_filter(bm, base_q, size=k, mode="nearest")
    s_b = _block_noise(bm)
    if s_b is not None:
        q = np.where(q < 5.0, q + 0.674 * s_b, q)
    ok = (q < 5.0).astype(np.float32)
    q = cv2.GaussianBlur(np.where(ok > 0, q, 0.0).astype(np.float32), (0, 0), sc) / np.maximum(
        cv2.GaussianBlur(ok, (0, 0), sc), 1e-3
    )
    q = np.where(cv2.GaussianBlur(ok, (0, 0), sc) > 0.05, np.minimum(q, 0.0), 0.0)
    return cv2.resize(q.astype(np.float32), (wd, h), interpolation=cv2.INTER_LINEAR)


def colour_opacity(ab: Any, sa: Any, base: Any, shrink: float) -> Any:
    """The colour opacity measured from the local clear baseline, shrunk by its clear-sky
    uncertainty ``sa``: a x a^2 / (a^2 + sa^2) (a departure within the clear spread counts
    little, one of several spreads counts fully)."""
    a0 = ab - base
    return a0**3 / (a0 * a0 + shrink * sa * sa)


def fuse(
    a_n: Any,
    a_l: Any,
    a_t: Any,
    wn_cloud: Any,
    wn_clear: Any,
    wl_cloud: Any,
    wl_clear: Any,
    wt: Any,
) -> tuple[Any, Any]:
    """The value from the single-cue opacities, and its precision. A cue's weight goes from
    its clear-verdict precision to its cloud-verdict precision as its opacity goes 0 -> 1; the
    brightness detects but does not grade (its cloud weight x opacity^2); texture is one-sided."""
    a_n = np.clip(a_n, -0.25, 1.25)
    a_l = np.clip(a_l, 0.0, 1.25)
    u_n, u_l = np.clip(a_n, 0.0, 1.0), np.clip(a_l, 0.0, 1.0)
    k_n = wn_clear * (1 - u_n) + wn_cloud * u_n
    k_l = wl_clear * (1 - u_l) + wl_cloud * u_l * u_l
    k = k_n + k_l + wt
    return (k_n * a_n + k_l * a_l + wt * a_t) / (k + 1e-6), k
