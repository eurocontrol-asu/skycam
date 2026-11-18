"""Steps 5 and 6: smooth, complete tables and their spreads.

- ``smooth_tables`` (prototype ``b4_smooth.py``): count-weighted normalised Gaussian
  convolution of the raw medians and spreads over (elevation node, angle to the sun, zenith),
  empty bins from the nearest filled one, nodes without data from the nearest node, a last
  light smoothing so every table is smooth for the trilinear lookup.
- ``adapted_spread`` (``b6_spread.py``): the clear-sky spread after the frame's own level is
  removed (each frame's median residual to the smooth clear table), ``Nc_sda`` / ``Lc_sda``.
- ``rebuild_spreads`` (``c14_spread.py``): each spread table shrunk towards a heavy smooth
  version where its deviation does not exceed its statistical error (the spread tables read by
  the confidence).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from skycloud.reference_build.binning import BinMeans
from skycloud.reference_build.labels import CLEAR, bin_centres, node_of
from skycloud.reference_build.params import BinParams, SmoothParams

__all__ = [
    "REBUILT_SPREADS",
    "adapted_spread",
    "fill_nearest",
    "rebuild_spreads",
    "smooth_tables",
    "weighted_smooth",
]

#: The spread tables rebuilt by c14 and the count each one is measured on.
REBUILT_SPREADS = {
    "Nc_sdf": "n_c", "Nc_sd": "n_c", "Lc_sda": "n_c", "Lc_sd": "n_c", "No_sd": "n_o",
    "Lo_sd": "n_o",
}  # fmt: skip


def fill_nearest(s: Any, ok: Any) -> Any:
    """Fill ``s`` (nodes, ...) where not ``ok``: inside a node from its nearest filled bin, a
    node without any filled bin from the nearest node that has one. In place; returns ``s``."""
    ne = s.shape[0]
    for e in range(ne):
        if ok[e].any():
            idx = ndimage.distance_transform_edt(
                ~ok[e], return_distances=False, return_indices=True
            )
            s[e] = s[e][tuple(idx)]
    have = [e for e in range(ne) if ok[e].any()]
    if not have:
        raise ValueError("a table has no measured bin at any elevation node: too few frames")
    for e in range(ne):
        if not ok[e].any():
            s[e] = s[min(have, key=lambda h: abs(h - e))]
    return s


def weighted_smooth(v: Any, w: Any, sig: Any, den_min: float) -> Any:
    """Normalised convolution of ``v`` with weights ``w``; NaN where the weight is too small."""
    num = ndimage.gaussian_filter(np.where(w > 0, v, 0) * w, sig, mode="nearest")
    den = ndimage.gaussian_filter(w, sig, mode="nearest")
    return np.where(den > den_min, num / np.maximum(den, 1e-9), np.nan)


def _smooth(v: Any, n: Any, sig: Any, shape: tuple[int, int, int], sp: SmoothParams) -> Any:
    v = v.reshape(shape)
    w = np.where(np.isfinite(v), np.minimum(n.reshape(shape), sp.count_cap).astype(np.float64), 0.0)
    s = weighted_smooth(v, w, sig, sp.den_min)
    s = fill_nearest(s, np.isfinite(s))
    return ndimage.gaussian_filter(s, sp.final_sigma, mode="nearest").astype(np.float32)


def smooth_tables(raw: Any, p: BinParams, sp: SmoothParams) -> dict[str, Any]:
    """The smooth tables (nodes, angle bins, zenith bins) from the raw tables, with the
    elevation nodes, bin sizes and counts (the prototype's ``v3_ref.npz``)."""
    shape = (len(p.elevation_nodes), p.n_gamma, p.n_zenith)
    out: dict[str, Any] = {}
    for tag in ("c", "o"):
        n = raw[f"n_{tag}"]
        for cue in ("N", "L"):
            out[f"{cue}{tag}"] = _smooth(raw[f"{cue}{tag}"], n, sp.median_sigma, shape, sp)
            out[f"{cue}{tag}_sd"] = _smooth(raw[f"{cue}{tag}_sd"], n, sp.spread_sigma, shape, sp)
        out[f"T{tag}"] = _smooth(raw[f"T{tag}"], n, sp.spread_sigma, shape, sp)
        out[f"n_{tag}"] = n.reshape(shape)
    return {
        "enodes": np.asarray(p.elevation_nodes, float),
        "GB": p.gamma_bin_deg,
        "ZB": p.zenith_bin_deg,
        **out,
    }


def _node_spread(
    bm: BinMeans,
    field: str,
    idx: Any,
    el: Any,
    table: Any,
    core: Any,
    p: BinParams,
    sp: SmoothParams,
) -> Any:
    """Robust sd over the frames ``idx`` of their residual to the clear table interpolated at
    their elevation, each frame's median residual over the ``core`` bins removed."""
    ne = len(p.elevation_nodes)
    nodes = np.asarray(p.elevation_nodes, float)
    x = np.interp(el[idx], nodes, np.arange(ne))
    e0 = np.minimum(np.floor(x).astype(int), ne - 2)
    t = (x - e0)[:, None]
    flat = table.reshape(ne, -1)
    ref = (1 - t) * flat[e0] + t * flat[e0 + 1]
    a = getattr(bm, field)[idx].astype(np.float32) - ref
    a = a - np.nanmedian(a[:, core], axis=1, keepdims=True)
    q = np.nanpercentile(a, [16, 84], axis=0)
    n = np.isfinite(a).sum(0)
    return np.where(n >= sp.table_min_frames, (q[1] - q[0]) / 2, np.nan)


def adapted_spread(
    bm: BinMeans, lab: Any, smooth: dict[str, Any], p: BinParams, sp: SmoothParams
) -> dict[str, Any]:
    """``Nc_sda`` and ``Lc_sda``: the clear-sky spread once the frame's own level is removed."""
    el = bm.elev
    cls = node_of(el, p.elevation_nodes)
    ne, nb = len(p.elevation_nodes), p.n_gamma * p.n_zenith
    gc, zc = bin_centres(p)
    core = (gc > sp.adapted_core_gamma) & (zc < sp.adapted_core_zenith)
    res: dict[str, Any] = {}
    for cue, field in (("N", "MN"), ("L", "ML")):
        out = np.full((ne, nb), np.nan, np.float32)
        with np.errstate(all="ignore"):
            for c in range(ne):
                idx = np.flatnonzero((cls == c) & (lab == CLEAR))
                if len(idx) < sp.table_min_frames:
                    continue
                out[c] = _node_spread(bm, field, idx, el, smooth[cue + "c"], core, p, sp)
        v = out.reshape(ne, p.n_gamma, p.n_zenith)
        w = np.where(np.isfinite(v), 1.0, 0.0)
        sig = sp.adapted_sigma
        num = ndimage.gaussian_filter(np.where(w > 0, v, 0), sig, mode="nearest")
        den = ndimage.gaussian_filter(w, sig, mode="nearest")
        s = num / np.maximum(den, 1e-9)
        ok = den > sp.adapted_den_min
        s = fill_nearest(np.where(ok, s, np.nan), ok)
        res[cue + "c_sda"] = s.astype(np.float32)
    return res


def rebuild_spreads(ref: dict[str, Any], sp: SmoothParams) -> dict[str, Any]:
    """The c14 spread tables: deviation from a heavy smooth version kept only to the extent it
    exceeds its statistical error s / sqrt(2 n) (Wiener factor d^2 / (d^2 + err^2))."""
    out: dict[str, Any] = {k: ref[k] for k in ("enodes", "GB", "ZB")}
    for k, nk in REBUILT_SPREADS.items():
        s = ref[k].astype(np.float64)
        n = np.minimum(ref[nk].astype(np.float64), sp.count_cap)
        w = (n > 0).astype(np.float64) + 1e-3
        sig = sp.rebuild_sigma
        sm = ndimage.gaussian_filter(s * w, sig, mode="nearest") / ndimage.gaussian_filter(
            w, sig, mode="nearest"
        )
        err = s / np.sqrt(2.0 * np.maximum(n, 1.0) * sp.rebuild_pool)
        d = s - sm
        f = d * d / (d * d + err * err + 1e-30)
        out[k] = (sm + f * d).astype(np.float32)
    return out
