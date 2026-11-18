"""Step 7, post-fit tables (prototype ``c1_postfit.py``, the joint-fit version of
``b7_postfit.py``).

Over the clear archive frames, the colour index's residual to the clear table AFTER the frame's
run-time level fit (``skycloud.level.fit_level``, joint on colour and brightness): per node and
bin, its median (``Nc_bias``, a bias of the level model) and its robust half-width (``Nc_sdf``,
the clear sky's real spread around the fitted level).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from skycloud.cues import cues, level_basis
from skycloud.geometry import angle_to_sun
from skycloud.level import NB, fit_level
from skycloud.params import CloudMapParams
from skycloud.reference_build.binning import bin_keys, frame_ok
from skycloud.reference_build.features import DayFeatures
from skycloud.reference_build.labels import CLEAR, node_of
from skycloud.reference_build.params import BinParams, SmoothParams
from skycloud.reference_build.raw_tables import column_percentiles
from skycloud.reference_build.smoothing import fill_nearest
from skycloud.tables import _Grid3

__all__ = ["frame_residual_bins", "postfit_tables", "table_grid"]


def table_grid(ref: dict[str, Any]) -> _Grid3:
    """The trilinear lookup of the renderer over a table dictionary."""
    names = [k for k, v in ref.items() if np.ndim(v) == 3]
    return _Grid3(
        {k: np.asarray(ref[k]).astype(np.float32) for k in names},
        np.asarray(ref["enodes"]).astype(np.float64),
        float(ref["GB"]),
        float(ref["ZB"]),
    )


def frame_residual_bins(
    f: Any, cor: Any, sun: Any, valid_n: Any, az: Any, zen: Any, ref: _Grid3,
    p: BinParams, sp: SmoothParams, cp: CloudMapParams,
) -> Any:  # fmt: skip
    """One clear frame: bin means (float16) of N - Nc - level, the level fitted jointly."""
    e = 90 - sun[1]
    m = frame_ok(f, valid_n, zen, p)
    g = angle_to_sun(az, zen, sun[0], sun[1])
    n_cue, l_cue = cues(f[:, 0], f[:, 1], f[:, 2], cor)
    nc, no, lc, lsd = ref.lookup(("Nc", "No", "Lc", "Lc_sd"), g, zen, e)
    use = m & (g > sp.postfit_fit_gamma) & (zen < sp.postfit_fit_zenith) & ((nc - no) > cp.con_min)
    if use.sum() >= sp.postfit_min_dirs:
        rl = (l_cue - lc)[use].astype(np.float64)
        beta = fit_level(
            (n_cue - nc)[use], level_basis(g[use], zen[use]), cp, rl, np.maximum(lsd[use], 1e-3)
        )[0]
    else:
        beta = np.zeros(NB)
    res = n_cue - nc - level_basis(g, zen) @ beta
    mm = m & (g > sp.postfit_bin_gamma)
    zi = np.clip((zen // p.zenith_bin_deg).astype(int), 0, p.n_zenith - 1)
    nb = p.n_gamma * p.n_zenith
    key = bin_keys(g[mm], zi[mm], p)
    n = np.bincount(key, minlength=nb)
    s = np.bincount(key, res[mm].astype(np.float64), nb)
    return np.where(n >= 2, s / np.maximum(n, 1), np.nan).astype(np.float16)


def _clear_rows(
    days: Iterable[DayFeatures], labels: dict[str, int], az: Any, zen: Any, ref: _Grid3,
    p: BinParams, sp: SmoothParams, cp: CloudMapParams,
) -> tuple[Any, Any]:  # fmt: skip
    el, rows = [], []
    for d in days:
        for i in range(len(d.ts)):
            e = 90 - d.sun[i, 1]
            if e < sp.postfit_min_elev or labels.get(str(d.ts[i]), 0) != CLEAR:
                continue
            rows.append(
                frame_residual_bins(
                    d.feat[i], d.corner[i], d.sun[i], d.valid_n, az, zen, ref, p, sp, cp
                )
            )
            el.append(e)
    return np.array(el), np.stack(rows)


def _smooth(v: Any, cnt: Any, sig: Any, shape: tuple[int, int, int], fill0: bool) -> Any:
    from scipy import ndimage

    v = v.reshape(shape)
    w = np.where(np.isfinite(v), np.minimum(cnt.reshape(shape), 60), 0.0)
    den = ndimage.gaussian_filter(w, sig, mode="nearest")
    num = ndimage.gaussian_filter(np.where(w > 0, v, 0) * w, sig, mode="nearest")
    s = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)
    ok = np.isfinite(s)
    if fill0:
        return np.where(ok, s, 0.0).astype(np.float32)
    return fill_nearest(s, ok).astype(np.float32)


def postfit_tables(
    days: Iterable[DayFeatures], labels: dict[str, int], az: Any, zen: Any,
    smooth: dict[str, Any], p: BinParams, sp: SmoothParams, cp: CloudMapParams,
) -> tuple[dict[str, Any], int]:  # fmt: skip
    """(``Nc_bias``, ``Nc_sdf``) and the number of clear frames read."""
    el, m = _clear_rows(days, labels, az, zen, table_grid(smooth), p, sp, cp)
    cls = node_of(el, p.elevation_nodes)
    ne, nb = len(p.elevation_nodes), p.n_gamma * p.n_zenith
    bias = np.full((ne, nb), np.nan, np.float32)
    sd = bias.copy()
    cnt = np.zeros((ne, nb))
    with np.errstate(all="ignore"):
        for c in range(ne):
            a = m[cls == c].astype(np.float32)
            if len(a) < sp.table_min_frames:
                continue
            n = np.isfinite(a).sum(0)
            q = column_percentiles(a, [16, 50, 84])
            cnt[c] = n
            bias[c] = np.where(n >= sp.table_min_frames, q[1], np.nan)
            sd[c] = np.where(n >= sp.table_min_frames, (q[2] - q[0]) / 2, np.nan)
    shape = (ne, p.n_gamma, p.n_zenith)
    out = {
        "Nc_bias": _smooth(bias, cnt, sp.bias_sigma, shape, fill0=True),
        "Nc_sdf": _smooth(sd, cnt, sp.sdf_sigma, shape, fill0=False),
    }
    return out, len(el)
