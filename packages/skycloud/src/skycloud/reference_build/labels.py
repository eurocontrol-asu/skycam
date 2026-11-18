"""Step 3, label-free clear / overcast labels of the archive frames (prototype
``b2_labels.py``). No human annotation:

1. per elevation node and bin, the archive's 85th and 10th percentiles of the bin-mean colour
   index N (clear-like and overcast-like);
2. per frame, the normalised gap a = (p85 - N) / (p85 - p10) on the bins where the colour cue
   separates (p85 - p10 above ``min_gap``, away from the sun and the horizon): clear when its
   90th percentile and median are small, overcast when its 10th percentile is large;
3. below ``episode_below`` of sun the colour cue is weak: a frame takes the label of its
   half-day episode, clear (overcast) when at least ``episode_share`` of the episode's frames
   between ``episode_window`` are clear (overcast).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from skycloud.reference_build.binning import BinMeans
from skycloud.reference_build.params import BinParams, LabelParams

__all__ = [
    "CLEAR",
    "OVERCAST",
    "FrameLabels",
    "bin_centres",
    "episode_labels",
    "frame_gap_quantiles",
    "label_frames",
    "node_of",
    "node_percentiles",
]

CLEAR, OVERCAST = 1, 2


def node_of(elev: Any, nodes: Any) -> Any:
    """Index of the nearest elevation node."""
    return np.abs(np.asarray(elev)[..., None] - np.asarray(nodes, float)).argmin(-1)


def bin_centres(p: BinParams) -> tuple[Any, Any]:
    """Angle to the sun and zenith angle at the centre of every flat bin."""
    gi, zi = np.divmod(np.arange(p.n_gamma * p.n_zenith), p.n_zenith)
    return (gi + 0.5) * p.gamma_bin_deg, (zi + 0.5) * p.zenith_bin_deg


def node_percentiles(
    mn: Any, cls: Any, n_nodes: int, n_bins: int, lp: LabelParams
) -> tuple[Any, Any]:
    """(P85, P10) float32 (nodes, bins): the archive's clear-like and overcast-like index."""
    p85 = np.full((n_nodes, n_bins), np.nan, np.float32)
    p10 = p85.copy()
    for c in range(n_nodes):
        m = mn[np.flatnonzero(cls == c)].astype(np.float32)
        n = np.isfinite(m).sum(0)
        with np.errstate(all="ignore"):
            q = np.nanpercentile(m, [lp.q_overcast, lp.q_clear], axis=0)
        p10[c] = np.where(n >= lp.node_min_frames, q[0], np.nan)
        p85[c] = np.where(n >= lp.node_min_frames, q[1], np.nan)
    return p85, p10


def frame_gap_quantiles(
    mn: Any, cls: Any, p85: Any, p10: Any, gc: Any, zc: Any, lp: LabelParams
) -> tuple[Any, Any, Any]:
    """Per frame the 10th, 50th, 90th percentiles of the normalised gap (NaN: not readable)."""
    q10, q50, q90 = (np.full(len(cls), np.nan) for _ in range(3))
    for c in range(p85.shape[0]):
        idx = np.flatnonzero(cls == c)
        use = (
            np.isfinite(p85[c]) & ((p85[c] - p10[c]) > lp.min_gap)
            & (gc > lp.min_gamma) & (zc < lp.max_zenith)
        )  # fmt: skip
        if use.sum() < lp.node_min_bins:
            continue
        a = (p85[c][use] - mn[idx][:, use].astype(np.float32)) / (p85[c][use] - p10[c][use])
        ok = np.isfinite(a).sum(1) > lp.frame_min_share * use.sum()
        with np.errstate(all="ignore"):
            q = np.nanpercentile(a, [10, 50, 90], axis=1)
        q10[idx], q50[idx], q90[idx] = (np.where(ok, v, np.nan) for v in q)
    return q10, q50, q90


def episode_labels(
    ts: Any, elev: Any, q50: Any, clear_f: Any, ovc_f: Any, lp: LabelParams
) -> tuple[NDArray[np.int8], dict[str, int]]:
    """The final label of every frame (1 clear, 2 overcast, 0 neither) and the half-day
    episodes' labels."""
    split = lp.half_day_split_utc_hour
    half = np.array([t[:8] + ("am" if int(t[8:10]) < split else "pm") for t in ts])
    e0, e1 = lp.episode_window
    ep: dict[str, int] = {}
    for h in np.unique(half):
        m = (half == h) & (elev >= e0) & (elev <= e1) & np.isfinite(q50)
        if m.sum() >= lp.episode_min_frames:
            ep[str(h)] = (
                CLEAR if clear_f[m].mean() >= lp.episode_share
                else OVERCAST if ovc_f[m].mean() >= lp.episode_share else 0
            )  # fmt: skip
    lab = np.zeros(len(ts), np.int8)
    for i in range(len(ts)):
        if elev[i] >= lp.episode_below:
            lab[i] = CLEAR if clear_f[i] else OVERCAST if ovc_f[i] else 0
        else:
            lab[i] = ep.get(str(half[i]), 0)
    return lab, ep


@dataclass(frozen=True)
class FrameLabels:
    """Labels of the archive frames, with the statistics they were decided on."""

    ts: Any
    lab: NDArray[np.int8]
    q10: Any
    q50: Any
    q90: Any
    el: Any
    p85: Any
    p10: Any
    episodes: dict[str, int]

    def by_stamp(self) -> dict[str, int]:
        """Label of every timestamp (a duplicated stamp keeps its last row, as the prototype's
        ``dict(zip(ts, lab))``)."""
        return {str(t): int(v) for t, v in zip(self.ts, self.lab, strict=True)}


def label_frames(bm: BinMeans, p: BinParams, lp: LabelParams) -> FrameLabels:
    """Label every frame of the bin means."""
    el = bm.elev
    cls = node_of(el, p.elevation_nodes)
    gc, zc = bin_centres(p)
    nb = p.n_gamma * p.n_zenith
    p85, p10 = node_percentiles(bm.MN, cls, len(p.elevation_nodes), nb, lp)
    q10, q50, q90 = frame_gap_quantiles(bm.MN, cls, p85, p10, gc, zc, lp)
    clear_f = (q90 < lp.clear_q90_max) & (q50 < lp.clear_q50_max)
    ovc_f = q10 > lp.overcast_q10_min
    lab, ep = episode_labels(bm.ts, el, q50, clear_f, ovc_f, lp)
    return FrameLabels(bm.ts, lab, q10, q50, q90, el, p85, p10, ep)
