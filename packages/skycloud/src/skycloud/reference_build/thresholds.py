"""Step 8, the small threshold tables read at run time (JSON), from the archive frames' labels
and the finished reference tables:

- ``c7_shape`` (prototype ``c7_shape_stats.py``): per sun band, the 90th percentile over the
  clear frames (R0) and the 10th over the overcast frames (R1) of the brightness-shape ratio
  sd(L - Lc) / sd(L - Lo);
- ``c8_release`` (``c8_shape_gate.py``): the joint clear-shape support of the low-sun frames,
  its maximum over the overcast frames (T_o) and its median over the clear frames (T_c);
- ``c11_lift_colour`` (``c7_lowsun_levels.py``, transcribed by hand in round c11): per sun
  band, the clear frames' 95th and the overcast frames' 5th percentile of the frame median of
  the raw colour opacity (Nc - N) / (Nc - No);
- ``c9_knowable`` (``c9_knowable.py``): the sun elevations from which the diffuse knowability
  K of the tables alone stays above the unknown threshold;
- ``c6_midscale`` (``c6_midstats.py``): the clear sky's own spread of the 1-3 deg band-pass of
  the log brightness, from full frames;
- ``lowcount`` (``c3_lowcount_table.py``, ``c4_lowcount_ped.py``): the colour index's extra
  spread across camera gain steps against the window signal, from 30 s frame pairs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from skycloud.cues import PED_BR, cues, mad, ped_g
from skycloud.geometry import angle_to_sun
from skycloud.level import shape_support_joint
from skycloud.params import CloudMapParams
from skycloud.reference_build.binning import frame_ok
from skycloud.reference_build.features import DayFeatures
from skycloud.reference_build.labels import CLEAR, OVERCAST
from skycloud.reference_build.params import BinParams, LowCountParams, ThresholdParams
from skycloud.tables import _Grid3

__all__ = [
    "FrameStats",
    "frame_stats",
    "knowable",
    "lift_colour",
    "lowcount",
    "midscale_frame",
    "midscale_table",
    "release_gate",
    "shape_table",
]


@dataclass(frozen=True)
class FrameStats:
    """Per labelled frame: sun elevation, label, brightness-shape ratio, joint clear-shape
    support, frame median of the raw colour opacity (NaN where the frame is not read)."""

    ts: list[str]
    elev: Any
    lab: Any
    shape: Any
    support: Any
    opacity: Any


def _shape_and_support(
    f: Any, cor: Any, base: Any, g: Any, zen: Any, e: float, ref: _Grid3, tp: ThresholdParams
) -> tuple[float, float]:
    m = base & (zen < 75) & (g > 15)
    if m.sum() < tp.frame_min_dirs:
        return np.nan, np.nan
    n_cue, l_cue = cues(f[:, 0], f[:, 1], f[:, 2], cor)
    lc, lo, nc, sn, sl = ref.lookup(("Lc", "Lo", "Nc", "Nc_sda", "Lc_sda"), g, zen, e)
    # at the cues' precision (float32), as c7_shape_stats.py divided its robust sds
    ratio = np.float32(mad((l_cue - lc)[m])) / np.maximum(np.float32(mad((l_cue - lo)[m])), 1e-4)
    sup = np.nan
    if tp.release_elev[0] <= e <= tp.release_elev[1]:
        sup = shape_support_joint((n_cue - nc)[m], sn[m], (l_cue - lc)[m], sl[m])[0]
    return float(ratio), float(sup)


def _opacity(
    f: Any, cor: Any, base: Any, g: Any, zen: Any, e: float, ref: _Grid3, tp: ThresholdParams
) -> float:
    m = base & (zen < 60) & (g > 15)
    if m.sum() < tp.frame_min_dirs:
        return np.nan
    n_cue, _ = cues(f[:, 0], f[:, 1], f[:, 2], cor)
    nc, no = ref.lookup(("Nc", "No"), g, zen, e)
    return float(np.median(((nc - n_cue) / np.maximum(nc - no, 0.02))[m]))


def frame_stats(
    days: Iterable[DayFeatures], labels: dict[str, int], az: Any, zen: Any, ref: _Grid3,
    p: BinParams, tp: ThresholdParams,
) -> FrameStats:  # fmt: skip
    """The per-frame statistics of the c7, c8 and c11 tables over the labelled frames."""
    lo_e = min(tp.shape_bands[0][0], tp.lift_bands[0][0], tp.release_elev[0])
    hi_e = max(tp.lift_bands[-1][1], tp.release_elev[1], tp.shape_bands[-1][1])
    rows = []
    for d in days:
        for i in range(len(d.ts)):
            e = 90 - d.sun[i, 1]
            lab = labels.get(str(d.ts[i]), 0)
            if e < lo_e or e > hi_e or lab not in (CLEAR, OVERCAST):
                continue
            f = d.feat[i]
            g = angle_to_sun(az, zen, d.sun[i, 0], d.sun[i, 1])
            base = frame_ok(f, d.valid_n, zen, p)
            shape, sup = _shape_and_support(f, d.corner[i], base, g, zen, e, ref, tp)
            opa = _opacity(f, d.corner[i], base, g, zen, e, ref, tp)
            rows.append((str(d.ts[i]), e, lab, shape, sup, opa))
    cols: list[Any] = [list(c) for c in zip(*rows, strict=True)] if rows else [[]] * 6
    ts, elev, lab, shape, sup, opa = cols
    return FrameStats(ts, *(np.asarray(c, float) for c in (elev, lab, shape, sup, opa)))


def shape_table(st: FrameStats, tp: ThresholdParams) -> dict[str, Any]:
    """c7_shape: per band R0 (clear p90) and R1 (overcast p10) of the shape ratio."""
    tab: dict[str, Any] = {"elev": [], "R0": [], "R1": []}
    q0, q1 = tp.shape_quantiles
    ok = np.isfinite(st.shape)
    for e0, e1 in tp.shape_bands:
        band = ok & (st.elev >= e0) & (st.elev < e1)
        c, o = band & (st.lab == CLEAR), band & (st.lab == OVERCAST)
        if c.sum() < tp.shape_min_frames or o.sum() < tp.shape_min_frames:
            continue
        tab["elev"].append(0.5 * (e0 + e1))
        tab["R0"].append(float(np.percentile(st.shape[c], q0)))
        tab["R1"].append(float(np.percentile(st.shape[o], q1)))
    return tab


def release_gate(st: FrameStats) -> dict[str, Any]:
    """c8_release: T_o = largest support of an overcast frame, T_c = median of the clear."""
    ok = np.isfinite(st.support)
    return {
        "T_o": float(st.support[ok & (st.lab == OVERCAST)].max()),
        "T_c": float(np.median(st.support[ok & (st.lab == CLEAR)])),
    }


def lift_colour(st: FrameStats, tp: ThresholdParams) -> dict[str, Any]:
    """c11_lift_colour: per band the clear p95 and the overcast p5 of the frame opacity."""
    qc, qo = tp.lift_quantiles
    ok = np.isfinite(st.opacity)
    tab: dict[str, Any] = {"clear_p95": [], "e": [], "ov_p5": []}
    for e0, e1 in tp.lift_bands:
        band = ok & (st.elev >= e0) & (st.elev < e1)
        c, o = band & (st.lab == CLEAR), band & (st.lab == OVERCAST)
        if c.sum() < tp.lift_min_frames:
            continue
        tab["e"].append(0.5 * (e0 + e1))
        tab["clear_p95"].append(round(float(np.percentile(st.opacity[c], qc)), tp.lift_decimals))
        ov = float(np.percentile(st.opacity[o], qo)) if o.sum() >= tp.lift_min_frames else np.nan
        tab["ov_p5"].append(round(ov, tp.lift_decimals))
    return tab


def _k_of(
    ref: _Grid3, az: Any, zen: Any, e: float, u: float, tp: ThresholdParams, cp: CloudMapParams
) -> Any:
    g = angle_to_sun(az, zen, tp.knowable_sun_azimuth, 90.0 - e)
    names = ("Nc", "No", "Lc", "Lo", "Nc_sdf", "Lc_sda", "Nc_sd", "Lc_sd", "No_sd", "Lo_sd")
    nc, no, lc, lo, snf, sla, snu, slu, sno, slo = ref.lookup(names, g, zen, e)
    con = np.maximum(nc - no, 0.0)
    snt2 = snf**2 + u * snu**2
    slt2 = sla**2 + u * slu**2
    wnc = (con / np.maximum(sno, 1e-3)) ** 2
    back = 1.0 - np.clip(1.0 - 2.0 / np.sqrt(wnc + 1e-6), 0.0, 1.0)
    d = (
        con**2 / (snt2 + np.maximum(sno, 1e-3) ** 2)
        + cp.kappa_clear * (lo - lc) ** 2 / (slt2 + np.maximum(slo, 1e-2) ** 2) * back
    )
    m = (zen < tp.knowable_max_zenith) & (g > tp.knowable_min_gamma)
    return (d / (d + cp.fill_eps))[m]


def _first_stable(elevs: Any, p5: Any, level: float) -> float:
    """The lowest elevation from which the K percentile stays at or above ``level``."""
    for j, e in enumerate(elevs):
        if all(v >= level for v in p5[j:]):
            return float(e)
    return float(elevs[-1])


def knowable(
    ref: _Grid3, az: Any, zen: Any, tp: ThresholdParams, cp: CloudMapParams
) -> dict[str, Any]:
    """c9_knowable: E_030_unsupported, E_030_supported and E_K (the comfortable level)."""
    e0, e1, de = tp.knowable_elevations
    elevs = np.arange(e0, e1 + 0.01, de)
    lo, hi = tp.knowable_levels
    p5 = {
        u: [
            float(np.percentile(_k_of(ref, az, zen, e, u, tp, cp), tp.knowable_share))
            for e in elevs
        ]
        for u in (1.0, 0.0)
    }
    return {
        "E_K": _first_stable(elevs, p5[1.0], hi),
        "E_030_unsupported": _first_stable(elevs, p5[1.0], lo),
        "E_030_supported": _first_stable(elevs, p5[0.0], lo),
    }


def midscale_frame(
    img: Any, e: float, g: Any, zen: Any, static: Any, px_deg: float, tp: ThresholdParams
) -> tuple[float, float, Any] | None:
    """One clear frame: (elevation, robust sd of the 1-3 deg band-pass of the log brightness,
    quantiles 99 / 99.5 / 99.9 of |bp| / sd) or None when too little sky is read."""
    a = np.asarray(img, np.float32)
    k = 60
    c = np.concatenate(
        [a[:k, :k].reshape(-1, 3), a[:k, -k:].reshape(-1, 3),
         a[-k:, :k].reshape(-1, 3), a[-k:, -k:].reshape(-1, 3)]
    ).mean(0)  # fmt: skip
    pg = float(ped_g(c))
    lg = np.log(np.maximum(a[..., 1] - pg, 0.02 * pg) / pg)
    m = (zen < tp.midscale_max_zenith) & (a.max(-1) < 250) & (g > tp.midscale_min_gamma) & static
    lg2, m2 = lg[::2, ::2], m[::2, ::2]
    mm = m2.astype(np.float32)
    s1, s2 = px_deg / 2, 3 * px_deg / 2

    def nb(s: float) -> Any:
        num = cv2.GaussianBlur(np.where(m2, lg2, 0).astype(np.float32), (0, 0), s)
        return num / np.maximum(cv2.GaussianBlur(mm, (0, 0), s), 1e-3)

    bp, cov = nb(s1) - nb(s2), cv2.GaussianBlur(mm, (0, 0), s2)
    x = bp[m2 & (cov > 0.9)]
    if x.size < tp.midscale_min_pixels:
        return None
    sd = 1.4826 * np.median(np.abs(x - np.median(x)))
    return e, float(sd), np.percentile(np.abs(x) / max(sd, 1e-6), [99, 99.5, 99.9])


def midscale_static(sky: Any, px_deg: float, tp: ThresholdParams) -> Any:
    """The a-priori sky mask eroded by the band-pass's outer scale."""
    k = round(tp.midscale_erode_deg * px_deg) * 2 + 1
    return cv2.erode(np.asarray(sky).astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def midscale_table(rows: list[tuple[float, float, Any]], tp: ThresholdParams) -> dict[str, Any]:
    """c6_midscale: per band the median and 90th percentile of the frames' sd and the median
    of their 99.5th percentile of |bp| / sd (the detection threshold K)."""
    e = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    q = np.array([r[2] for r in rows])
    tab: dict[str, Any] = {"elev": [], "sd": [], "sd_p90": [], "K": []}
    for e0, e1 in zip(tp.midscale_bands[:-1], tp.midscale_bands[1:], strict=True):
        b = (e >= e0) & (e < e1)
        if b.sum() < 3:
            continue
        tab["elev"].append(0.5 * (e0 + e1))
        tab["sd"].append(float(np.median(s[b])))
        tab["sd_p90"].append(float(np.percentile(s[b], 90)))
        tab["K"].append(float(np.median(q[b, 1])))
    return tab


def _pair_changes(
    a: Any, b: Any, cor: Any, base: Any, p: BinParams, lp: LowCountParams
) -> tuple[Any, Any, Any]:
    m = base & (a[:, 3] == 0) & (b[:, 3] == 0)
    m &= (a[:, :3].max(1) < p.bright_max) & (b[:, :3].max(1) < p.bright_max)
    m &= a[:, 4] / np.maximum(a[:, 1], 1) < lp.smooth_max
    m &= b[:, 4] / np.maximum(b[:, 1], 1) < lp.smooth_max
    na = (a[m, 2] - a[m, 0]) / np.maximum(a[m, 2] + a[m, 0], 1)
    nb = (b[m, 2] - b[m, 0]) / np.maximum(b[m, 2] + b[m, 0], 1)
    s_counts = np.minimum(a[m, 0] + a[m, 2], b[m, 0] + b[m, 2])
    s_ped = (a[m, 0] + a[m, 2]) / (cor[0] * (1 + PED_BR))
    return np.abs(nb - na), s_counts, s_ped


def lowcount(
    days: Iterable[DayFeatures], zen: Any, p: BinParams, lp: LowCountParams,
    on_pair: Callable[[str, str], None] | None = None,
) -> dict[str, dict[str, Any]]:  # fmt: skip
    """The c3 (raw counts) and c4 (pedestal units) tables: per signal bin the 90th percentile
    of the 30 s change |dN| of pairs with and without a gain step, and the excess in
    quadrature e = sqrt(q_step^2 - q_none^2). Pairs are consecutive frames of a day file."""
    edges = {"counts": np.array(lp.edges_counts), "pedestal": np.array(lp.edges_pedestal)}
    acc: dict[str, dict[str, list[list[Any]]]] = {
        k: {"step": [[] for _ in v[:-1]], "none": [[] for _ in v[:-1]]} for k, v in edges.items()
    }
    rng = np.random.default_rng(lp.seed)
    for d in days:
        base = (d.valid_n >= p.window_min_sky) & (zen < lp.max_zenith)
        for i in range(len(d.ts) - 1):
            if 90 - d.sun[i, 1] < lp.min_elev:
                continue
            st = abs(np.log(d.corner[i + 1, 0] / d.corner[i, 0]))
            key = "step" if st > lp.step_min else ("none" if st < lp.none_max else None)
            if key is None or (key == "none" and rng.random() > lp.none_keep):
                continue
            a, b = d.feat[i].astype(np.float64), d.feat[i + 1].astype(np.float64)
            dn, s_counts, s_ped = _pair_changes(a, b, d.corner[i], base, p, lp)
            if on_pair is not None:
                on_pair(str(d.ts[i]), str(d.ts[i + 1]))
            for name, s in (("counts", s_counts), ("pedestal", s_ped)):
                e = edges[name]
                k = np.clip(np.searchsorted(e, s, side="right") - 1, 0, len(e) - 2)
                for j in range(len(e) - 1):
                    x = dn[k == j]
                    if x.size:
                        acc[name][key][j].append(x[:: lp.subsample])
    return {name: _lowcount_table(acc[name], edges[name], lp) for name in edges}


def _lowcount_table(
    acc: dict[str, list[list[Any]]], edges: Any, lp: LowCountParams
) -> dict[str, Any]:
    def q(v: list[Any]) -> float:
        n = sum(len(x) for x in v)
        return (
            float(np.percentile(np.concatenate(v), lp.quantile))
            if v and n > lp.min_samples
            else np.nan
        )

    qs = {k: np.array([q(v) for v in acc[k]]) for k in acc}
    e = np.sqrt(np.maximum(qs["step"] ** 2 - qs["none"] ** 2, 0.0))
    return {
        "S": 0.5 * (edges[:-1] + edges[1:]),
        "e": np.nan_to_num(e),
        "q_step": qs["step"],
        "q_none": qs["none"],
    }
