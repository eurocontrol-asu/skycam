"""The frame's clear-sky level (method stage "clear-level fit") and its smoothing in time.

Per frame, on the archive's 16,200 grid directions (9 x 9 px window means of the sky): the
clear-sky level of the colour index -- a constant, two solar-side terms and a horizon-haze term
-- is found by mode seeking around the archive's clear sky with a prior (it cannot follow a sky
further than ~2 sigma from it: an overcast is never taken for the clear sky), JOINTLY with a
brightness level (a direction supports the level only if its colour and its brightness are
both near the clear sky's). The frame state also carries its brightness-shape ratio, its 5th
percentile and median of raw colour opacity and its joint clear-shape support, all smoothed in
time over the neighbouring frames (``smooth_states``), each frame weighted by its support.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from skycloud.cues import cues, level_basis, mad
from skycloud.geometry import angle_to_sun
from skycloud.images import frame_path, pedestal, stamp_minutes, when_of
from skycloud.params import CloudMapParams
from skycloud.tables import ReferenceTables

__all__ = [
    "NB",
    "Grid",
    "context_frames",
    "fit_level",
    "frame_state",
    "shape_support_joint",
    "smooth_states",
]

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
B = NDArray[np.bool_]
U8 = NDArray[np.uint8]

NB = 4  # level coefficients
WIN = 9  # px: the window of a grid direction
_SUPJ_N = np.linspace(-0.15, 0.15, 121)
_SUPJ_L = np.linspace(-0.5, 0.5, 101)


@dataclass(frozen=True)
class Grid:
    """The archive's grid directions on the frame: the pixels of each 9 x 9 window, which of
    them are sky (the static mask), and their azimuth and zenith."""

    idx: NDArray[np.int64]
    m: B
    n: F32
    ok0: B
    az: F64
    zen: F64

    @classmethod
    def build(cls, tables: ReferenceTables, shape: tuple[int, int], sky: B) -> Grid:
        """Windows around the grid pixels of ``tables``, clipped to a frame of ``shape``;
        ``sky``: the frame's static sky mask."""
        h, w = shape
        y, x = np.divmod(tables.grid_gp, w)
        off = np.arange(WIN) - WIN // 2
        yy = np.clip(y[:, None, None] + off[None, :, None], 0, h - 1)
        xx = np.clip(x[:, None, None] + off[None, None, :], 0, w - 1)
        idx = (yy * w + xx).reshape(len(y), WIN * WIN)
        m = sky.ravel()[idx]
        zen = tables.grid_zen.astype(np.float64)
        return cls(
            idx=idx,
            m=m,
            n=np.maximum(m.sum(1), 1).astype(np.float32),
            ok0=(m.sum(1) >= 70) & (zen < 88.5),
            az=tables.grid_az.astype(np.float64),
            zen=zen,
        )

    def features(self, img: U8, sat: int) -> tuple[Any, Any]:
        """R, G, B window means (sky pixels) and saturated share on the grid directions."""
        x = img.reshape(-1, 3)[self.idx].astype(np.float32)
        rgb = (x * self.m[..., None]).sum(1) / self.n[:, None]
        share = ((x.max(-1) >= sat) & self.m).sum(1) / self.n
        return rgb, share

    def centre_values(self, full: F32) -> Any:
        """A full-resolution field read at the centre pixel of each window."""
        return full.ravel()[self.idx[:, self.idx.shape[1] // 2]]


def fit_level(r: F64, x: F64, p: CloudMapParams, rl: Any = None, sl: Any = None) -> Any:
    """Mode seeking of the level surface around the archive's clear sky (Welsch weights,
    kernel s_n) with a prior at 0 scaled by the archive's clear-sky spread of each coefficient,
    bounded. With ``rl`` (L - L_clear) and ``sl`` (its archive clear spread) the seeking is
    JOINT and also returns the brightness level and the weights: (beta, aL, weights)."""
    s = p.s_n
    b_sd = np.array(p.b_sd)
    lam = p.prior * len(r) * (s / b_sd) ** 2
    lim = p.b_lim * b_sd
    beta = np.zeros(x.shape[1])
    al = 0.0
    joint = rl is not None and sl is not None
    lam_l = lam[0]
    lo_l, hi_l = p.al_lim
    if joint:
        beta = fit_level(r, x, p)
        wn = np.exp(-0.5 * ((r - x @ beta) / s) ** 2)
        grid_l = np.linspace(lo_l, hi_l, 2 * round((hi_l - lo_l) / 0.02) + 1)
        sup = [(wn * np.exp(-0.5 * ((rl - a) / sl) ** 2)).mean() for a in grid_l]
        al = float(grid_l[int(np.argmax(sup))])
    for _ in range(60):
        w = np.exp(-0.5 * ((r - x @ beta) / s) ** 2)
        if joint:
            w = w * np.exp(-0.5 * ((rl - al) / sl) ** 2)
        lhs = (x * w[:, None]).T @ x + np.diag(lam)
        new = np.clip(np.linalg.lstsq(lhs, (x * w[:, None]).T @ r, rcond=None)[0], -lim, lim)
        new_al = float(np.clip((w * rl).sum() / (w.sum() + lam_l), lo_l, hi_l)) if joint else 0.0
        if np.abs(new - beta).max() < 1e-5 and abs(new_al - al) < 1e-4:
            beta, al = new, new_al
            break
        beta, al = new, new_al
    if not joint:
        return beta
    w = np.exp(-0.5 * ((r - x @ beta) / s) ** 2) * np.exp(-0.5 * ((rl - al) / sl) ** 2)
    return beta, al, w


def _shape_ratio(lum: Any, lc: Any, lo: Any, g: Any, zen: Any, ok: B) -> float | None:
    """c8: robust sd of L - Lc over that of L - Lo (zenith < 75, > 15 deg from the sun)."""
    m = ok & (zen < 75) & (g > 15)
    if m.sum() < 200:
        return None
    sc = mad((lum - lc)[m])
    so = mad((lum - lo)[m])
    return float(sc / max(so, 1e-4))


def shape_support_joint(dn: Any, sn: Any, dl: Any, sl: Any) -> tuple[float, float, float]:
    """c8: the largest share of directions whose colour AND brightness departures from the
    clear tables lie within 2 adapted clear spreads of one common pair of offsets."""
    if len(dn) < 200:
        return 0.0, 0.0, 0.0
    in_n = (np.abs(dn[:, None] - _SUPJ_N[None, :]) < 2.0 * np.maximum(sn, 0.004)[:, None]).astype(
        np.float32
    )
    in_l = (np.abs(dl[:, None] - _SUPJ_L[None, :]) < 2.0 * np.maximum(sl, 0.01)[:, None]).astype(
        np.float32
    )
    f = in_n.T @ in_l / len(dn)
    k = np.unravel_index(int(np.argmax(f)), f.shape)
    return float(f[k]), float(_SUPJ_N[k[0]]), float(_SUPJ_L[k[1]])


@dataclass(frozen=True)
class _GridCues:
    n: Any
    lum: Any
    g: Any
    nc: Any
    no: Any
    lc: Any
    lc_sd: Any
    clean: B


def _low_sun_stats(
    out: dict[str, Any],
    c: _GridCues,
    grid: Grid,
    t: ReferenceTables,
    g_elev: tuple[F64, float],
    con_floor: float,
) -> None:
    """c10: the brightness-shape ratio, the 5th percentile of raw colour opacity and the joint
    clear-shape support, below +12 deg (smoothed in time by smooth_states)."""
    g, elev = g_elev
    zen = grid.zen
    lo = t.ref.lookup(("Lo",), g, zen, elev)[0]
    rg = _shape_ratio(c.lum, c.lc, lo, g, zen, c.clean)
    if rg is not None:
        out["ratio"] = float(rg)
    mq = c.clean & (zen < 80) & (g > 12)
    if mq.sum() >= 200:
        opa = (c.nc - c.n) / np.maximum(c.nc - c.no, con_floor)
        out["opa_q5"] = float(np.percentile(opa[mq], 5))
    nsa, lsa = t.ref.lookup(("Nc_sda", "Lc_sda"), g, zen, elev)
    mj = c.clean & (zen < 75) & (g > 15)
    if mj.sum() >= 200:
        sup = shape_support_joint((c.n - c.nc)[mj], nsa[mj], (c.lum - c.lc)[mj], lsa[mj])
        out["joint"] = float(sup[0])


def frame_state(
    img: U8,
    saz: float,
    szen: float,
    grid: Grid,
    tables: ReferenceTables,
    p: CloudMapParams,
    offset_full: F32 | None,
) -> dict[str, Any]:
    """The frame's own clear-sky level of each cue (before smoothing in time), its support
    and its frame statistics. ``offset_full``: the gain-ledger offset of N (full resolution)."""
    elev = 90.0 - szen
    out: dict[str, Any] = {
        "elev": round(elev, 3),
        "aN": 0.0,
        "fN": 0.0,
        "aL": 0.0,
        "fL": 0.0,
        "beta": [0.0] * NB,
    }
    if elev < p.elev_min:
        return out
    ped = pedestal(img)
    out["pedR"] = float(ped[0])
    rgb, sat = grid.features(img, p.sat)
    g = angle_to_sun(grid.az, grid.zen, saz, szen)
    n, lum = cues(rgb[:, 0], rgb[:, 1], rgb[:, 2], ped)
    if offset_full is not None:
        n = n + grid.centre_values(offset_full)
    nc, no, lc, lsd = tables.ref.lookup(("Nc", "No", "Lc", "Lc_sd"), g, grid.zen, elev)
    clean = grid.ok0 & (sat == 0) & (rgb.max(1) < 245)
    c = _GridCues(n, lum, g, nc, no, lc, lsd, clean)
    if elev < 12.0:
        _low_sun_stats(out, c, grid, tables, (g, elev), p.con_floor)
    use = clean & (g > 12) & (grid.zen < 86) & ((nc - no) > p.con_min)
    out["n"] = int(use.sum())
    if use.sum() < 400:
        return out
    ma = clean & (grid.zen < 60) & (g > 15)
    if ma.sum() >= 200:
        out["opa_med"] = float(np.median(((nc - n) / np.maximum(nc - no, p.con_floor))[ma]))
    beta, al, w = fit_level(
        (n - nc)[use],
        level_basis(g[use], grid.zen[use]),
        p,
        (lum - lc)[use],
        np.maximum(lsd[use], 1e-3),
    )
    f_n = float(w.mean())
    out["beta"], out["aN"], out["fN"] = [float(x) for x in beta], float(beta[0]), f_n
    out["aL"], out["fL"] = al, f_n
    return out


LEGACY_KEYS = ("aN", "fN", "aL", "fL", "beta", "n", "pedR", "elev")


def _smoothed(kern: F64, vals: F64, ok: B) -> tuple[float, bool]:
    w = kern * ok
    if w.sum() > 0:
        return float((w * np.nan_to_num(vals)).sum() / w.sum()), True
    return 0.0, False


def _smooth_frame_stats(
    o: dict[str, Any], k: str, states: dict[str, Any], ts: list[str], dt: F64, p: CloudMapParams
) -> None:
    """The time-smoothed frame statistics of target ``k``."""
    om = np.array([states[x].get("opa_med", np.nan) for x in ts])
    v, ok = _smoothed(np.exp(-0.5 * (dt / 2.0) ** 2), om, np.isfinite(om))
    if ok:
        o["opa_med"] = v
    rr = np.array([states[x].get("ratio", np.nan) for x in ts])
    wk = np.exp(-0.5 * (dt / p.t_shape) ** 2) * np.isfinite(rr)
    if wk.sum() > 0 and k in states and np.isfinite(states[k].get("ratio", np.nan)):
        lr = np.log(np.where(np.isfinite(rr), np.maximum(rr, 1e-4), 1.0))
        o["ratio_s"] = float(np.exp((wk * lr).sum() / wk.sum()))
        o["ratio_g"] = float(states[k]["ratio"])
    qq = np.array([states[x].get("opa_q5", np.nan) for x in ts])
    v, ok = _smoothed(np.exp(-0.5 * (dt / p.t_shape) ** 2), qq, np.isfinite(qq))
    if ok:
        o["q5_s"] = v
    jj = np.array([states[x].get("joint", np.nan) for x in ts])
    v, ok = _smoothed(np.exp(-0.5 * (dt / p.t_amb) ** 2), jj, np.isfinite(jj))
    if ok:
        o["joint_s"] = v


def _own_gate(k: str, states: dict[str, Any], p: CloudMapParams) -> float:
    """c2: a frame whose colour can be read but where no direction is jointly clear does not
    import other frames' level: their weight is scaled by f / (f + f_half)."""
    if k in states and states[k].get("n", 0) >= 400:
        f_self = states[k]["fN"]
        f_half = next(v for e1, v in p.f_half_bands if states[k]["elev"] < e1 or e1 == 90.0)
        return float(f_self / (f_self + f_half))
    return 1.0


def _smooth_core(
    states: dict[str, Any], targets: Iterable[str], p: CloudMapParams
) -> dict[str, Any]:
    ts = sorted(states)
    t = np.array([stamp_minutes(k) for k in ts])
    out = {}
    f_all = {
        "fN": np.array([states[k]["fN"] for k in ts]),
        "fL": np.array([states[k]["fL"] for k in ts]),
    }
    vals: dict[str, tuple[F64, str]] = {"aL": (np.array([states[k]["aL"] for k in ts]), "fL")}
    for j in range(NB):
        vals[f"b{j}"] = (np.array([states[k]["beta"][j] for k in ts]), "fN")
    lpr = np.log(np.array([states[k].get("pedR", np.nan) for k in ts], np.float64))
    for k in targets:
        dt = stamp_minutes(k) - t
        kern = np.exp(-0.5 * (dt / p.t_sig) ** 2) + p.t_far_w * np.exp(-0.5 * (dt / p.t_far) ** 2)
        kern[np.abs(dt) > 3 * p.t_far] = 0
        near = np.exp(-0.5 * (dt / p.t_sig) ** 2)
        gate = _own_gate(k, states, p)
        if k in states and states[k].get("n", 0) >= 400:
            other = np.array([x != k for x in ts])
            kern = np.where(other, kern * gate, kern)
            near = np.where(other, near * gate, near)
        o: dict[str, Any] = {"gate": round(float(gate), 4)}
        for key, (a, fkey) in vals.items():
            wsum = (kern * f_all[fkey]).sum()
            o[key] = float((kern * f_all[fkey] * a).sum() / (wsum + p.t_prior))
            o[fkey] = float((near * f_all[fkey]).sum())
        if k in states and np.isfinite(lpr).any():
            j = ts.index(k)
            nb = (np.abs(dt) <= 2.0) & (np.arange(len(ts)) != j) & np.isfinite(lpr)
            ok = nb.sum() >= 2 and np.isfinite(lpr[j])
            o["gain_dev"] = float(abs(lpr[j] - np.median(lpr[nb]))) if ok else 0.0
        _smooth_frame_stats(o, k, states, ts, dt, p)
        o["beta"] = [o.pop(f"b{j}") for j in range(NB)]
        o["aN"] = o["beta"][0]
        out[k] = o
    return out


def smooth_states(
    states: dict[str, Any], targets: Iterable[str], p: CloudMapParams
) -> dict[str, Any]:
    """The level of each target from all the frames' own levels: kernel-weighted mean in time
    (a narrow Gaussian plus a faint wide one), each frame weighted by its support, a weak prior
    at 0. c10: daytime targets (sun >= 10 deg) read the levels obtained with the legacy ledger."""
    targets = list(targets)
    if any("legacy" in v for v in states.values()):
        day_t = [k for k in targets if k in states and states[k].get("elev", -99) >= 10.0]
        if day_t:
            leg = {k: ({**v, **v["legacy"]} if "legacy" in v else v) for k, v in states.items()}
            o_day = _smooth_core(leg, day_t, p)
            rest = [k for k in targets if k not in set(day_t)]
            o_rest = _smooth_core(states, rest, p)
            return {**o_rest, **o_day}
    return _smooth_core(states, targets, p)


def context_frames(
    targets: list[str], root: Path, minutes: float, step: int, exclude: frozenset[str] = frozenset()
) -> list[str]:
    """The existing frames within ``minutes`` of a target, thinned to one per ``step`` seconds
    (targets kept); excluded frames are never read."""
    days = sorted({t[:8] for t in targets})
    out = set(targets)
    for d in days:
        folder = frame_path(root, d + "000000").parent
        have = sorted(
            t for t in (f.stem[6:20] for f in folder.glob("image_*.jp2")) if t not in exclude
        )
        tt = np.array([when_of(t).timestamp() for t in targets if t[:8] == d])
        last = -1e18
        for t in have:
            x = when_of(t).timestamp()
            if np.abs(tt - x).min() <= minutes * 60 and x - last >= step:
                out.add(t)
                last = x
    return sorted(out - set(exclude))
