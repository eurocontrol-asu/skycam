"""The camera's gain-step ledger of a day (method stage "gain ledger", rounds c5 and c10).

Across a camera gain step the raw colour index N shifts by 0.01-0.02, spatially coherent,
count-dependent, of varying sign; nothing else moves. Per day:

* pass 1: the R-corner pedestal of every frame with the sun above -4 deg;
* steps: consecutive frames (<= 90 s apart, sun below ``e_period``) whose |d log pedestal|
  exceeds T_STEP;
* delta: for each step the offset field N_after - N_before measured at 1/8 resolution on the
  sky that did not change (normalised brightness change below R_STEP), filled by normalised
  convolution (SIG_D cells). The c10 form weights the local field by the local share of
  unchanged sky and shrinks the rest towards the step's robust median, with a noise variance;
* periods: each step is oriented towards the daytime anchor it connects to (the first / last
  frame at or above ``e_period``); the cumulated offsets minus their temporal average
  (``t_smooth``) are added to N: steps are removed as discontinuities.

Two ledgers are kept per day: the c10 one and the c5-c9 ("legacy") one, which daytime frames
(sun at or above ``e_day``) read so that daytime is unchanged; 8-10 deg blends the two.
The ledger is computed from the day's frames: it is a cache of this package, not a reference
table. Thresholds (T_STEP, R_STEP, N_NOISE, SIG_D) come from the reference tables
(``c5_ledger_params``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from skycloud.images import pedestal, stamp_minutes, when_of

__all__ = [
    "DayLedger",
    "LedgerSet",
    "SmallFrame",
    "build_day_ledger",
    "offset_from",
    "sky8",
    "small_frame",
    "step_delta",
]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]
U8 = NDArray[np.uint8]


def sky8(zen99: F32) -> B:
    """Sky directions at 1/8 resolution: inside the fisheye, zenith < 85 deg."""
    h, w = zen99.shape
    return np.asarray(cv2.resize(zen99, (w // 8, h // 8), interpolation=cv2.INTER_AREA) < 85)


@dataclass(frozen=True)
class SmallFrame:
    """Pedestal (R corner) and the 1/8-resolution fields of a frame: N, the normalised
    brightness g and the usable sky."""

    ped_r: float
    n: F32
    g: F32
    sky: B


def small_frame(img: U8, sky: B) -> SmallFrame:
    """The 1/8-resolution fields of a frame (``sky``: :func:`sky8`)."""
    a = img.astype(np.float32)
    c = float(pedestal(img)[0])
    h, w = a.shape[:2]
    s: Any = cv2.resize(a, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    r, g, b = s[..., 0], s[..., 1], s[..., 2]
    n = (b - r) / np.maximum(b + r, 1.0)
    ok = sky & (s.max(-1) < 245)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    g = g / max(float(np.median(g[ok])), 1e-3)
    return SmallFrame(c, n.astype(np.float32), g.astype(np.float32), ok)


def _nconv(x: Any, m: B, sig: float) -> tuple[Any, Any]:
    mm = m.astype(np.float32)
    num = cv2.GaussianBlur(np.where(m, x, 0.0).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(mm, (0, 0), sig)
    return num / np.maximum(den, 1e-4), den


def step_delta(
    f0: SmallFrame, f1: SmallFrame, prm: dict[str, Any], c10: bool
) -> tuple[F32, float, F32 | None]:
    """The offset field of a step f0 -> f1 (1/8 resolution), its support (share of the sky
    used) and, in the c10 form, the noise variance of the field."""
    sky = f0.sky & f1.sky
    rel = (f1.g - f0.g) / np.maximum((f1.g + f0.g) / 2.0, 1e-3)
    same = sky & (np.abs(rel) < prm["R_STEP"])
    d, den = _nconv(f1.n - f0.n, same, prm["SIG_D"])
    sup = float(same.sum() / max(sky.sum(), 1))
    if not c10:
        return d, sup, None
    _, sden = _nconv(np.zeros_like(f0.n), sky, prm["SIG_D"])
    frac = np.clip(den / np.maximum(sden, 1e-4), 0.0, 1.0) * sky
    n = int(same.sum())
    m = float(np.median((f1.n - f0.n)[same])) if n >= 20 else 0.0
    se2 = (1.2533 * prm["N_NOISE"]) ** 2 / max(n, 1)
    m_eff = m * m * m / (m * m + se2) if n >= 20 else 0.0
    d = frac * d + (1.0 - frac) * m_eff
    n_eff = 4.0 * np.pi * prm["SIG_D"] ** 2 * np.maximum(frac, 1e-3)
    var = frac**2 * prm["N_NOISE"] ** 2 / n_eff + (1.0 - frac) ** 2 * se2
    return d.astype(np.float32), sup, var.astype(np.float32)


@dataclass(frozen=True)
class DayLedger:
    """The gain steps of one day: for each step (t0 -> t1) its offset field, support and (c10)
    variance; the frames and their pedestal; the morning and evening daytime anchors."""

    t0: NDArray[np.str_]
    t1: NDArray[np.str_]
    delta: F32
    support: NDArray[np.float64]
    var: F32
    frames: NDArray[np.str_]
    ped: NDArray[np.float64]
    anchors: NDArray[np.str_]
    params: str

    def save(self, path: Path) -> None:
        """Write the ledger (compressed npz; the prototype's field names)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t0=self.t0,
            t1=self.t1,
            delta=self.delta,
            support=self.support,
            var=self.var,
            frames=self.frames,
            ped=self.ped,
            anchors=self.anchors,
            params=np.array(self.params),
        )

    @classmethod
    def load(cls, path: Path) -> DayLedger:
        with np.load(path) as d:
            return cls(
                **{k: d[k] for k in cls.__dataclass_fields__ if k != "params"},
                params=str(d["params"]),
            )


def _steps(
    ts: list[str],
    ped: dict[str, float],
    elev: Callable[[str], float],
    prm: dict[str, Any],
    e_period: float,
) -> list[tuple[str, str]]:
    steps = []
    for a, b in pairwise(ts):
        if (when_of(b) - when_of(a)).total_seconds() > 90:
            continue
        if max(elev(a), elev(b)) >= e_period:
            continue
        if abs(np.log(ped[b] / ped[a])) > prm["T_STEP"]:
            steps.append((a, b))
    return steps


def build_day_ledger(
    frames: list[str],
    pedestals: Callable[[list[str]], dict[str, float | None]],
    deltas: Callable[[list[tuple[str, str]]], Iterable[tuple[str, str, Any, float, Any]]],
    elev: Callable[[str], float],
    prm: dict[str, Any],
    *,
    c10: bool,
    e_period: float,
    shape8: tuple[int, int],
) -> DayLedger:
    """The ledger of a day from its frames (time order, sun above -4 deg, exclusions applied).

    ``pedestals`` reads the R-corner pedestal of frames (None: unreadable); ``deltas`` measures
    the steps (t0, t1) -> (t0, t1, delta or None, support, variance); ``elev`` gives the sun
    elevation of a frame."""
    ped = pedestals(frames)
    ts = [t for t in frames if ped.get(t)]
    pv = {t: float(ped[t]) for t in ts}  # type: ignore[arg-type]
    el = np.array([elev(t) for t in ts])
    steps = _steps(ts, pv, elev, prm, e_period)
    res = [r for r in deltas(steps) if r[2] is not None]
    day_idx = np.flatnonzero(el >= e_period)
    morning = ts[day_idx[0]] if day_idx.size else ts[int(np.argmax(el))]
    evening = ts[day_idx[-1]] if day_idx.size else ts[int(np.argmax(el))]
    empty = np.zeros((0, *shape8), np.float32)
    return DayLedger(
        t0=np.array([r[0] for r in res]),
        t1=np.array([r[1] for r in res]),
        delta=np.array([r[2] for r in res], np.float32) if res else empty,
        support=np.array([r[3] for r in res]),
        var=np.array([r[4] for r in res], np.float32) if (res and c10) else empty,
        frames=np.array(ts),
        ped=np.array([pv[t] for t in ts]),
        anchors=np.array([morning, evening]),
        params=json.dumps(prm),
    )


def _step_weights(
    ledger: DayLedger, ts: str, k: NDArray[np.float64], morning: str, evening: str
) -> list[tuple[int, float, float]]:
    out = []
    for j in range(len(ledger.t0)):
        t1 = str(ledger.t1[j])
        if t1 <= morning:  # morning step: frames before it are lifted to the after-state
            side = (ledger.frames < t1).astype(np.float64)
            own, sgn = float(ts < t1), 1.0
        elif t1 > evening:  # evening step: frames after it are brought back to the before-state
            side = (ledger.frames >= t1).astype(np.float64)
            own, sgn = float(ts >= t1), -1.0
        else:
            continue
        w = own - float((k * side).sum())
        if abs(w) > 1e-3:
            out.append((j, sgn, w))
    return out


def offset_from(
    ledger: DayLedger | None, ts: str, sky: B, t_smooth: float
) -> tuple[F32 | None, str]:
    """The frame's offset field at 1/8 resolution to ADD to N (None if the day has no ledger
    or the frame is not in it), and a note: the cumulated step offsets (each step towards its
    period's daytime anchor) minus their temporal average (Gaussian ``t_smooth``)."""
    if ledger is None:
        return None, "no gain ledger for this day: index not corrected"
    if ts not in set(ledger.frames.tolist()):
        return (
            None,
            "frame not in the day's gain ledger (held out or unreadable): index not corrected",
        )
    morning, evening = str(ledger.anchors[0]), str(ledger.anchors[1])
    tf = np.array([stamp_minutes(str(x)) for x in ledger.frames])
    t = stamp_minutes(ts)
    k = np.exp(-0.5 * ((tf - t) / t_smooth) ** 2)
    k[np.abs(tf - t) > 3 * t_smooth] = 0.0
    k /= k.sum()
    off = np.zeros(sky.shape, np.float32)
    v = np.zeros(sky.shape, np.float32)
    has_var = len(ledger.var) == len(ledger.t0) and len(ledger.t0) > 0
    weights = _step_weights(ledger, ts, k, morning, evening)
    for j, sgn, w in weights:
        off += sgn * w * ledger.delta[j]
        if has_var:
            v += w * w * ledger.var[j]
    n = len(weights)
    note = f"gain ledger: {n} step(s) within {3 * t_smooth:g} min removed as discontinuities"
    if has_var and n > 0:
        # c10: many steps close together must not accumulate noise: the spatial structure is
        # kept to the extent it exceeds the expected noise of the sum, its median likewise
        med = float(np.median(off[sky]))
        dev = off - med
        s2 = float(np.mean(dev[sky] ** 2))
        vm = float(np.mean(v[sky]))
        f_sp = float(np.clip(1.0 - vm / max(s2, 1e-12), 0.0, 1.0))
        vmin = float(np.min(v[sky]))
        f_md = med * med / (med * med + vmin) if med != 0 else 0.0
        off = (f_md * med + f_sp * dev).astype(np.float32)
        note += f"; noise check: spatial factor {f_sp:.2f}, median factor {f_md:.2f}"
    return off, note


class LedgerSet:
    """The day ledgers of a render (c10 and legacy), read from a cache folder, and the
    frame offset with the daytime blend."""

    def __init__(self, folder: Path, sky: B, t_smooth: float, e_day: float) -> None:
        self.folder = Path(folder)
        self.sky = sky
        self.t_smooth = t_smooth
        self.e_day = e_day
        self._cache: dict[tuple[str, str], DayLedger | None] = {}

    @staticmethod
    def path(folder: Path, day: str, legacy: bool) -> Path:
        """Where a day's ledger is cached: ``FOLDER/c10/DAY.npz`` or ``FOLDER/legacy/DAY.npz``."""
        return Path(folder) / ("legacy" if legacy else "c10") / f"{day}.npz"

    def day(self, day: str, legacy: bool) -> DayLedger | None:
        key = (day, "legacy" if legacy else "c10")
        if key not in self._cache:
            f = self.path(self.folder, day, legacy)
            self._cache[key] = DayLedger.load(f) if f.exists() else None
        return self._cache[key]

    def offset(self, ts: str, elev: float, legacy: bool = False) -> tuple[F32 | None, str]:
        """The frame's offset (1/8 resolution): the c10 ledger below +8 deg, the legacy one from
        ``e_day`` up (or always with ``legacy``), blended linearly in between."""
        day = ts[:8]
        if legacy:
            return offset_from(self.day(day, True), ts, self.sky, self.t_smooth)
        b = float(np.clip((self.e_day - elev) / 2.0, 0.0, 1.0))
        if b >= 1.0:
            return offset_from(self.day(day, False), ts, self.sky, self.t_smooth)
        o_old, n_old = offset_from(self.day(day, True), ts, self.sky, self.t_smooth)
        if b <= 0.0:
            return o_old, n_old
        o_new, n_new = offset_from(self.day(day, False), ts, self.sky, self.t_smooth)
        if o_new is None or o_old is None:
            return (o_new, n_new) if o_old is None else (o_old, n_old)
        blend = (b * o_new + (1.0 - b) * o_old).astype(np.float32)
        return blend, f"{n_new} (blended {b:.2f} with the c5 ledger, sun 8-10 deg)"
