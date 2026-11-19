"""Render a list of frames: the one entry point of the command line and of Python callers.

``render`` reads, around the requested frames, everything the method needs from their
temporal context, in this order:

1. the +-30 s neighbours of every target (rendered too, into ``_neighbours/``, for the
   temporal passes);
2. the day's gain ledgers (built and cached in ``ledger_dir`` when missing);
3. the frame states (clear-sky level, shape statistics) of the context frames within
   ``context_min`` of a target, one per ``context_step`` s, smoothed in time per target;
4. the per-frame render (value, confidence, layers);
5. the temporal-consistency cap of the confidence;
6. the frame factors (ambiguity, contradiction, overcast rim, deck) smoothed in time;
7. the unknown state.

A frame's result therefore depends on its neighbours in time: the same frame rendered within
another list can differ (different context frames).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from skystatic import load_day_mask, sky_mask
from skystatic.products import read_metadata

from skycloud.consistency import SkyChange, apply_cap
from skycloud.engine import FrameInputs, FrameResult, render_frame
from skycloud.geometry import CameraGeometry, Site
from skycloud.images import frame_path, read_rgb, shift_stamp, stamp_minutes, when_of
from skycloud.ledger import LedgerSet, sky8
from skycloud.ledger_build import ensure_ledgers
from skycloud.level import Grid, context_frames, frame_state, smooth_states
from skycloud.params import CloudMapParams
from skycloud.products import write_frame
from skycloud.sun import sun_area, sun_position
from skycloud.tables import ReferenceTables
from skycloud.unknown import FrameFactors, UnknownContext, apply_unknown

__all__ = ["RenderSetup", "RunReport", "ambiguity", "render"]

B = NDArray[np.bool_]
U8 = NDArray[np.uint8]


@dataclass
class RenderSetup:
    """Everything a render reads besides the frame list: camera geometry and site, reference
    tables, parameters, where the frames, the daily static masks (skystatic products) and the
    gain-ledger cache are, and the frames never to read."""

    geometry: CameraGeometry
    site: Site
    images: Path
    mask_dir: Path
    ledger_dir: Path
    tables: ReferenceTables = field(default_factory=ReferenceTables.load)
    params: CloudMapParams = field(default_factory=CloudMapParams)
    exclude: frozenset[str] = frozenset()
    allow_unreliable_mask: bool = False
    workers: int = 1

    def sun(self, ts: str) -> tuple[float, float]:
        """Sun azimuth and zenith angle (deg) at a frame's timestamp."""
        return sun_position(when_of(ts), self.site.latitude_deg, self.site.longitude_deg)

    def elev(self, ts: str) -> float:
        return 90.0 - self.sun(ts)[1]

    def image(self, ts: str) -> U8:
        return read_rgb(frame_path(self.images, ts))


@dataclass
class RunReport:
    """Per target frame its diagnostics (``frames.json``), and the output folder."""

    out: Path
    frames: list[dict[str, Any]]


class _Worker:
    """Per-process state of a render: setup, masks and ledgers (cached per day)."""

    def __init__(self, setup: RenderSetup) -> None:
        cv2.setNumThreads(1)
        self.s = setup
        self.ledgers = LedgerSet(
            setup.ledger_dir, sky8(setup.geometry.zen99), setup.params.t_smooth, setup.params.e_day
        )
        self._masks: dict[str, B] = {}
        self._spoilt: dict[str, B] = {}
        self._grids: dict[str, Grid] = {}

    def mask(self, ts: str) -> B:
        """The day's static sky (class 0) at the frame size, through skystatic."""
        day = ts[:8]
        if day not in self._masks:
            h, w = self.s.geometry.shape
            when = datetime.strptime(ts, "%Y%m%d%H%M%S")
            self._masks[day] = sky_mask(
                self.s.mask_dir, when, (w, h), allow_unreliable=self.s.allow_unreliable_mask
            )
        return self._masks[day]

    def spoilt(self, ts: str) -> B:
        """The day's variable static (skystatic class 2) at the frame size: sky behind it, the
        measurement spoilt (read from the same daily product as ``mask``)."""
        day = ts[:8]
        if day not in self._spoilt:
            h, w = self.s.geometry.shape
            when = datetime.strptime(ts, "%Y%m%d%H%M%S").date()
            paths = [
                q
                for q in sorted(self.s.mask_dir.glob("classes_*.png"))
                if read_metadata(q).day == when
            ]
            classes = load_day_mask(paths[0], (w, h)).classes  # sky_mask checked it is unique
            self._spoilt[day] = np.asarray(classes == 2)
        return self._spoilt[day]

    def grid(self, ts: str) -> Grid:
        day = ts[:8]
        if day not in self._grids:
            self._grids[day] = Grid.build(self.s.tables, self.s.geometry.shape, self.mask(ts))
        return self._grids[day]

    def offset_full(self, ts: str, elev: float, legacy: bool = False) -> tuple[Any, str]:
        off8, note = self.ledgers.offset(ts, elev, legacy)
        if off8 is None:
            return None, note
        h, w = self.s.geometry.shape
        return cv2.resize(off8, (w, h), interpolation=cv2.INTER_LINEAR), note

    def state(self, ts: str) -> tuple[str, dict[str, Any]]:
        """The frame's own state (c10: also with the legacy ledger below +12 deg)."""
        try:
            img = self.s.image(ts)
            saz, szen = self.s.sun(ts)
            elev = 90.0 - szen
            g = self.grid(ts)
            st = frame_state(
                img, saz, szen, g, self.s.tables, self.s.params, self.offset_full(ts, elev)[0]
            )
            if st.get("elev", 99) < 12.0:
                lg = frame_state(
                    img,
                    saz,
                    szen,
                    g,
                    self.s.tables,
                    self.s.params,
                    self.offset_full(ts, elev, True)[0],
                )
                st["legacy"] = {
                    k: lg[k]
                    for k in ("aN", "fN", "aL", "fL", "beta", "n", "pedR", "elev")
                    if k in lg
                }
            return ts, st
        except Exception as e:  # an unreadable frame has no state
            return ts, {"error": repr(e)[:200]}

    def neighbour_loader(self, ts: str, s: int) -> Callable[[], U8 | None]:
        n = shift_stamp(ts, s)

        def load() -> U8 | None:
            p = frame_path(self.s.images, n)
            if n in self.s.exclude or not p.exists():
                return None
            try:
                return read_rgb(p)
            except Exception:
                return None

        return load

    def render(self, item: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
        ts, level, out = item
        try:
            img = self.s.image(ts)
            saz, szen = self.s.sun(ts)
            off, note = self.offset_full(ts, 90.0 - szen)
            site = self.s.site
            area = (
                sun_area(img, when_of(ts), site.latitude_deg, site.longitude_deg, self.s.geometry)
                if self.s.params.unseen_sky == "fill"
                else None
            )
            fill = self.s.params.unseen_sky == "fill"
            x = FrameInputs(
                saz, szen, self.mask(ts), level, off, note,
                {s: self.neighbour_loader(ts, s) for s in (-30, 30)}, area,
                self.spoilt(ts) if fill else None,
            )  # fmt: skip
            res: FrameResult = render_frame(
                img, ts, x, self.s.geometry, self.s.tables, self.s.params
            )
            res.info["static_mask"] = f"static mask of {ts[:4]}-{ts[4:6]}-{ts[6:8]} (skystatic)"
            write_frame(Path(out), ts, res)
            res.info.update({k: round(float(v), 5) for k, v in level.items() if k != "beta"})
            return res.info
        except Exception as e:  # one frame's failure is reported, not fatal
            import traceback

            return {"ts": ts, "error": repr(e)[:300], "tb": traceback.format_exc()[-800:]}


_W: dict[str, _Worker] = {}


def _init(setup: RenderSetup) -> None:
    _W["w"] = _Worker(setup)


def _state(ts: str) -> tuple[str, dict[str, Any]]:
    return _W["w"].state(ts)


def _render(item: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
    return _W["w"].render(item)


def _map(setup: RenderSetup, fn: Callable[[Any], Any], items: list[Any]) -> Iterator[Any]:
    if setup.workers <= 1:
        _init(setup)
        yield from (fn(i) for i in items)
        return
    with ProcessPoolExecutor(min(setup.workers, 16), initializer=_init, initargs=(setup,)) as pool:
        yield from pool.map(fn, items, chunksize=2)


def _lift_colour(t: dict[str, Any], elev: float) -> tuple[float, float]:
    e = max(elev, 3.5)
    m0, m1 = float(np.interp(e, t["e"], t["clear_p95"])), float(np.interp(e, t["e"], t["ov_p5"]))
    return m0, max(m1, m0 + 0.05)


def _contradiction(
    rows: list[dict[str, Any]], lift_t: dict[str, Any], p: CloudMapParams
) -> dict[str, dict[str, Any]]:
    """c11/c12/c15: contradiction C = w_ov (1 - w_col), overcast weight W = w_ov w_col and
    gap-free deck weight Wg = W (1 - b_gap), smoothed in time (Gaussian t_shape)."""
    out: dict[str, dict[str, Any]] = {}
    rc = [r for r in rows if "error" not in r and r.get("elev", 99) < 10.0]
    if not rc:
        return out
    tc = np.array([stamp_minutes(r["ts"]) for r in rc])

    def lift(r: dict[str, Any], key: str) -> float:
        lf = r.get("lift")
        return float(lf[key]) if isinstance(lf, dict) else 0.0

    def b_gap(r: dict[str, Any]) -> float:
        q5 = (r.get("overcast_level") or {}).get("q5_s")
        if q5 is None or not np.isfinite(q5):
            return 1.0
        m0, m1 = _lift_colour(lift_t, r["elev"])
        return float(np.clip((m1 - q5) / (m1 - m0), 0.0, 1.0))

    cr = np.array(
        [
            lift(r, "w_ov") * (1.0 - lift(r, "w_col")) if isinstance(r.get("lift"), dict) else 0.0
            for r in rc
        ]
    )
    wr = np.array(
        [lift(r, "w_ov") * lift(r, "w_col") if isinstance(r.get("lift"), dict) else 0.0 for r in rc]
    )
    wgr = wr * np.array([1.0 - b_gap(r) for r in rc])
    for r, ti in zip(rc, tc, strict=True):
        k = np.exp(-0.5 * ((tc - ti) / p.t_shape) ** 2)
        c_ = float((k * cr).sum() / k.sum())
        w_ = float((k * wr).sum() / k.sum())
        wg_ = float((k * wgr).sum() / k.sum())
        if c_ > 0 or w_ > 0:
            out[r["ts"]] = {"A": 0.0, "C": round(c_, 4), "W": round(w_, 4), "Wg": round(wg_, 4)}
    return out


def ambiguity(
    rows: list[dict[str, Any]], tables: ReferenceTables, p: CloudMapParams
) -> dict[str, dict[str, Any]]:
    """c10/c11: the frame factors of every rendered frame below +10 deg, smoothed in time:
    A = shape-ambiguous (joint clear-shape support between the archive overcast p98 and the
    largest overcast support T_o, back to 0 at the clear median T_c) x structure-less
    (certified share below amb_s) x clip((10 - sun) / 2); plus C, W, Wg (_contradiction)."""
    rel = tables.table("c8_release")
    t_o, t_c, j0 = rel["T_o"], rel["T_c"], p.amb_j0
    s0, s1 = p.amb_s
    out = _contradiction(rows, tables.table("c11_lift_colour"), p)
    rr = [
        r
        for r in rows
        if "error" not in r and "joint_s" in r and "struct_share" in r and r.get("elev", 99) < 10.0
    ]
    if not rr:
        return out
    t = np.array([stamp_minutes(r["ts"]) for r in rr])
    sh = np.array([r["struct_share"] for r in rr])
    for r, ti in zip(rr, t, strict=True):
        k = np.exp(-0.5 * ((t - ti) / p.t_amb) ** 2)
        s = float((k * sh).sum() / k.sum())
        j = r["joint_s"]
        if j < t_o:
            a_sh = float(np.clip((j - j0) / (t_o - j0), 0.0, 1.0))
        else:
            a_sh = float(np.clip((t_c - j) / (t_c - t_o), 0.0, 1.0))
        a_st = float(np.clip((s1 - s) / (s1 - s0), 0.0, 1.0))
        a = a_sh * a_st * float(np.clip((10.0 - r["elev"]) / 2.0, 0.0, 1.0))
        out.setdefault(r["ts"], {}).update(
            {
                "A": round(a, 4),
                "a_shape": round(a_sh, 4),
                "a_struct": round(a_st, 4),
                "struct_s": round(s, 5),
            }
        )
    return out


def _neighbours(targets: list[str], setup: RenderSetup) -> list[str]:
    want = sorted(
        {shift_stamp(t, s) for t in targets for s in (-30, 30)} - set(targets) - set(setup.exclude)
    )
    return [t for t in want if frame_path(setup.images, t).exists()]


def _states(frames: list[str], setup: RenderSetup) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in _map(setup, _state, frames) if "error" not in v}


def render(targets: Iterable[str], setup: RenderSetup, out: Path) -> RunReport:
    """Render ``targets`` (timestamps ``YYYYmmddHHMMSS``) into the folder ``out``.

    Writes the products of every target (and of its +-30 s neighbours into
    ``out/_neighbours``), ``states.json`` and ``frames.json``; returns the per-frame
    diagnostics. Excluded frames are never read, as targets, context or neighbours."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    main = sorted(set(targets) - set(setup.exclude))
    nbrs = _neighbours(main, setup)
    if nbrs:
        (out / "_neighbours").mkdir(exist_ok=True)
    every = sorted(set(main) | set(nbrs))
    p = setup.params
    ensure_ledgers(sorted({t[:8] for t in every}), setup)
    frames = context_frames(every, setup.images, p.context_min, p.context_step, setup.exclude)
    states = _states(frames, setup)
    levels = smooth_states(states, every, p)
    (out / "states.json").write_text(json.dumps(states))
    nset = set(nbrs)
    items = [(t, levels[t], str(out / "_neighbours") if t in nset else str(out)) for t in every]
    all_rows = list(_map(setup, _render, items))
    rows = sorted((r for r in all_rows if r["ts"] not in nset), key=lambda r: r["ts"])
    _init(setup)
    w = _W["w"]
    prm = setup.tables.table("c5_ledger_params")
    change = SkyChange(
        setup.images, w.ledgers.sky, w.ledgers, setup.elev, prm["R_STEP"], prm["N_NOISE"]
    )
    d0 = setup.tables.table("c4_consistency")["D0"]
    cinfo = apply_cap(out, main, change, d0, p.cap, p.cap_sig)
    for r in rows:
        r["consistency"] = cinfo.get(r["ts"], {"note": "no value: not checked"})
    amb = ambiguity(sorted(all_rows, key=lambda r: r["ts"]), setup.tables, p)
    for r in rows:
        if r["ts"] in amb:
            r["ambiguity"] = amb[r["ts"]]
    factors = {
        t: FrameFactors(a.get("A", 0.0), a.get("C", 0.0), a.get("W", 0.0), a.get("Wg", 0.0))
        for t, a in amb.items()
    }
    ctx = UnknownContext(p, change.same, setup.elev, setup.geometry.zen99)
    shares = apply_unknown(out, main, factors, ctx)
    for r in rows:
        r["unknown_share"] = shares.get(r["ts"])
    (out / "frames.json").write_text(json.dumps(rows, indent=0))
    return RunReport(out, rows)
