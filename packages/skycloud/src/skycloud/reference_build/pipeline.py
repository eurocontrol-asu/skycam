"""``build_reference``: an archive of frames + camera geometry + site -> a reference archive in
the format ``skycloud.tables`` loads.

The build runs in a work folder, one step after the other; every step writes its output there
and records a fingerprint of what it was computed from (its parameters, its inputs and the
fingerprints of the steps it reads). A step whose fingerprint is unchanged and whose outputs
exist is not run again: an interrupted build resumes, a changed parameter re-runs the steps
that depend on it, and each step's output can be checked alone.

Steps (prototype scripts in brackets): ``grid`` (a2_features.grid), ``features`` (a2),
``binmeans`` (b1), ``labels`` (b2), ``raw`` (b3), ``smooth`` (b4), ``adapted`` (b6),
``postfit`` (c1_postfit), ``spread`` (c14_spread), ``thresholds`` (c7_shape_stats,
c8_shape_gate, c7_lowsun_levels -> c11, c9_knowable), ``midscale`` (c6_midstats), ``lowcount``
(c3_lowcount_table, c4_lowcount_ped), ``carried`` (tables the build does not measure),
``pack``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from skycloud.cues import PED_BR, PED_GR
from skycloud.geometry import CameraGeometry, Site, angle_to_sun
from skycloud.images import read_rgb, when_of
from skycloud.reference_build.binning import BinMeans, write_bin_means
from skycloud.reference_build.features import FeatureGrid, load_day_features, write_day_features
from skycloud.reference_build.frames import ArchiveFrames
from skycloud.reference_build.labels import CLEAR, label_frames
from skycloud.reference_build.params import BuildParams
from skycloud.reference_build.postfit import postfit_tables, table_grid
from skycloud.reference_build.raw_tables import raw_tables
from skycloud.reference_build.smoothing import adapted_spread, rebuild_spreads, smooth_tables
from skycloud.reference_build.thresholds import (
    frame_stats,
    knowable,
    lift_colour,
    lowcount,
    midscale_frame,
    midscale_static,
    midscale_table,
    release_gate,
    shape_table,
)
from skycloud.sun import sun_position
from skycloud.tables import file_sha256, pack_reference_tables

__all__ = ["STEPS", "BuildInputs", "BuildReport", "CarriedTables", "build_reference"]

#: The tables a build does not measure; they are carried from an existing archive.
#: c4_consistency needs renders of the map itself (c4_consistency.py calibrate);
#: c5_ledger_params was calibrated on two dense days, then SIG_D edited by hand (c5_sigd.py).
CARRIED_JSON = ("c4_consistency", "c5_ledger_params")
LOWCOUNT_FILES = {"lowcount": "c3_lowcount.npz", "lowcount_ped": "c4_lowcount_ped.npz"}
THRESHOLD_JSON = ("c7_shape", "c8_release", "c11_lift_colour", "c9_knowable")
#: The prototype (clouds_v3/scripts) each step reproduces.
PROTOTYPE_SCRIPTS = {
    "grid": "a2_features.grid (kit sky_auto.grid_pixels)", "features": "a2_features.py",
    "binmeans": "b1_binmeans.py", "labels": "b2_labels.py", "raw": "b3_tables.py",
    "smooth": "b4_smooth.py", "adapted": "b6_spread.py", "postfit": "c1_postfit.py",
    "spread": "c14_spread.py",
    "thresholds": "c7_shape_stats.py, c8_shape_gate.py, c7_lowsun_levels.py (c11), c9_knowable.py",
    "midscale": "c6_midstats.py", "lowcount": "c3_lowcount_table.py, c4_lowcount_ped.py",
}  # fmt: skip


@dataclass(frozen=True)
class CarriedTables:
    """Tables copied verbatim from an existing reference archive (checked by its SHA-256)."""

    sha256: str
    json_tables: dict[str, Any]
    arrays: dict[str, dict[str, Any]]

    @classmethod
    def from_archive(cls, path: Path, sha256: str) -> CarriedTables:
        """Read the carried tables of a reference archive whose digest must be ``sha256``."""
        digest = file_sha256(path)
        if digest != sha256:
            raise ValueError(f"carried archive {path}: SHA-256 {digest}, expected {sha256}")
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["__metadata__"]))
            arrays = {
                pre: {k: d[f"{pre}/{k}"] for k in ("S", "e", "q_step", "q_none")}
                for pre in LOWCOUNT_FILES
            }
        return cls(digest, meta["thresholds"], arrays)


@dataclass(frozen=True)
class BuildInputs:
    """What a build reads: the camera (name, geometry, a-priori static sky mask: True = sky),
    the site, the archive frames, optionally dense 30 s days for the low-count tables, and
    the archive the unmeasured tables are carried from."""

    camera: str
    geometry: CameraGeometry
    sky_mask: NDArray[np.bool_]
    site: Site
    frames: ArchiveFrames
    calibration_frames: ArchiveFrames | None = None
    carried: CarriedTables | None = None


@dataclass
class BuildReport:
    """What each step did (run or reused, seconds, notes) and the archive's SHA-256."""

    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    sha256: str = ""


@dataclass
class _Ctx:
    inputs: BuildInputs
    params: BuildParams
    work: Path
    workers: int
    log: Callable[[str], None]
    fingerprints: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ref_dir(self) -> Path:
        return self.work / "ref"

    def grid_dirs(self) -> tuple[Any, Any]:
        with np.load(self.work / "grid.npz") as g:
            return g["az"].astype(np.float64), g["zen"].astype(np.float64)

    def day_files(self, folder: str = "features") -> list[Path]:
        """The finished day files of a feature folder (not the partial ``.tmp.npz``)."""
        return sorted(f for f in (self.work / folder).glob("*.npz") if f.stem.isdigit())

    def days(self, folder: str = "features") -> Iterator[Any]:
        for f in self.day_files(folder):
            yield load_day_features(f)

    def labels(self) -> dict[str, int]:
        with np.load(self.work / "labels.npz") as d:
            return {str(t): int(v) for t, v in zip(d["ts"], d["lab"], strict=True)}


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _array_sha(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _frames_sha(frames: ArchiveFrames | None) -> str:
    return _sha(None if frames is None else [(t, str(p)) for t, p in frames.listed])


# --- steps -------------------------------------------------------------------------------


def _step_grid(c: _Ctx) -> dict[str, Any]:
    grid = FeatureGrid.build(c.inputs.geometry, c.inputs.sky_mask, c.params.bins)
    grid.save(c.work / "grid.npz")
    return {"directions": len(grid.gp)}


_W: dict[str, Any] = {}


def _init_features(grid: FeatureGrid, site: Site, p: Any) -> None:
    _W["grid"], _W["site"], _W["p"] = grid, site, p


def _features_day(args: tuple[Path, list[tuple[str, Path]]]) -> tuple[str, int, list[str]]:
    path, frames = args
    errors: list[str] = []
    n = write_day_features(
        path, frames, _W["grid"], _W["site"], _W["p"], lambda ts, e: errors.append(f"{ts}: {e!r}")
    )
    return path.stem, n, errors


def _feature_days(c: _Ctx, frames: ArchiveFrames, folder: str) -> dict[str, Any]:
    out = c.work / folder
    out.mkdir(parents=True, exist_ok=True)
    grid = FeatureGrid.build(c.inputs.geometry, c.inputs.sky_mask, c.params.bins)
    todo = [
        (out / f"{d}.npz", fs) for d, fs in frames.days.items() if not (out / f"{d}.npz").exists()
    ]
    c.log(f"{folder}: {len(frames.days)} days, {len(todo)} to measure")
    errors: list[str] = []
    args = (grid, c.inputs.site, c.params.bins)
    with contextlib.ExitStack() as stack:
        if c.workers <= 1:
            _init_features(*args)
            results: Any = map(_features_day, todo)
        else:
            pool = stack.enter_context(
                ProcessPoolExecutor(c.workers, initializer=_init_features, initargs=args)
            )
            results = pool.map(_features_day, todo)
        for k, (day, n, err) in enumerate(results):
            errors += err
            c.log(f"{folder}: {day} {n} frames ({k + 1}/{len(todo)})")
    return {"days": len(frames.days), "unreadable": errors}


def _step_features(c: _Ctx) -> dict[str, Any]:
    return _feature_days(c, c.inputs.frames, "features")


def _step_binmeans(c: _Ctx) -> dict[str, Any]:
    az, zen = c.grid_dirs()
    n = 0
    for f in c.day_files():
        with np.load(f) as d:
            n += len(d["ts"])
    written = write_bin_means(c.work / "binmeans", c.days(), n, az, zen, c.params.bins)
    return {"frames": written, **_pedestal_check(c)}


def _pedestal_check(c: _Ctx) -> dict[str, Any]:
    """The pedestal colour of the archive against the renderer's constants (a check only)."""
    cor = np.load(c.work / "binmeans" / "corner.npy")
    dark = cor[:, 0] >= c.params.bins.pedestal_dark_r
    if not dark.any():
        return {"pedestal_check": "no frame with dark corners"}
    g_r = float(np.median(cor[dark, 1] / cor[dark, 0]))
    b_r = float(np.median(cor[dark, 2] / cor[dark, 0]))
    for name, value, const in (("G/R", g_r, PED_GR), ("B/R", b_r, PED_BR)):
        if abs(value - const) > 0.02:
            c.log(f"warning: archive pedestal {name} {value:.3f}, the renderer reads {const}")
    return {"pedestal_g_over_r": round(g_r, 4), "pedestal_b_over_r": round(b_r, 4),
            "dark_corner_frames": int(dark.sum())}  # fmt: skip


def _step_labels(c: _Ctx) -> dict[str, Any]:
    lb = label_frames(BinMeans.open(c.work / "binmeans"), c.params.bins, c.params.labels)
    np.savez(
        c.work / "labels.npz", ts=lb.ts, lab=lb.lab, q10=lb.q10, q50=lb.q50, q90=lb.q90, el=lb.el
    )
    np.savez(c.work / "percentiles.npz", P85=lb.p85, P10=lb.p10)
    for label, name in ((1, "clear"), (2, "overcast")):
        if not (lb.lab == label).any():
            raise ValueError(
                f"no {name} frame in the archive (label-free rule, parameters 'labels'): the "
                "tables need both clear and overcast skies at every sun elevation"
            )
    return {
        "clear": int((lb.lab == 1).sum()),
        "overcast": int((lb.lab == 2).sum()),
        "episodes": {str(v): sum(x == v for x in lb.episodes.values()) for v in (0, 1, 2)},
    }


def _labels_array(c: _Ctx) -> Any:
    with np.load(c.work / "labels.npz") as d:
        return d["lab"]


def _step_raw(c: _Ctx) -> dict[str, Any]:
    bm = BinMeans.open(c.work / "binmeans")
    out = raw_tables(bm, _labels_array(c), c.params.bins, c.params.smooth.table_min_frames)
    np.savez(c.work / "tables_raw.npz", **out)
    return {}


def _step_smooth(c: _Ctx) -> dict[str, Any]:
    with np.load(c.work / "tables_raw.npz") as raw:
        out = smooth_tables(dict(raw), c.params.bins, c.params.smooth)
    np.savez(c.work / "smooth.npz", **out)
    return {}


def _step_adapted(c: _Ctx) -> dict[str, Any]:
    bm = BinMeans.open(c.work / "binmeans")
    with np.load(c.work / "smooth.npz") as s:
        out = adapted_spread(bm, _labels_array(c), dict(s), c.params.bins, c.params.smooth)
    np.savez(c.work / "adapted.npz", **out)
    return {}


def _step_postfit(c: _Ctx) -> dict[str, Any]:
    az, zen = c.grid_dirs()
    with np.load(c.work / "smooth.npz") as s:
        smooth = dict(s)
    out, n = postfit_tables(
        c.days(), c.labels(), az, zen, smooth, c.params.bins, c.params.smooth, c.params.cloudmap
    )
    np.savez(c.work / "postfit.npz", **out)
    return {"clear_frames": n}


def _reference(c: _Ctx) -> dict[str, Any]:
    with np.load(c.work / "smooth.npz") as s, np.load(c.work / "adapted.npz") as a:
        ref = {**dict(s), **dict(a)}
    with np.load(c.work / "postfit.npz") as pf:
        ref.update(dict(pf))
    return ref


def _step_spread(c: _Ctx) -> dict[str, Any]:
    ref = _reference(c)
    c.ref_dir.mkdir(parents=True, exist_ok=True)
    np.savez(c.ref_dir / "c1_ref.npz", **ref)
    np.savez(c.ref_dir / "c14_spread.npz", **rebuild_spreads(ref, c.params.smooth))
    with np.load(c.work / "grid.npz") as g:
        np.savez(c.ref_dir / "grid.npz", gp=g["gp"], az=g["az"], zen=g["zen"])
    return {}


def _write_json(c: _Ctx, name: str, content: dict[str, Any], source: str) -> None:
    c.ref_dir.mkdir(parents=True, exist_ok=True)
    (c.ref_dir / f"{name}.json").write_text(json.dumps({**content, "source": source}))


def _step_thresholds(c: _Ctx) -> dict[str, Any]:
    az, zen = c.grid_dirs()
    ref = table_grid(_reference(c))
    tp = c.params.thresholds
    st = frame_stats(c.days(), c.labels(), az, zen, ref, c.params.bins, tp)
    _write_json(
        c, "c7_shape", shape_table(st, tp), "skycloud.reference_build: brightness-shape ratio"
    )
    _write_json(
        c, "c8_release", release_gate(st), "skycloud.reference_build: joint clear-shape support"
    )
    _write_json(
        c, "c11_lift_colour", lift_colour(st, tp), "skycloud.reference_build: frame colour opacity"
    )
    k = knowable(ref, az, zen, tp, c.params.cloudmap)
    _write_json(c, "c9_knowable", k, "skycloud.reference_build: diffuse knowability of the tables")
    return {"frames_read": len(st.ts)}


def _step_midscale(c: _Ctx) -> dict[str, Any]:
    tp = c.params.thresholds
    geo, site = c.inputs.geometry, c.inputs.site
    labels = c.labels()
    paths = [p for ts, p in c.inputs.frames.listed if labels.get(ts) == CLEAR]
    rng = np.random.default_rng(tp.midscale_seed)
    pick = [
        paths[i] for i in rng.choice(len(paths), min(tp.midscale_frames, len(paths)), replace=False)
    ]
    static = midscale_static(c.inputs.sky_mask, geo.px_deg, tp)
    az = np.nan_to_num(geo.azimuth_deg).astype(np.float32)
    rows = []
    for k, path in enumerate(pick):
        ts = path.stem[6:20]
        saz, szen = sun_position(when_of(ts), site.latitude_deg, site.longitude_deg)
        try:
            img = read_rgb(path)
        except Exception:  # unreadable frame: skipped, as the prototype did  # noqa: S112
            continue
        g = angle_to_sun(az, geo.zen90, saz, szen)
        r = midscale_frame(img, 90 - szen, g, geo.zen90, static, geo.px_deg, tp)
        if r is not None:
            rows.append(r)
        if k % 50 == 0:
            c.log(f"midscale: {k + 1}/{len(pick)}")
    _write_json(
        c, "c6_midscale", midscale_table(rows, tp), "skycloud.reference_build: mid-scale band-pass"
    )
    return {"frames_read": len(rows)}


def _carried(c: _Ctx) -> CarriedTables:
    if c.inputs.carried is None:
        raise ValueError(
            "the tables " + ", ".join(CARRIED_JSON) + " are not measured by the build: give an "
            "archive to carry them from (BuildInputs.carried / --carry)"
        )
    return c.inputs.carried


def _step_lowcount(c: _Ctx) -> dict[str, Any]:
    c.ref_dir.mkdir(parents=True, exist_ok=True)
    if c.inputs.calibration_frames is None:
        carried = _carried(c)
        for pre, fname in LOWCOUNT_FILES.items():
            np.savez(c.ref_dir / fname, **carried.arrays[pre])
        return {"carried_from": carried.sha256}
    info = _feature_days(c, c.inputs.calibration_frames, "calibration_features")
    _, zen = c.grid_dirs()
    tabs = lowcount(c.days("calibration_features"), zen, c.params.bins, c.params.lowcount)
    np.savez(c.ref_dir / "c3_lowcount.npz", **tabs["counts"])
    np.savez(c.ref_dir / "c4_lowcount_ped.npz", **tabs["pedestal"])
    return {"calibration_days": info["days"]}


def _step_carried(c: _Ctx) -> dict[str, Any]:
    carried = _carried(c)
    c.ref_dir.mkdir(parents=True, exist_ok=True)
    for name in CARRIED_JSON:
        (c.ref_dir / f"{name}.json").write_text(json.dumps(carried.json_tables[name]))
    return {"carried_from": carried.sha256}


@dataclass(frozen=True)
class _Step:
    run: Callable[[_Ctx], dict[str, Any]]
    deps: tuple[str, ...]
    keys: Callable[[_Ctx], Any]
    outputs: tuple[str, ...]
    resumable: str = ""  # a folder of day files kept across runs of the same fingerprint


def _inputs_key(c: _Ctx) -> Any:
    i = c.inputs
    return {
        "geometry": _array_sha(i.geometry.azimuth_deg, i.geometry.zenith_deg),
        "sky_mask": _array_sha(i.sky_mask),
        "site": i.site.model_dump(),
    }


STEPS: dict[str, _Step] = {
    "grid": _Step(_step_grid, (), lambda c: [_inputs_key(c), c.params.bins], ("grid.npz",)),
    "features": _Step(
        _step_features, ("grid",), lambda c: [_frames_sha(c.inputs.frames), c.params.bins],
        ("features",), "features",
    ),
    "binmeans": _Step(_step_binmeans, ("features",), lambda c: [c.params.bins], ("binmeans",)),
    "labels": _Step(
        _step_labels, ("binmeans",), lambda c: [c.params.bins, c.params.labels],
        ("labels.npz", "percentiles.npz"),
    ),
    "raw": _Step(_step_raw, ("labels",), lambda c: [c.params.smooth], ("tables_raw.npz",)),
    "smooth": _Step(_step_smooth, ("raw",), lambda c: [c.params.smooth], ("smooth.npz",)),
    "adapted": _Step(_step_adapted, ("smooth",), lambda c: [c.params.smooth], ("adapted.npz",)),
    "postfit": _Step(
        _step_postfit, ("adapted",), lambda c: [c.params.smooth, c.params.cloudmap],
        ("postfit.npz",),
    ),
    "spread": _Step(
        _step_spread, ("postfit",), lambda c: [c.params.smooth],
        ("ref/c1_ref.npz", "ref/c14_spread.npz", "ref/grid.npz"),
    ),
    "thresholds": _Step(
        _step_thresholds, ("spread",), lambda c: [c.params.thresholds, c.params.cloudmap],
        tuple(f"ref/{n}.json" for n in THRESHOLD_JSON),
    ),
    "midscale": _Step(
        _step_midscale, ("labels",), lambda c: [c.params.thresholds, _inputs_key(c)],
        ("ref/c6_midscale.json",),
    ),
    "lowcount": _Step(
        _step_lowcount, ("grid",),
        lambda c: [
            c.params.lowcount, _frames_sha(c.inputs.calibration_frames),
            None if c.inputs.calibration_frames else _carried(c).sha256,
        ],
        tuple(f"ref/{f}" for f in LOWCOUNT_FILES.values()), "calibration_features",
    ),
    "carried": _Step(
        _step_carried, (), lambda c: [_carried(c).sha256],
        tuple(f"ref/{n}.json" for n in CARRIED_JSON),
    ),
}  # fmt: skip


def _fingerprint(c: _Ctx, name: str) -> str:
    st = STEPS[name]
    raw = st.keys(c)
    keys = [k.model_dump(mode="json") if isinstance(k, BaseModel) else k for k in raw]
    return _sha({"step": name, "keys": keys, "deps": [c.fingerprints[d] for d in st.deps]})


def _run_step(c: _Ctx, name: str, report: BuildReport) -> None:
    st = STEPS[name]
    fp = _fingerprint(c, name)
    marker = c.work / "steps" / f"{name}.json"
    done = json.loads(marker.read_text()) if marker.exists() else {}
    if done.get("fingerprint") == fp and all((c.work / o).exists() for o in st.outputs):
        c.fingerprints[name] = fp
        report.steps[name] = {"reused": True, **done.get("info", {})}
        c.log(f"step {name}: up to date")
        return
    if st.resumable:  # day files measured for another fingerprint are stale
        folder = c.work / st.resumable
        folder.mkdir(parents=True, exist_ok=True)
        stamp = folder / "_fingerprint"
        if not stamp.exists() or stamp.read_text() != fp:
            for f in folder.glob("*.npz"):
                f.unlink()
            stamp.write_text(fp)
    c.log(f"step {name}: running")
    t0 = time.monotonic()
    info = st.run(c)
    info["seconds"] = round(time.monotonic() - t0, 1)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"fingerprint": fp, "info": info}, indent=1, default=str))
    c.fingerprints[name] = fp
    report.steps[name] = {"reused": False, **info}


def _metadata(c: _Ctx) -> dict[str, Any]:
    i, p = c.inputs, c.params
    stamps = sorted(str(t) for t in np.load(c.work / "binmeans" / "ts.npy"))  # frames read
    days = sorted({t[:8] for t in stamps})
    carried: dict[str, str] = {}
    if i.carried is not None:
        note = f"carried verbatim from the reference archive {i.carried.sha256}"
        carried = dict.fromkeys(CARRIED_JSON, note)
        if i.calibration_frames is None:
            carried.update(dict.fromkeys(LOWCOUNT_FILES.values(), note))
    try:
        version = metadata.version("skycloud")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "format": "skycloud reference tables",
        "schema_version": 1,
        "camera": i.camera,
        "site": i.site.model_dump(),
        "version": f"skycloud {version} build-reference",
        "archive": {
            "frames": len(stamps), "frames_listed": len(i.frames.listed), "days": len(days),
            "first_day": f"{days[0][:4]}-{days[0][4:6]}-{days[0][6:]}",
            "last_day": f"{days[-1][:4]}-{days[-1][4:6]}-{days[-1][6:]}",
            "calibration_days": sorted(i.calibration_frames.days) if i.calibration_frames else [],
        },
        "grid": {
            "directions": round(360 / p.bins.azimuth_step_deg) * round(90 / p.bins.zenith_step_deg),
            "azimuth_step_deg": p.bins.azimuth_step_deg, "zenith_step_deg": p.bins.zenith_step_deg,
            "window_px": p.bins.window_px,
            "tables": f"{len(p.bins.elevation_nodes)} sun elevations x {p.bins.n_gamma} angles "
            f"to the sun ({p.bins.gamma_bin_deg} deg) x {p.bins.n_zenith} zenith angles "
            f"({p.bins.zenith_bin_deg} deg)",
        },
        "built_by": {
            "steps": {k: {x: v for x, v in info.items() if x not in ("seconds", "reused")}
                      for k, info in c.info.items()},
            "prototype_scripts": PROTOTYPE_SCRIPTS,
            "carried": carried,
        },
        "params": p.model_dump(mode="json"),
    }  # fmt: skip


def build_reference(
    inputs: BuildInputs,
    work: Path,
    out: Path,
    params: BuildParams | None = None,
    *,
    until: str | None = None,
    workers: int = 1,
    log: Callable[[str], None] = lambda s: None,
) -> BuildReport:
    """Run (or resume) the build in ``work`` and pack the reference archive into ``out``.
    ``until``: stop after that step (no archive written). Returns what each step did."""
    if until is not None and until not in STEPS:
        raise ValueError(f"unknown step {until!r}; steps: {', '.join(STEPS)}")
    c = _Ctx(inputs, params or BuildParams(), Path(work), max(workers, 1), log)
    _carried(c)  # checked before hours of measurements, not at the end
    c.work.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    for name in STEPS:
        _run_step(c, name, report)
        c.info[name] = report.steps[name]
        if name == until:
            return report
    report.sha256 = pack_reference_tables(c.ref_dir, Path(out), _metadata(c))
    log(f"reference archive {out}: SHA-256 {report.sha256}")
    return report
