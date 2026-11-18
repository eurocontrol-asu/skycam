"""Calibration of the reflection table on an archive of the camera's frames.

1. ``measure_archive``: for every labelled archive frame (clear or overcast, sun above
   ``min_elevation``), the zone profile and the light of every ring sector, written per day
   (``<day>.npz``) so that an interrupted run resumes where it stopped.
2. ``build_table``: from those measurements, (a) the visibility ramp: 0 when no pixel of the
   zone's core is saturated, 1 from the core's saturated share that 95 % of clear frames of the
   same elevation reach, and the glow ramp used when the core is mostly not sky; (b) the
   reference contrast of the reflections layer; (c) the table: per structure, elevation bin and
   sector direction relative to the sun, the median contrast over clear frames where the sun
   fully shines.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from skysun.frames import FrameFile, list_frames, read_rgb
from skysun.geometry import CameraGeometry
from skysun.params import ZoneParams
from skysun.pipeline import SunCamera
from skysun.static import StaticSource
from skysun.sun import Site, sun_position
from skysun.table import ReflectionTable, StructureInfo, TableMetadata, VisibilityRamp
from skysun.zone import _profile_value, core_state, radial_profile

__all__ = ["BuildConfig", "MeasureConfig", "build_table", "measure_archive", "measure_day"]

F64 = NDArray[np.float64]


class MeasureConfig(BaseModel):
    """Where the archive and its labels are, and which frames to measure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    geometry: Path
    static: Path
    latitude: float
    longitude: float
    labels: Path = Field(
        description="NPZ with 'ts' (YYYYmmddHHMMSS, UTC) and 'lab' (1 clear, "
        "2 overcast, 0 neither) for the archive frames"
    )
    min_elevation: float = Field(3.0, description="Frames with the sun lower are not measured")
    exclude_days: list[date] = Field(default_factory=list, description="Never read these days")


class BuildConfig(BaseModel):
    """Rules of the table build, each with its origin (README)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    camera_id: str
    elevation_edges: list[float] = Field(
        default_factory=lambda: [float(e) for e in range(0, 70, 5)],
        description="Sun-elevation bins (deg): 5-degree bins up to the site's highest sun (65)",
    )
    clear_quantile: float = Field(
        0.05,
        description="Visibility 1 from this quantile of the core's saturated share on clear "
        "frames of the same elevation bin whose core has at least one saturated pixel (the "
        "disc reaches the lens: clear labels at low sun include suns behind haze or trees, "
        "which put the plain 5th percentile at 0.000 for +5-10 deg); per bin, review S4: a low "
        "clear sun saturates less of its core (0.33 at +5-10 against 0.84 at +60-65); 0 when "
        "no core pixel is saturated",
    )
    ramp_min_frames: int = Field(
        30, description="An elevation bin with fewer clear frames takes its nearest bin's top"
    )
    glow_quantiles: tuple[float, float] = Field(
        (0.95, 0.05),
        description="Fallback when the core is mostly not sky: visibility 0 at the first "
        "quantile of the profile's glow over hidden-sun frames (overcast label, no saturated "
        "core pixel, core mostly sky), 1 at the second quantile over clear frames where the sun "
        "fully shines",
    )
    glow_max_elevation: float = Field(
        20.0,
        description="The glow ramp is measured on frames with the sun below this: the core is "
        "mostly not sky only with a low sun (archive: +6.3 to +11.4 deg, p10-p90 of 328 such "
        "frames); over all elevations the bright overcast of a high sun overlaps the clear glow",
    )
    reference_quantile: float = Field(
        0.99,
        description="The reflections layer's reference: this quantile of single-frame sector "
        "contrasts over clear frames where the sun fully shines, rings pooled, the eye left "
        "out (Addendum 2 of criteria.md: the strongest reflections of this optic)",
    )
    negligible_contrast: float = Field(
        0.003,
        description="A predicted contrast below this is negligible: skystatic's level of the "
        "clean optics' own texture (0.29-0.31 %, faint_weak) and of its ring lines (0.31 %, "
        "ring_level)",
    )
    min_frames: int = Field(
        10, description="An elevation bin with fewer clear frames takes its nearest filled bin"
    )
    command: str = ""


_DAY_KEYS = (
    "stamp",
    "label",
    "elevation",
    "azimuth",
    "sun_xy",
    "level",
    "saturated",
    "counts",
    "saturated_counts",
    "sky_level",
    "core_saturated",
    "core_sky",
    "band",
    "flank",
    "usable",
)


@dataclass
class _Worker:
    camera: SunCamera
    static: StaticSource
    labels: dict[str, int]
    min_elevation: float


_WORKER: _Worker | None = None


def _start(config: MeasureConfig) -> None:
    global _WORKER  # one camera per worker process
    geometry = CameraGeometry.from_file(config.geometry)
    static = StaticSource(config.static)
    camera = SunCamera(
        geometry,
        Site(latitude_deg=config.latitude, longitude_deg=config.longitude),
        static.optics(geometry.size),
        None,
    )
    with np.load(config.labels, allow_pickle=False) as stored:
        labels = {str(t): int(v) for t, v in zip(stored["ts"], stored["lab"], strict=True)}
    _WORKER = _Worker(camera, static, labels, config.min_elevation)


def _measure_frame(frame: FrameFile, worker: _Worker) -> dict[str, object] | None:
    label = worker.labels.get(frame.stamp, 0)
    camera = worker.camera
    sun = sun_position(frame.time, camera.site)
    if label not in (1, 2) or sun.elevation_deg < worker.min_elevation or camera.layout is None:
        return None
    try:
        rgb = read_rgb(frame.path)
    except OSError:  # a corrupt archive frame is left out
        return None
    geometry = camera.geometry
    sky = worker.static.sky(frame.time.date(), geometry.size) & geometry.inside
    angle = geometry.angle_to(sun)
    profile = radial_profile(rgb, angle, sky, camera.params.zone)
    core = core_state(rgb, angle, geometry.inside, sky, camera.params.zone)
    luminance = rgb.astype(np.float32).mean(axis=-1)
    usable = sky & (rgb.max(axis=-1) < camera.params.zone.saturation)
    sectors = camera.layout.measure(luminance, usable)
    return {
        "stamp": frame.stamp,
        "label": label,
        "elevation": sun.elevation_deg,
        "azimuth": sun.azimuth_deg,
        "sun_xy": geometry.sun_pixel(angle),
        "level": profile.level,
        "saturated": profile.saturated,
        "counts": profile.counts,
        "saturated_counts": profile.saturated_counts,
        "sky_level": profile.sky_level,
        "core_saturated": core.saturated,
        "core_sky": core.sky,
        "band": sectors.band,
        "flank": sectors.flank,
        "usable": sectors.usable,
    }


def measure_day(folder: Path, output: Path) -> int:
    """Measure one day folder into ``output/<day>.npz`` (worker initialised); frames kept."""
    if _WORKER is None:
        raise RuntimeError("measure_day runs inside a worker started by measure_archive")
    rows = [r for f in list_frames([folder]) if (r := _measure_frame(f, _WORKER)) is not None]
    arrays = {k: np.array([r[k] for r in rows]) for k in _DAY_KEYS} if rows else {}
    layout = _WORKER.camera.structures()
    temporary = output.with_name(f".{output.stem}.{uuid4().hex}.tmp.npz")
    np.savez_compressed(
        temporary,
        structures=np.array(json.dumps([s.model_dump(mode="json") for s in layout])),
        **arrays,  # type: ignore[arg-type]
    )
    temporary.replace(output)
    return len(rows)


def _day_of(folder: Path) -> date:
    parts = folder.parts[-3:]
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def measure_archive(
    days: Sequence[Path], config: MeasureConfig, output: Path, workers: int = 1
) -> dict[str, int]:
    """Measure archive day folders (``.../YYYY/MM/DD``); days already measured are skipped,
    excluded days are refused before anything is read."""
    excluded = set(config.exclude_days)
    for folder in days:
        if _day_of(folder) in excluded:
            raise ValueError(f"{folder}: an excluded day (evaluation) is in the archive list")
    output.mkdir(parents=True, exist_ok=True)
    todo = [(f, output / f"{_day_of(f):%Y%m%d}.npz") for f in days]
    todo = [(f, o) for f, o in todo if not o.exists()]
    done: dict[str, int] = {}
    with ProcessPoolExecutor(max(1, workers), initializer=_start, initargs=(config,)) as pool:
        for (folder, target), count in zip(
            todo, pool.map(measure_day, [f for f, _ in todo], [o for _, o in todo]), strict=True
        ):
            done[target.stem] = count
            del folder
    return done


@dataclass(frozen=True)
class _Measurements:
    structures: list[StructureInfo]
    stamp: NDArray[np.str_]
    label: NDArray[np.int8]
    elevation: F64
    sun_xy: F64
    core: F64  # saturated share of the core's field pixels
    core_sky: F64  # share of the core's field pixels that are sky
    glow: F64  # the profile's glow (mean zone value over ZoneParams.glow_deg)
    contrast: F64  # (frames, structures, sectors)


def _glow(level: F64, share: F64, params: ZoneParams) -> F64:
    """The profile's glow of every frame, as ``ZoneProfile.glow`` computes it."""
    centres = ((np.arange(level.shape[1]) + 0.5) * params.step_deg).astype(np.float64)
    inner = (centres >= params.glow_deg[0]) & (centres < params.glow_deg[1])
    return np.array(
        [
            _profile_value(centres, lv, sh, params)[1][inner].mean()
            for lv, sh in zip(level, share, strict=True)
        ]
    )


def _load(folder: Path, params: ZoneParams) -> _Measurements:
    parts: dict[str, list[NDArray[np.generic]]] = {k: [] for k in _DAY_KEYS}
    structures: list[StructureInfo] | None = None
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            these = [StructureInfo(**s) for s in json.loads(str(stored["structures"].item()))]
            if structures is None:
                structures = these
            elif these != structures:
                raise ValueError(f"{path}: measured on other ring circles than the other days")
            if "stamp" not in stored.files:
                continue
            for key in _DAY_KEYS:
                parts[key].append(stored[key])
    if structures is None or not parts["stamp"]:
        raise ValueError(f"{folder}: no measurement")
    data = {k: np.concatenate(v) for k, v in parts.items()}
    flank = data["flank"]
    return _Measurements(
        structures,
        data["stamp"],
        data["label"].astype(np.int8),
        data["elevation"].astype(np.float64),
        data["sun_xy"].astype(np.float64),
        data["core_saturated"].astype(np.float64),
        data["core_sky"].astype(np.float64),
        _glow(data["level"], data["saturated"], params),
        np.where(data["usable"], data["band"] / np.maximum(flank, 1e-6) - 1.0, np.nan),
    )


def _relative_bins(m: _Measurements, sectors: int) -> NDArray[np.int64]:
    """(frames, structures, sectors): the relative-sector bin of every absolute sector."""
    step = 360.0 / sectors
    centres = (np.arange(sectors) + 0.5) * step
    out = np.empty(m.contrast.shape, dtype=np.int64)
    for i, s in enumerate(m.structures):
        psi = np.degrees(np.arctan2(m.sun_xy[:, 1] - s.centre[1], m.sun_xy[:, 0] - s.centre[0]))
        relative = np.mod(centres[None, :] - psi[:, None], 360.0)
        out[:, i] = np.minimum((relative / step).astype(np.int64), sectors - 1)
    return out


def _medians(
    values: F64, relative: NDArray[np.int64], rows: NDArray[np.bool_], sectors: int
) -> F64:
    """Median per relative sector of one structure over the chosen frames (NaN if none)."""
    picked, rel = values[rows], relative[rows]
    out = np.full(sectors, np.nan)
    for d in range(sectors):
        here = picked[rel == d]
        here = here[np.isfinite(here)]
        if here.size:
            out[d] = float(np.median(here))
    return out


def _filled(table: F64, frames: NDArray[np.int32], min_frames: int) -> NDArray[np.float32]:
    """Elevation bins with too few frames take the nearest filled bin of the same structure."""
    out = table.copy()
    for i in range(table.shape[0]):
        good = np.flatnonzero(frames[i] >= min_frames)
        for e in range(table.shape[1]):
            if frames[i, e] < min_frames:
                out[i, e] = table[i, good[np.argmin(np.abs(good - e))]] if good.size else 0.0
    return np.nan_to_num(out).astype(np.float32)


def _table_arrays(
    m: _Measurements, chosen: NDArray[np.bool_], edges: F64, sectors: int, min_frames: int
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    n, bins = len(m.structures), len(edges) - 1
    contrast = np.full((n, bins, sectors), np.nan)
    frames = np.zeros((n, bins), dtype=np.int32)
    relative = _relative_bins(m, sectors)
    which = np.digitize(m.elevation, edges) - 1
    for e in range(bins):
        rows = chosen & (which == e)
        for i in range(n):
            frames[i, e] = int(np.isfinite(m.contrast[rows, i]).any(axis=1).sum())
            contrast[i, e] = _medians(m.contrast[:, i], relative[:, i], rows, sectors)
    return _filled(contrast, frames, min_frames), frames


def _ramp_tops(
    m: _Measurements, readable: NDArray[np.bool_], edges: F64, config: BuildConfig
) -> list[float]:
    """Per elevation bin, the ``clear_quantile`` of the core's saturated share over clear
    frames whose core is mostly sky; sparse bins take the nearest filled one."""
    which = np.digitize(m.elevation, edges) - 1
    clear = (m.label == 1) & readable & (m.core > 0)
    tops = [
        float(np.quantile(m.core[clear & (which == e)], config.clear_quantile))
        if (clear & (which == e)).sum() >= config.ramp_min_frames
        else np.nan
        for e in range(len(edges) - 1)
    ]
    good = np.flatnonzero(np.isfinite(tops))
    if not good.size:
        raise ValueError("No elevation bin holds enough clear frames for the visibility ramp")
    return [
        tops[e] if np.isfinite(tops[e]) else tops[int(good[np.argmin(np.abs(good - e))])]
        for e in range(len(tops))
    ]


def build_table(
    measurements: Path,
    config: BuildConfig,
    *,
    saturation: int = 250,
    core_deg: float = 2.0,
    step_deg: float = 0.5,
    image_size: tuple[int, int],
    sectors: int = 36,
) -> ReflectionTable:
    """The reflection table from the per-day measurements of ``measure_archive``."""
    params = ZoneParams(core_deg=core_deg, step_deg=step_deg, saturation=saturation)
    m = _load(measurements, params)
    edges = np.asarray(config.elevation_edges, dtype=np.float64)
    readable = m.core_sky >= params.core_sky_min
    tops = _ramp_tops(m, readable, edges, config)
    centres = (edges[:-1] + edges[1:]) / 2
    shining = (m.label == 1) & readable & (m.core >= np.interp(m.elevation, centres, tops))
    hidden = (m.label == 2) & readable & (m.core == 0)
    low_sun = m.elevation < config.glow_max_elevation
    if not (shining & low_sun).any() or not (hidden & low_sun).any():
        raise ValueError("The build needs clear frames with a shining sun and hidden-sun frames")
    glow_low = float(np.quantile(m.glow[hidden & low_sun], config.glow_quantiles[0]))
    glow_high = float(np.quantile(m.glow[shining & low_sun], config.glow_quantiles[1]))
    ramp = VisibilityRamp(
        elevation_edges=[float(e) for e in edges],
        high=tops,
        glow_low=glow_low,
        glow_high=glow_high,
        origin=f"core: saturated share of the field pixels within {core_deg} deg of the sun, 0 "
        f"when none is saturated, 1 from the {config.clear_quantile} quantile per elevation bin "
        f"over {int(((m.label == 1) & readable & (m.core > 0)).sum())} clear frames whose core "
        f"is mostly sky and holds a saturated pixel; glow (core mostly not sky): mean zone "
        f"value {params.glow_deg[0]}-{params.glow_deg[1]} deg, sun below "
        f"{config.glow_max_elevation} deg, 0 at the {config.glow_quantiles[0]} quantile over "
        f"{int((hidden & low_sun).sum())} hidden-sun overcast frames ({glow_low:.3f}), 1 at the "
        f"{config.glow_quantiles[1]} quantile over {int((shining & low_sun).sum())} clear frames "
        f"with a shining sun ({glow_high:.3f})",
    )
    rings = [i for i, st in enumerate(m.structures) if not st.disc]
    pooled = m.contrast[shining][:, rings]
    reference = float(np.quantile(pooled[np.isfinite(pooled)], config.reference_quantile))
    contrast, frames = _table_arrays(m, shining, edges, sectors, config.min_frames)
    stamps = m.stamp[shining]
    metadata = TableMetadata(
        camera_id=config.camera_id,
        image_size_wh=image_size,
        structures=m.structures,
        elevation_edges=[float(e) for e in edges],
        sectors=sectors,
        saturation=saturation,
        core_deg=core_deg,
        visibility=ramp,
        reference_contrast=reference,
        negligible_contrast=config.negligible_contrast,
        frames=int(shining.sum()),
        days=sorted({str(s)[:8] for s in stamps}),
        period=(str(min(stamps)), str(max(stamps))),
        selection=f"labelled clear, core mostly sky and saturated at or above the ramp's top "
        f"({int(shining.sum())} of {int((m.label == 1).sum())} clear frames; "
        f"{int((m.label > 0).sum())} labelled frames measured); reference contrast: the "
        f"{config.reference_quantile} quantile of {int(np.isfinite(pooled).sum())} ring sector "
        f"contrasts on those frames (eye left out)",
        command=config.command,
    )
    return ReflectionTable(metadata, contrast, frames).sealed()
