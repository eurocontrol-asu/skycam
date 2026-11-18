"""End-to-end: frames of several days to installation and variable-static masks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import binary_dilation, find_objects, label

from skystatic.aggregate import Aggregates, aggregate_parts, daytime_frames, frame_size
from skystatic.contrast import (
    FaintEvidence,
    FaintInputs,
    MarkEvidence,
    SmoothSky,
    faint_inputs,
    faint_marks,
    fixed_marks,
    local_contrast,
)
from skystatic.frames import FrameSource, source_date, source_fingerprint
from skystatic.geometry import SkyGeometry
from skystatic.installation import FrozenInstallation, installation_digest
from skystatic.masks import (
    CLASS_NAMES,
    class_map,
    enclosed_sky,
    hysteresis,
    installation_mask,
    variable_mask,
)
from skystatic.optics import RingFit, concentric_rings, ring_mask, without_rings
from skystatic.products import MaskMetadata, png_metadata
from skystatic.render import overlay
from skystatic.segment import static_mask_all_parts
from skystatic.sun import sun_clearance

__all__ = [
    "DayResult",
    "RunReport",
    "StaticConfig",
    "compute_day",
    "day_classes",
    "run",
    "skyline_glow",
]


class StaticConfig(BaseModel):
    """Parameters of the method. Lengths in pixels are given at a width of 1232.

    Every default states its origin: a property of the data (measured on the GVCCS site camera,
    review of 2026-11-17), the day's own noise, or a design choice. See the README table.
    """

    model_config = ConfigDict(frozen=True)

    aggregate: Literal["mean", "geometric"] = "geometric"
    parts: int = Field(
        3,
        ge=1,
        description="Consecutive parts of a day that must all agree (design choice: the sun "
        "moves about 45 degrees of azimuth per third of a day, so its track and glare do not "
        "overlap between parts; more parts make each average noisier)",
    )
    max_side: int = Field(1232, ge=0, description="Working resolution; 0 keeps the native size")
    daytime_contrast: float = Field(
        2.0,
        ge=1,
        description="A daytime frame shows its fisheye circle: sky around the zenith at least this "
        "many times brighter than the frame beyond the horizon, in at least one colour channel. "
        "A flat night frame has 1; daytime frames of the site camera have 2.2-4 (5th-50th "
        "percentile); between them only the twilight ramp",
    )
    daytime_relative: float = Field(
        0.25,
        gt=0,
        le=1,
        description="Only for frames without pixels beyond the horizon: a daytime frame reaches "
        "this fraction of the 95th percentile of the central luminance (design choice: twilight "
        "is several times dimmer than the brightest daylight)",
    )
    daytime_floor: float = Field(
        0.0, ge=0, le=1, description="Absolute minimum central luminance (0..1); 0 = no floor"
    )
    zenith_limit: float = Field(
        90.0, gt=0, description="Analysed domain, degrees (the horizon: beyond it no sky)"
    )
    core_zenith: float = Field(
        60.0,
        gt=0,
        description="Where the sky certainly is, degrees (site camera: nothing fixed stands "
        "above 30 degrees of elevation but the tip of one antenna, at zenith 53-58). Faint "
        "marks are looked for only there: open sky, away from the slow haze and cloud banks "
        "of the horizon",
    )
    edge: float = Field(
        0.05,
        gt=0,
        description="Edge strength above which a pixel is an edge (measured on the site camera: "
        "99th percentile of the sky around the zenith 0.01-0.02 on overcast parts, median at the "
        "installation border 0.06-0.11)",
    )
    background: int = Field(
        36,
        ge=0,
        description="Window of the local gradient level removed from edges, pixels (design "
        "choice: several times the width of an edge after the 1-pixel blur, small against the "
        "scale of the horizon brightening); never below 15 pixels at any resolution",
    )
    mark_sigma: float = Field(
        12.0,
        gt=0,
        description="Window (Gaussian sigma, pixels) of the smooth sky against which marks are "
        "measured: larger than the largest mark seen on the optics (15 px), small enough for a "
        "quadratic surface to follow the horizon brightening",
    )
    mark_k: float = Field(
        3.0,
        ge=0,
        description="A mark exceeds this many times the window's sky noise (robust standard "
        "deviation of the contrast at its zenith angle), either sign, in more than half of the "
        "day's windows: for Gaussian noise one window passes with probability 0.003, a majority "
        "of 6 or more windows less than 1e-12 per pixel; 0 disables marks",
    )
    mark_opposite: float = Field(
        0.2,
        ge=0,
        le=1,
        description="A mark changes sign with the lighting (see skyline_glow) when it is "
        "significant in hours of both signs, the other sign at no less than this fraction of its "
        "strongest hour's amplitude. Measured on the 10 reference days of the site camera: the "
        "marks that change sign during the day (dirt lit from the front, then from behind) keep "
        "0.22 of their strongest amplitude in the other sign at the 5th percentile (median 0.49, "
        "42 marks)",
    )
    faint_scales: tuple[float, float] = Field(
        (1.2, 4.0),
        description="Faint marks: the band-pass (sigmas of two Gaussian means, pixels) keeps the "
        "scale of the smallest drop and dust discs (5-7 px across, measured) and of a 1-3 px "
        "trail, and removes the smooth sky and slow cloud residue",
    )
    faint_weak: float = Field(
        0.003,
        ge=0,
        description="Faint marks, weak level (log units, 0.003 = 0.3 %): above the largest "
        "fixed texture of the clean optics in the band-passed day mean (0.29-0.31 % within 60 "
        "degrees on the quiet reference days 2024-09-05, 09-25, 10-15); a weak pixel is kept "
        "only when connected to a mark (hysteresis)",
    )
    faint_weak_k: float = Field(
        2.0,
        ge=0,
        description="Faint marks, weak level in units of the day's TEMPORAL noise (spread of the "
        "differences between parts of the day, where everything fixed cancels); also the level "
        "every part must reach for a stand-alone faint mark",
    )
    faint_alone: float = Field(
        0.005,
        ge=0,
        description="Faint marks without a seed: band-passed day contrast of at least this "
        "(log units, 0.5 %, above the clean optics' texture of 0.3 % with margin) and "
        "faint_alone_k temporal noise units; 0 disables faint marks",
    )
    faint_alone_k: float = Field(
        4.0,
        ge=0,
        description="Faint marks without a seed, in units of the day's temporal noise",
    )
    window_minutes: int = Field(
        60,
        ge=1,
        description="Length of the short windows in which marks are judged (design choice: the "
        "sun moves about 15 degrees in an hour, so neither it nor a cloud comes back to the same "
        "pixel in most separate hours, while a day still holds 6-15 windows)",
    )
    mark_band: float = Field(
        5.0,
        gt=0,
        description="Zenith bands (degrees) in which the sky noise is measured (design choice: "
        "the noise grows towards the horizon; a band holds thousands of pixels)",
    )
    ring_level: float = Field(
        0.0031,
        ge=0,
        description="Permanent rings of the optics, level mode: every line down to this contrast "
        "(log units, 0.0031 = 0.31 %, about 4 % of the sky on the site camera; 0.0023 about "
        "12 %); 0 disables the rings in both modes",
    )
    ring_halfwidth: float = Field(
        1.5,
        ge=0,
        description="Permanent rings, lines mode (default): only the bright lines, masked over "
        "+- this many pixels of their radius (the measured width of a line, ~3 px); 0 selects "
        "the level mode",
    )
    sun_clearance: float = Field(
        20.0,
        ge=0,
        description="With an azimuth map and a site: a window in which the sun comes within this "
        "angle (degrees) of a pixel does not judge it. Measured on the 10 reference days of the "
        "site camera: within 20 degrees of the sun the share of significant sky pixels in an "
        "hourly mean is an order of magnitude above its far-field level (26x at 0-5, 9x at "
        "15-20 degrees); farther, the excess decays slowly (7x at 20-25, 2x at 40-45) and "
        "removing those hours too would leave marks near the sun's path undecided; 0 disables",
    )
    reference_window: int = Field(
        365,
        ge=0,
        description="The median date of the reference days must lie within this many days of "
        "every processed day (design choice: one seasonal cycle; dirt episodes of months must "
        "not be voted into the installation of another period); 0 disables the check",
    )
    min_days: float = Field(
        0.6,
        gt=0,
        le=1,
        description="Fraction of reference days on which a pixel is static to be installation "
        "(design choice: a clear majority, so that one dirt episode among ten or more reference "
        "days spread over a year or more cannot reach it)",
    )
    tolerance: int = Field(
        2,
        ge=0,
        description="Slack around the installation, pixels (design choice: an edge moves by a "
        "pixel or two between days with light and foliage)",
    )
    outlier_ratio: float = Field(
        0.5,
        ge=0,
        lt=1,
        description="A reference day is dropped when its static fraction is below this ratio "
        "of the median one (fog: no visible skyline), or above its inverse (static fractions "
        "of usable days span 11.5-14 % on the site camera; a lost skyline halves it)",
    )
    workers: int = Field(16, ge=1, description="Decoding threads per day")
    jobs: int = Field(1, ge=1, description="Days processed at the same time")

    def aggregation_key(self, geometry: SkyGeometry | None = None) -> str:
        """Identifies cached averages: only what changes them.

        The selection of daytime frames depends on the zenith map, which is part of the key.
        """
        fields = (
            "daytime-structure-v4-channels-windows",
            self.window_minutes,
            self.parts,
            self.max_side,
            self.daytime_contrast,
            self.core_zenith,
            self.daytime_relative,
            self.daytime_floor,
            None if geometry is None else _geometry_fingerprint(geometry),
        )
        return hashlib.sha256(repr(fields).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class DayResult:
    """Static mask of one day at the working resolution."""

    label: str
    mask: NDArray[np.bool_]
    domain: NDArray[np.bool_]
    image: NDArray[np.float32]
    frames_daytime: int
    frames_total: int
    contrast: NDArray[np.float32] | None = None  # local contrast of the whole-day mean (rings)
    evidence: MarkEvidence | None = None  # marks of the day and its weak (faint fixed) pixels
    faint: FaintInputs | None = None  # contrasts of the day and its parts, for faint marks

    @property
    def static_fraction(self) -> float:
        return float(self.mask.sum() / max(int(self.domain.sum()), 1))


class DayReport(BaseModel):
    label: str
    date: str
    variable_reliable: bool
    frames_daytime: int
    frames_total: int
    static_fraction: float
    variable_fraction: float
    variable_components: int
    skyline_visible: bool
    # Rings specific to the day: none since round c9 (the rings are circles fitted once on the
    # reference days, in the installation); kept, always empty, for the report's format.
    ring_lines: list[float] = []


class RunReport(BaseModel):
    config: StaticConfig
    geometry_sha256: str
    installation_id: str | None = None
    size_wh: tuple[int, int]
    reference_days: list[str]
    rejected_reference_days: list[str]
    installation_fraction: float
    classes: dict[int, str]
    days: list[DayReport]
    optics: RingFit | None = None


def _load_or_aggregate(
    source: FrameSource, config: StaticConfig, cache: Path | None, geometry: SkyGeometry
) -> tuple[Aggregates, int, int]:
    key = config.aggregation_key(geometry)
    name = f"{source.label}_{key}_{source_fingerprint(source)}.npz"
    path = None if cache is None else cache / name
    if path is not None and path.exists():
        with np.load(path) as stored:
            aggregates = Aggregates(
                stored["mean"],
                stored["geometric"],
                tuple(int(c) for c in stored["counts"]),
                stored["windows"],
                tuple(int(c) for c in stored["window_counts"]),
                tuple(datetime.fromisoformat(str(t)) for t in stored["window_starts"]),
            )
            return aggregates, int(stored["frames_daytime"]), int(stored["frames_total"])
    frames, total = daytime_frames(
        source,
        config.daytime_relative,
        config.daytime_floor,
        config.workers,
        geometry=geometry,
        core_zenith=config.core_zenith,
        min_contrast=config.daytime_contrast,
    )
    if len(frames) < config.parts:
        raise ValueError(f"{source.label}: {len(frames)} daytime frames out of {total}")
    size = frame_size(source, frames, config.max_side)
    aggregates = aggregate_parts(
        source, frames, size, config.parts, config.workers, config.window_minutes
    )
    if path is not None:
        cache.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.npz")
        np.savez(
            temporary,
            mean=aggregates.mean,
            geometric=aggregates.geometric,
            counts=np.array(aggregates.counts),
            windows=aggregates.windows,
            window_counts=np.array(aggregates.window_counts),
            window_starts=np.array([t.isoformat() for t in aggregates.window_starts]),
            frames_daytime=len(frames),
            frames_total=total,
        )
        temporary.replace(path)
    return aggregates, len(frames), total


def compute_day(
    source: FrameSource,
    geometry: SkyGeometry,
    config: StaticConfig,
    cache: Path | None = None,
) -> DayResult:
    """Static mask of one day: edges agreed by all parts, marks agreed by most windows."""
    source_date(source)
    aggregates, daytime, total = _load_or_aggregate(source, config, cache, geometry)
    parts = aggregates.geometric if config.aggregate == "geometric" else aggregates.mean
    height, width = parts.shape[1:3]
    local = geometry.resized((width, height))
    domain = local.below(config.zenith_limit)
    core = local.below(config.core_zenith)
    if not core.any():
        raise ValueError("No pixel inside the core zenith angle; check the zenith map")
    # Scaled with the image, but never so small that it follows the edges themselves.
    window = max(15, round(config.background * width / 1232)) if config.background else 0
    mask = static_mask_all_parts(parts, domain, core, config.edge, window)
    evidence = None
    faint = None
    usable = _usable_windows(local, aggregates, config)
    if config.mark_k > 0:
        evidence = fixed_marks(
            aggregates.windows,
            domain & ~mask,
            local.zenith_deg,
            config.mark_sigma * width / 1232,
            config.mark_k,
            config.mark_band,
            usable,
            opposite=config.mark_opposite,
        )
        mask = mask | evidence.marks
        if config.faint_alone > 0:
            faint = faint_inputs(
                aggregates.windows,
                aggregates.window_counts,
                domain & ~mask,
                evidence.marks,
                config.mark_sigma * width / 1232,
                config.parts,
                usable,
            )
            if faint is not None:
                weak = _faint(faint, local.zenith_deg, config).weak
                evidence = replace(evidence, weak=weak)
    weights = np.array(aggregates.counts, dtype=np.float32)[:, None, None, None]
    image = (parts * weights).sum(axis=0) / weights.sum()
    contrast = None
    if config.ring_level > 0:
        day_mean = (aggregates.mean * weights).sum(axis=0) / weights.sum()
        surface = SmoothSky(domain & ~mask, config.mark_sigma * width / 1232)
        contrast = local_contrast(day_mean.astype(np.float32), surface)
    return DayResult(
        source.label,
        mask,
        domain,
        image.astype(np.float32),
        daytime,
        total,
        contrast,
        evidence,
        faint,
    )


def _faint(
    inputs: FaintInputs,
    zenith: NDArray[np.float32],
    config: StaticConfig,
    exclude: NDArray[np.bool_] | None = None,
) -> FaintEvidence:
    """``faint_marks`` with the parameters of ``config`` (lengths scaled to the image)."""
    scale = inputs.day.shape[1] / 1232
    return faint_marks(
        inputs,
        zenith,
        scales=(config.faint_scales[0] * scale, config.faint_scales[1] * scale),
        weak_level=config.faint_weak,
        weak_k=config.faint_weak_k,
        alone_level=config.faint_alone,
        alone_k=config.faint_alone_k,
        central_zenith=config.core_zenith,
        band=config.mark_band,
        exclude=exclude,
    )


def _usable_windows(
    geometry: SkyGeometry, aggregates: Aggregates, config: StaticConfig
) -> NDArray[np.bool_] | None:
    """Windows in which each pixel is far enough from the sun to vote for a mark."""
    if config.sun_clearance <= 0 or geometry.azimuth_deg is None or geometry.site is None:
        return None
    length = timedelta(minutes=config.window_minutes)
    spans = [(start, start + length) for start in aggregates.window_starts]
    return np.asarray(sun_clearance(geometry, spans) >= config.sun_clearance)


def _ring_band(rings: NDArray[np.bool_], tolerance: int) -> NDArray[np.bool_]:
    """The ring lines with the installation's slack: what lies along them is the rings' flank."""
    return binary_dilation(rings, iterations=tolerance) if tolerance and rings.any() else rings


def _grown_installation(
    fixed: NDArray[np.bool_],
    voters: list[DayResult],
    config: StaticConfig,
    exclude: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """The installation grown by hysteresis through what is faint but fixed on the references.

    The thin top of a mast is too faint to be static on its own on most days, but it is fixed
    (static, or weak in the sense of ``faint_marks``) at the same pixels on most reference days
    and attached to the mast's base: it is installation, every day. Pixels static or weak on at
    least ``min_days`` of the days, connected to the installation, join it; never through the
    ring lines' band (``exclude``), whose depth the ring parameters set.
    """
    if not any(r.evidence is not None and r.evidence.weak is not None for r in voters):
        return fixed
    fixed_days = [
        r.mask if r.evidence is None or r.evidence.weak is None else r.mask | r.evidence.weak
        for r in voters
    ]
    days_needed = max(1, int(np.ceil(config.min_days * len(fixed_days))))
    steady = np.sum(fixed_days, axis=0) >= days_needed
    return hysteresis(fixed, steady & voters[0].domain & ~exclude)


def _check_reference_period(
    days: Sequence[FrameSource], reference: Sequence[FrameSource], window: int
) -> None:
    """Refuse an installation voted on references that describe another period than the days.

    Dirt can last for months: a spot present on most references of a later year becomes
    installation and masks open sky on earlier days. The median reference date must lie within
    ``window`` days of every processed day.
    """
    if not reference or window <= 0:
        return
    dates = sorted(date.fromisoformat(source_date(s)) for s in reference)
    middle = dates[len(dates) // 2]
    for source in days:
        day = date.fromisoformat(source_date(source))
        if abs((day - middle).days) > window:
            raise ValueError(
                f"{day}: the reference days centre on {middle}, more than {window} days away; "
                "choose references from the period of the processed days (reference_window)"
            )


def _permanent_rings(
    voters: list[DayResult],
    fixed: NDArray[np.bool_],
    geometry: SkyGeometry,
    config: StaticConfig,
) -> tuple[NDArray[np.bool_], RingFit | None]:
    """Rings of the optics, fitted on the mean contrast of the reference days."""
    maps = [r.contrast for r in voters if r.contrast is not None]
    if config.ring_level <= 0 or len(maps) != len(voters):
        return np.zeros(fixed.shape, dtype=bool), None
    height, width = fixed.shape
    local = geometry.resized((width, height))
    zenith = np.where(np.isfinite(local.zenith_deg), local.zenith_deg, np.inf)
    row, col = np.unravel_index(int(np.argmin(zenith)), zenith.shape)
    # Measured on the site camera: the ring centre is 10 px (at width 1232) from the zenith pixel.
    search = max(2, round(20 * width / 1232))
    # Each ring's own centre is searched within 4 px (at width 1232) of the common one: beyond the
    # largest offset measured on the reference days of the site camera (2.8 px).
    own = 4 * width / 1232
    sky = local.below(config.core_zenith) & ~fixed
    mask, fit = concentric_rings(
        np.mean(maps, axis=0).astype(np.float32),
        sky,
        (float(col), float(row)),
        search,
        config.ring_level,
        config.ring_halfwidth * width / 1232,
        own=own,
    )
    return mask, fit


def _usable_reference(
    candidates: list[DayResult], ratio: float
) -> tuple[list[DayResult], list[DayResult]]:
    """Reference days whose static fraction is in line with the others, and the rejected ones.

    On a foggy day the skyline has no edge and the sky leaks into the installation; such a day
    must not vote.
    """
    median = float(np.median([r.static_fraction for r in candidates]))
    usable: list[DayResult] = []
    rejected: list[DayResult] = []
    for result in candidates:
        ok = ratio * median <= result.static_fraction and ratio * result.static_fraction <= median
        (usable if ok else rejected).append(result)
    return usable, rejected


def _scaled_tolerance(config: StaticConfig, width: int) -> int:
    return round(config.tolerance * width / 1232) if config.tolerance else 0


def _unique_sources(
    days: Sequence[FrameSource], reference: Sequence[FrameSource]
) -> dict[str, FrameSource]:
    """Refuse ambiguous days before reading any cache or writing any output."""
    unique: dict[str, FrameSource] = {}
    fingerprints: dict[str, str] = {}
    for group in (days, reference):
        labels: set[str] = set()
        dates: set[str] = set()
        for source in group:
            day = source_date(source)
            name = source.label
            if not name or name in {".", ".."} or Path(name).name != name:
                raise ValueError(f"Invalid day label: {name!r}")
            if name in labels or day in dates:
                raise ValueError(f"Duplicate day label or date: {name!r} ({day})")
            labels.add(name)
            dates.add(day)
            fingerprint = source_fingerprint(source)
            if name in unique and fingerprints[name] != fingerprint:
                raise ValueError(f"Duplicate day label refers to different sources: {name!r}")
            unique[name] = source
            fingerprints[name] = fingerprint
    return unique


def _geometry_fingerprint(geometry: SkyGeometry) -> str:
    values = geometry.zenith_deg.astype("<f8")
    digest = hashlib.sha256(str(values.shape).encode() + values.tobytes())
    if geometry.azimuth_deg is not None:  # the sun guard changes the masks
        digest.update(geometry.azimuth_deg.astype("<f8").tobytes())
        digest.update(repr(geometry.site).encode())
    return digest.hexdigest()


def _check_installation(
    installation: FrozenInstallation,
    days: Sequence[FrameSource],
    geometry: SkyGeometry,
    config: StaticConfig,
) -> None:
    metadata = installation.metadata
    if installation_digest(metadata, installation.classes, installation.votes) != metadata.sha256:
        raise ValueError("Frozen installation checksum mismatch")
    if metadata.geometry_sha256 != _geometry_fingerprint(geometry):
        raise ValueError("Frozen installation has a different camera geometry")
    ignored = {"workers", "jobs", "min_days", "outlier_ratio", "tolerance", "reference_window"}
    parameters = config.model_dump(mode="json")
    mismatched = [
        key
        for key, value in parameters.items()
        if key not in ignored and metadata.parameters.get(key) != value
    ]
    if mismatched:
        raise ValueError(f"Frozen installation has incompatible parameters: {mismatched}")
    for source in days:
        day = date.fromisoformat(source_date(source))
        if not metadata.valid_from <= day < metadata.valid_until:
            raise ValueError(f"{day}: outside frozen installation validity period")


def _day_static(
    result: DayResult,
    fixed: NDArray[np.bool_],
    rings: NDArray[np.bool_],
    optics: RingFit | None,
    zenith: NDArray[np.float32],
    config: StaticConfig,
) -> NDArray[np.bool_]:
    """The day's static pixels completed once the installation and the rings are known.

    The rings of the optics (each a circle with its own centre, fitted once on the reference
    days and part of the installation) are left out of the faint marks, and their own light is
    removed within the band-pass reach before faint marks are measured. It depends on that day
    and on the installation only, never on the other days of the run.
    """
    width = fixed.shape[1]
    halfwidth = config.ring_halfwidth * width / 1232
    day_mask = result.mask
    if result.evidence is not None and result.evidence.flips is not None:
        day_mask = day_mask & ~skyline_glow(
            result.evidence.marks,
            result.evidence.flips,
            fixed & ~rings,
            zenith,
            config.core_zenith,
            _scaled_tolerance(config, width) + 1,
        )
    if result.faint is not None:
        inputs = result.faint
        if optics is not None and optics.rings:
            circles = optics.circles()
            reach = halfwidth + config.faint_scales[1] * width / 1232
            day, *parts = (
                without_rings(
                    c,
                    inputs.support,
                    optics.centre,
                    [r for _, r in circles],
                    reach,
                    centres=[c for c, _ in circles],
                )
                for c in (inputs.day, *inputs.parts)
            )
            inputs = replace(inputs, day=day, parts=tuple(parts))
        day_mask = day_mask | _faint(inputs, zenith, config, exclude=rings).marks
    return day_mask


def skyline_glow(
    marks: NDArray[np.bool_],
    flips: NDArray[np.bool_],
    installation: NDArray[np.bool_],
    zenith: NDArray[np.float32],
    core_zenith: float,
    reach: int,
) -> NDArray[np.bool_]:
    """Hourly marks that are the glow of a lit roof or wall, not marks on the optics.

    A sunlit white roof at the skyline makes the sky just above it brighter than the smooth sky
    surface, and the same roof in shadow darker: the band hugging the roof is significant in most
    hours, with the sign of its lighting. A mark is that glow when it touches the installation
    (within the slack + 1 px; the rings excluded), lies beyond the core zenith angle (at the
    skyline, where the buildings are) and changes sign at comparable amplitude (``flips``) over
    most of its pixels. Dirt that changes sign in the same way is on the optics, inside the
    core; foliage of the day keeps one sign.
    """
    near = binary_dilation(installation, iterations=max(1, reach))
    components, _ = label(marks, structure=np.ones((3, 3)))
    glow = np.zeros(marks.shape, dtype=bool)
    finite = np.where(np.isfinite(zenith), zenith, 0)
    for index, box in enumerate(find_objects(components), start=1):
        if box is None:
            continue
        own = components[box] == index
        if (
            near[box][own].any()
            and float(np.median(finite[box][own])) > core_zenith
            and float(flips[box][own].mean()) >= 0.5
        ):
            glow[box] |= own
    return glow


def day_classes(
    day_mask: NDArray[np.bool_],
    fixed: NDArray[np.bool_],
    domain: NDArray[np.bool_],
    core: NDArray[np.bool_],
    rings: NDArray[np.bool_],
    tolerance: int,
) -> NDArray[np.uint8]:
    """Classes of a day: its static pixels beyond the installation's slack are variable static;
    sky not connected to the open sky across the ring lines (``rings``, which hide no sky behind
    them) takes the class around it."""
    variable = variable_mask(day_mask, fixed, tolerance)
    return enclosed_sky(class_map(domain, fixed, variable), core, rings)


def run(
    days: Sequence[FrameSource],
    geometry: SkyGeometry,
    config: StaticConfig,
    output: Path,
    reference: Sequence[FrameSource] = (),
    cache: Path | None = None,
    *,
    installation: FrozenInstallation | None = None,
) -> RunReport:
    """Masks for ``days``; the installation is voted on ``reference`` (default: ``days``).

    Writes ``installation.png``, ``reference_votes.png`` (number of reference days on which each
    pixel is static), one ``classes_<day>.png`` and one ``overlay_<day>.jpg`` per
    processed day, and ``report.json`` into ``output``.
    """
    if not days:
        raise ValueError("No day to process")
    if installation is not None:
        if reference:
            raise ValueError("Choose a frozen installation or reference days, not both")
        _check_installation(installation, days, geometry, config)
    unique = _unique_sources(days, reference)
    _check_reference_period(days, reference, config.reference_window)
    with ThreadPoolExecutor(config.jobs) as pool:
        computed = pool.map(lambda s: compute_day(s, geometry, config, cache), unique.values())
        results = dict(zip(unique, computed, strict=True))
    shapes = {r.mask.shape for r in results.values()}
    if len(shapes) != 1:
        raise ValueError(f"Days have different image sizes: {sorted(shapes)}")
    if installation is None:
        candidates = [results[s.label] for s in (reference or days)]
        voters, rejected = _usable_reference(candidates, config.outlier_ratio)
        fixed = installation_mask([r.mask for r in voters], config.min_days)
        votes = np.sum([r.mask for r in voters], axis=0).astype(np.uint8)
        domain = voters[0].domain
        rings, optics = _permanent_rings(voters, fixed, geometry, config)
        rings = rings & domain
        band = _ring_band(rings, _scaled_tolerance(config, rings.shape[1]))
        fixed = _grown_installation(fixed & ~band, voters, config, band) | fixed | rings
        reference_days = [r.label for r in voters]
        rejected_days = [r.label for r in rejected]
    else:
        stored = installation.metadata.parameters.get("optics")
        optics = None if stored is None else RingFit.model_validate(stored)
        rings = (
            np.zeros(installation.classes.shape, dtype=bool)
            if optics is None
            else ring_mask(
                optics,
                installation.classes.shape,
                config.ring_halfwidth * installation.classes.shape[1] / 1232,
            )
        )
        fixed = installation.classes == 1
        domain = installation.classes != 3
        votes = installation.votes
        reference_days = installation.metadata.reference_days
        rejected_days = installation.metadata.rejected_reference_days
        if fixed.shape != next(iter(results.values())).mask.shape:
            raise ValueError("Frozen installation and daily images have different sizes")
    height, width = fixed.shape
    output.mkdir(parents=True, exist_ok=True)
    core = geometry.resized((width, height)).below(config.core_zenith)
    if installation is None:  # a frozen installation is used as frozen
        fixed = enclosed_sky(class_map(domain, fixed), core, rings) == 1
    Image.fromarray(class_map(domain, fixed)).save(output / "installation.png")
    Image.fromarray(votes).save(output / "reference_votes.png")
    tolerance = _scaled_tolerance(config, width)
    zenith = geometry.resized((width, height)).zenith_deg
    reports = []
    for source in days:
        result = results[source.label]
        day_mask = _day_static(result, fixed, rings, optics, zenith, config)
        classes = day_classes(day_mask, fixed, domain, core, rings, tolerance)
        variable = classes == 2
        visible = bool(
            fixed.any() and (result.mask & fixed).sum() >= config.outlier_ratio * fixed.sum()
        )
        metadata = MaskMetadata(
            day=date.fromisoformat(source_date(source)),
            label=result.label,
            skyline_visible=visible,
            variable_reliable=visible,
            installation_id=None if installation is None else installation.metadata.artifact_id,
        )
        Image.fromarray(classes).save(
            output / f"classes_{result.label}.png", pnginfo=png_metadata(metadata)
        )
        Image.fromarray(overlay(result.image, classes)).save(
            output / f"overlay_{result.label}.jpg", quality=90
        )
        reports.append(
            DayReport(
                label=result.label,
                date=metadata.day.isoformat(),
                variable_reliable=visible,
                frames_daytime=result.frames_daytime,
                frames_total=result.frames_total,
                static_fraction=result.static_fraction,
                variable_fraction=float(variable.sum() / max(int(domain.sum()), 1)),
                variable_components=int(label(variable, structure=np.ones((3, 3)))[1]),
                # Preserve the mask, while exposing a conservative quality flag to consumers.
                skyline_visible=visible,
            )
        )
    report = RunReport(
        config=config,
        geometry_sha256=_geometry_fingerprint(geometry),
        installation_id=None if installation is None else installation.metadata.artifact_id,
        size_wh=(width, height),
        reference_days=reference_days,
        rejected_reference_days=rejected_days,
        installation_fraction=float(fixed.sum() / max(int(domain.sum()), 1)),
        classes=CLASS_NAMES,
        days=reports,
        optics=optics,
    )
    (output / "report.json").write_text(json.dumps(report.model_dump(), indent=2) + "\n")
    return report
