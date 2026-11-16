"""End-to-end: frames of several days to installation and variable-static masks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import label

from skystatic.aggregate import Aggregates, aggregate_parts, daytime_frames, frame_size
from skystatic.frames import FrameSource
from skystatic.geometry import SkyGeometry
from skystatic.masks import CLASS_NAMES, class_map, installation_mask, variable_mask
from skystatic.render import overlay
from skystatic.segment import static_mask_all_parts

__all__ = ["DayResult", "RunReport", "StaticConfig", "compute_day", "run"]


class StaticConfig(BaseModel):
    """Parameters of the method. Lengths in pixels are given at a width of 1232."""

    model_config = ConfigDict(frozen=True)

    aggregate: Literal["mean", "geometric"] = "geometric"
    parts: int = Field(3, ge=1, description="Consecutive parts of a day that must all agree")
    max_side: int = Field(1232, ge=0, description="Working resolution; 0 keeps the native size")
    daytime_relative: float = Field(0.25, gt=0, le=1)
    daytime_floor: float = Field(0.0, ge=0, le=1)
    zenith_limit: float = Field(90.0, gt=0, description="Analysed domain, degrees")
    core_zenith: float = Field(60.0, gt=0, description="Where the sky certainly is, degrees")
    edge: float = Field(0.05, gt=0, description="Edge strength above which a pixel is an edge")
    background: int = Field(
        36, ge=0, description="Window of the local gradient level removed from edges, pixels"
    )
    min_days: float = Field(0.6, gt=0, le=1, description="Fraction of reference days")
    tolerance: int = Field(2, ge=0, description="Slack around the installation, pixels")
    outlier_ratio: float = Field(
        0.5,
        ge=0,
        lt=1,
        description="A reference day is dropped when its static fraction is below this ratio "
        "of the median one (fog: no visible skyline), or above its inverse",
    )
    workers: int = Field(16, ge=1, description="Decoding threads per day")
    jobs: int = Field(1, ge=1, description="Days processed at the same time")

    def aggregation_key(self) -> str:
        """Identifies cached averages: only what changes them."""
        fields = (self.parts, self.max_side, self.daytime_relative, self.daytime_floor)
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

    @property
    def static_fraction(self) -> float:
        return float(self.mask.sum() / max(int(self.domain.sum()), 1))


class DayReport(BaseModel):
    label: str
    frames_daytime: int
    frames_total: int
    static_fraction: float
    variable_fraction: float
    variable_components: int
    skyline_visible: bool


class RunReport(BaseModel):
    config: StaticConfig
    size_wh: tuple[int, int]
    reference_days: list[str]
    rejected_reference_days: list[str]
    installation_fraction: float
    classes: dict[int, str]
    days: list[DayReport]


def _load_or_aggregate(
    source: FrameSource, config: StaticConfig, cache: Path | None
) -> tuple[Aggregates, int, int]:
    path = None if cache is None else cache / f"{source.label}_{config.aggregation_key()}.npz"
    if path is not None and path.exists():
        with np.load(path) as stored:
            aggregates = Aggregates(
                stored["mean"], stored["geometric"], tuple(int(c) for c in stored["counts"])
            )
            return aggregates, int(stored["frames_daytime"]), int(stored["frames_total"])
    frames, total = daytime_frames(
        source, config.daytime_relative, config.daytime_floor, config.workers
    )
    if len(frames) < config.parts:
        raise ValueError(f"{source.label}: {len(frames)} daytime frames out of {total}")
    size = frame_size(source, frames, config.max_side)
    aggregates = aggregate_parts(source, frames, size, config.parts, config.workers)
    if path is not None:
        cache.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        temporary = path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            mean=aggregates.mean,
            geometric=aggregates.geometric,
            counts=np.array(aggregates.counts),
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
    """Static mask of one day: averages per part, then agreement between parts."""
    aggregates, daytime, total = _load_or_aggregate(source, config, cache)
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
    weights = np.array(aggregates.counts, dtype=np.float32)[:, None, None, None]
    image = (parts * weights).sum(axis=0) / weights.sum()
    return DayResult(source.label, mask, domain, image.astype(np.float32), daytime, total)


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


def run(
    days: Sequence[FrameSource],
    geometry: SkyGeometry,
    config: StaticConfig,
    output: Path,
    reference: Sequence[FrameSource] = (),
    cache: Path | None = None,
) -> RunReport:
    """Masks for ``days``; the installation is voted on ``reference`` (default: ``days``).

    Writes ``installation.png``, ``reference_votes.png`` (number of reference days on which each
    pixel is static), one ``classes_<day>.png`` and one ``overlay_<day>.jpg`` per
    processed day, and ``report.json`` into ``output``.
    """
    if not days:
        raise ValueError("No day to process")
    unique = {source.label: source for source in [*days, *reference]}
    with ThreadPoolExecutor(config.jobs) as pool:
        computed = pool.map(lambda s: compute_day(s, geometry, config, cache), unique.values())
        results = dict(zip(unique, computed, strict=True))
    shapes = {r.mask.shape for r in results.values()}
    if len(shapes) != 1:
        raise ValueError(f"Days have different image sizes: {sorted(shapes)}")
    candidates = [results[s.label] for s in (reference or days)]
    voters, rejected = _usable_reference(candidates, config.outlier_ratio)
    installation = installation_mask([r.mask for r in voters], config.min_days)
    votes = np.sum([r.mask for r in voters], axis=0).astype(np.uint8)
    domain = voters[0].domain
    height, width = installation.shape
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_map(domain, installation)).save(output / "installation.png")
    Image.fromarray(votes).save(output / "reference_votes.png")
    tolerance = _scaled_tolerance(config, width)
    reports = []
    for source in days:
        result = results[source.label]
        variable = variable_mask(result.mask, installation, tolerance)
        classes = class_map(domain, installation, variable)
        Image.fromarray(classes).save(output / f"classes_{result.label}.png")
        Image.fromarray(overlay(result.image, classes)).save(
            output / f"overlay_{result.label}.jpg", quality=90
        )
        reports.append(
            DayReport(
                label=result.label,
                frames_daytime=result.frames_daytime,
                frames_total=result.frames_total,
                static_fraction=result.static_fraction,
                variable_fraction=float(variable.sum() / max(int(domain.sum()), 1)),
                variable_components=int(label(variable, structure=np.ones((3, 3)))[1]),
                # Fog hides the skyline: the sky then leaks into the installation. The variable
                # static inside the sky is still valid, the day must just not serve as reference.
                skyline_visible=bool(
                    (result.mask & installation).sum() >= config.outlier_ratio * installation.sum()
                ),
            )
        )
    report = RunReport(
        config=config,
        size_wh=(width, height),
        reference_days=[r.label for r in voters],
        rejected_reference_days=[r.label for r in rejected],
        installation_fraction=float(installation.sum() / max(int(domain.sum()), 1)),
        classes=CLASS_NAMES,
        days=reports,
    )
    (output / "report.json").write_text(json.dumps(report.model_dump(), indent=2) + "\n")
    return report
