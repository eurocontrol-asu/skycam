"""Averaged images of a day.

Averaging the daytime frames erases what moves (clouds, sun) and keeps what is fixed. The day is
cut into consecutive parts averaged separately, so that a later step can ask for agreement
between them. Two averages are produced:

- the arithmetic mean (sum / N);
- the geometric mean (product of the frames scaled to 0..1, to the power 1/N, computed as a mean
  of logs), which is pulled down by the dark moments of a pixel and so strengthens dark fixed
  objects.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skystatic.frames import Frame, FrameSource, decode_rgb

__all__ = [
    "Aggregates",
    "aggregate_parts",
    "central_luminance",
    "daytime_frames",
    "frame_size",
    "select_daytime",
]

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_THUMBNAIL = (160, 128)


@dataclass(frozen=True)
class Aggregates:
    """Per-part averages, shape (parts, height, width, 3), values in 0..255."""

    mean: NDArray[np.float32]
    geometric: NDArray[np.float32]
    counts: tuple[int, ...]


def central_luminance(rgb: NDArray[np.float32]) -> float:
    """Median luminance (0..1) of the central third of an image."""
    h, w = rgb.shape[:2]
    centre = rgb[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    return float(np.median(centre @ _LUMA) / 255)


def select_daytime(levels: list[float | None], relative: float, floor: float) -> list[bool]:
    """Frames bright enough to be daytime.

    A frame is kept when its level reaches ``relative`` times the 95th percentile of the batch
    and the absolute ``floor``. The relative rule adapts to the exposure of the camera; the
    floor rejects batches that are dark throughout.
    """
    known = [level for level in levels if level is not None]
    if not known:
        return [False] * len(levels)
    threshold = max(floor, relative * float(np.percentile(known, 95)))
    return [level is not None and level >= threshold for level in levels]


def frame_size(source: FrameSource, frames: list[Frame], max_side: int) -> tuple[int, int]:
    """(width, height) of the working resolution: native, or reduced to ``max_side``."""
    for frame in frames:
        try:
            with Image.open(BytesIO(source.read(frame))) as image:
                width, height = image.size
        except OSError:
            continue
        factor = min(1.0, max_side / max(width, height)) if max_side else 1.0
        return round(width * factor), round(height * factor)
    raise ValueError(f"No decodable frame in {source.label}")


def aggregate_parts(
    source: FrameSource,
    frames: list[Frame],
    size: tuple[int, int],
    parts: int,
    workers: int = 16,
) -> Aggregates:
    """Arithmetic and geometric means of ``frames`` over ``parts`` consecutive groups."""
    if parts < 1:
        raise ValueError("parts must be at least 1")
    if len(frames) < parts:
        raise ValueError(f"{len(frames)} frames cannot be cut into {parts} parts")
    width, height = size
    bounds = np.linspace(0, len(frames), parts + 1).astype(int)
    mean = np.zeros((parts, height, width, 3), dtype=np.float64)
    log_mean = np.zeros_like(mean)
    counts = []

    def load(frame: Frame) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
        rgb = decode_rgb(source.read(frame), size)
        return None if rgb is None else (rgb, np.log((rgb + 1.0) / 256.0))

    with ThreadPoolExecutor(workers) as pool:
        for part in range(parts):
            group = frames[bounds[part] : bounds[part + 1]]
            used = 0
            for loaded in pool.map(load, group):
                if loaded is None:
                    continue
                mean[part] += loaded[0]
                log_mean[part] += loaded[1]
                used += 1
            if used == 0:
                raise ValueError(f"Part {part} of {source.label} has no decodable frame")
            mean[part] /= used
            log_mean[part] /= used
            counts.append(used)
    geometric = 256.0 * np.exp(log_mean) - 1.0
    return Aggregates(mean.astype(np.float32), geometric.astype(np.float32), tuple(counts))


def daytime_frames(
    source: FrameSource,
    relative: float,
    floor: float,
    workers: int = 16,
) -> tuple[list[Frame], int]:
    """Daytime frames of a source and the total number of frames it holds."""
    frames = source.frames()

    def level(frame: Frame) -> float | None:
        rgb = decode_rgb(source.read(frame), _THUMBNAIL)
        return None if rgb is None else central_luminance(rgb)

    with ThreadPoolExecutor(workers) as pool:
        levels = list(pool.map(level, frames))
    keep = select_daytime(levels, relative, floor)
    return [frame for frame, kept in zip(frames, keep, strict=True) if kept], len(frames)
