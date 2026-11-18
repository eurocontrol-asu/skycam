"""Averaged images of a day.

Only daytime frames are averaged. A daytime frame is recognised by its STRUCTURE, not by its
brightness: the fisheye circle is visible, i.e. the sky around the zenith is clearly brighter
than the part of the frame that looks beyond the horizon (lens housing, unexposed corners). At
night some cameras write a flat uniform frame (the GVCCS site camera writes flat magenta, brighter
in its centre than a daytime frame), for which that ratio is 1.

Averaging the daytime frames erases what moves (clouds, sun) and keeps what is fixed. The day is
cut into consecutive parts averaged separately, so that a later step can ask for agreement
between them. Two averages are produced:

- the arithmetic mean (sum / N);
- the geometric mean (product of the frames scaled to 0..1, to the power 1/N, computed as a mean
  of logs), which is pulled down by the dark moments of a pixel and so strengthens dark fixed
  objects.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skystatic.frames import Frame, FrameSource, decode_rgb
from skystatic.geometry import SkyGeometry

__all__ = [
    "Aggregates",
    "aggregate_parts",
    "bounded_map",
    "central_luminance",
    "circle_contrast",
    "daytime_frames",
    "frame_size",
    "select_daytime",
]

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# Side of the thumbnails used to classify frames: enough to see the fisheye circle, cheap to decode.
_THUMBNAIL_SIDE = 160
# Zenith angle of the horizon: pixels beyond it see no sky (housing, ground, unexposed corners).
HORIZON_DEG = 90.0


def bounded_map[T, R](
    pool: ThreadPoolExecutor, function: Callable[[T], R], items: Iterable[T], ahead: int
) -> Iterator[R]:
    """``pool.map`` in order, with at most ``ahead`` results waiting in memory.

    ``ThreadPoolExecutor.map`` submits every item at once, so decoded frames pile up whenever
    decoding is faster than their accumulation: a day of frames would then sit in memory.
    """
    pending: deque[Future[R]] = deque()
    for item in items:
        pending.append(pool.submit(function, item))
        if len(pending) >= ahead:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


@dataclass(frozen=True)
class Aggregates:
    """Per-part averages, shape (parts, height, width, 3), values in 0..255.

    ``windows`` (windows, 2, height, width): for each short window of the day (clock hours by
    default), the log of the mean luminance and the mean of the log luminance; ``window_counts``
    the frames in each; ``window_starts`` their start, in the clock of the file names.
    """

    mean: NDArray[np.float32]
    geometric: NDArray[np.float32]
    counts: tuple[int, ...]
    windows: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 2, 0, 0), dtype=np.float32)
    )
    window_counts: tuple[int, ...] = ()
    window_starts: tuple[datetime, ...] = ()


def central_luminance(rgb: NDArray[np.float32]) -> float:
    """Median luminance (0..1) of the central third of an image."""
    h, w = rgb.shape[:2]
    centre = rgb[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    return float(np.median(centre @ _LUMA) / 255)


def circle_contrast(
    rgb: NDArray[np.float32], sky: NDArray[np.bool_], beyond: NDArray[np.bool_]
) -> float | None:
    """Mean of the ``sky`` pixels over that of the pixels ``beyond`` the horizon, in the colour
    channel where that ratio is largest.

    A flat frame (night output of some cameras) has 1 in every channel; a daytime frame shows
    its fisheye circle. Per channel, because a camera may paint the area beyond the circle with a
    bright colour during the day (the site camera: magenta, which lifts red and blue but not
    green). None when the frame has no pixel beyond the horizon: the rule then does not apply.
    """
    if not beyond.any() or not sky.any():
        return None
    inside = rgb[sky].mean(axis=0)
    outside = np.maximum(rgb[beyond].mean(axis=0), 1.0)
    return float((inside / outside).max())


def select_daytime(
    levels: list[float | None],
    relative: float,
    floor: float,
    contrasts: list[float | None] | None = None,
    min_contrast: float = 2.0,
) -> list[bool]:
    """Frames that are daytime.

    A frame with a ``contrast`` (see :func:`circle_contrast`) is daytime when its fisheye circle is
    visible: contrast at least ``min_contrast``. Brightness is not used for it: on the GVCCS site
    camera the night frames are brighter than the day, and an overcast day varies by more than a
    factor four in brightness while its circle stays visible.

    A frame without a contrast (the camera has no pixel beyond the horizon, or no geometry was
    given) falls back on brightness: its level must reach ``relative`` times the 95th percentile of
    those frames. Every frame must also reach the absolute ``floor``.
    """
    if contrasts is None:
        contrasts = [None] * len(levels)
    pairs = list(zip(levels, contrasts, strict=True))
    unstructured = [lv for lv, ct in pairs if lv is not None and ct is None]
    relative_level = relative * float(np.percentile(unstructured, 95)) if unstructured else 0.0
    keep = []
    for level, contrast in pairs:
        if level is None or level < floor:
            keep.append(False)
        elif contrast is None:
            keep.append(level >= relative_level)
        else:
            keep.append(contrast >= min_contrast)
    return keep


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


class _Windows:
    """Running sums of luminance and log luminance over consecutive clock windows."""

    def __init__(self, minutes: int, shape: tuple[int, int]) -> None:
        self.minutes = minutes
        self.shape = shape
        self.key: datetime | None = None
        self.total = np.zeros(shape, dtype=np.float64)
        self.log_total = np.zeros(shape, dtype=np.float64)
        self.used = 0
        self.done: list[tuple[datetime, int, NDArray[np.float32]]] = []

    def _start(self, time: datetime) -> datetime:
        minute = (time.hour * 60 + time.minute) // self.minutes * self.minutes
        return time.replace(hour=minute // 60, minute=minute % 60, second=0, microsecond=0)

    def add(self, time: datetime, luminance: NDArray[np.float32]) -> None:
        start = self._start(time)
        if start != self.key:
            self.close()
            self.key = start
        self.total += luminance
        self.log_total += np.log(np.maximum(luminance, 0.5))
        self.used += 1

    def close(self) -> None:
        if self.key is not None and self.used:
            pair = np.stack(
                [np.log(np.maximum(self.total / self.used, 0.5)), self.log_total / self.used]
            ).astype(np.float32)
            self.done.append((self.key, self.used, pair))
        self.total[:] = 0
        self.log_total[:] = 0
        self.used = 0

    def result(self) -> tuple[NDArray[np.float32], tuple[int, ...], tuple[datetime, ...]]:
        """Windows holding at least half the frames of a typical window (dawn/dusk stubs out)."""
        self.close()
        counts = [used for _, used, _ in self.done]
        if not counts:
            return np.zeros((0, 2, *self.shape), dtype=np.float32), (), ()
        kept = [w for w in self.done if w[1] >= np.median(counts) / 2]
        return (
            np.stack([w[2] for w in kept]),
            tuple(w[1] for w in kept),
            tuple(w[0] for w in kept),
        )


def aggregate_parts(
    source: FrameSource,
    frames: list[Frame],
    size: tuple[int, int],
    parts: int,
    workers: int = 16,
    window_minutes: int = 60,
) -> Aggregates:
    """Arithmetic and geometric means of ``frames`` over ``parts`` consecutive groups, and the
    luminance of each ``window_minutes`` clock window. Frames are accumulated one by one."""
    if parts < 1:
        raise ValueError("parts must be at least 1")
    if len(frames) < parts:
        raise ValueError(f"{len(frames)} frames cannot be cut into {parts} parts")
    width, height = size
    bounds = np.linspace(0, len(frames), parts + 1).astype(int)
    mean = np.zeros((parts, height, width, 3), dtype=np.float64)
    log_mean = np.zeros_like(mean)
    counts = []
    windows = _Windows(window_minutes, (height, width))

    def load(frame: Frame) -> tuple[Frame, NDArray[np.float32]] | None:
        rgb = decode_rgb(source.read(frame), size)
        return None if rgb is None else (frame, rgb)

    with ThreadPoolExecutor(workers) as pool:
        for part in range(parts):
            group = frames[bounds[part] : bounds[part + 1]]
            used = 0
            for loaded in bounded_map(pool, load, group, 2 * workers):
                if loaded is None:
                    continue
                frame, rgb = loaded
                mean[part] += rgb
                log_mean[part] += np.log((rgb + 1.0) / 256.0)
                windows.add(frame.timestamp, rgb @ _LUMA)
                used += 1
            if used == 0:
                raise ValueError(f"Part {part} of {source.label} has no decodable frame")
            mean[part] /= used
            log_mean[part] /= used
            counts.append(used)
    geometric = 256.0 * np.exp(log_mean) - 1.0
    stack, window_counts, starts = windows.result()
    return Aggregates(
        mean.astype(np.float32),
        geometric.astype(np.float32),
        tuple(counts),
        stack,
        window_counts,
        starts,
    )


def daytime_frames(
    source: FrameSource,
    relative: float,
    floor: float,
    workers: int = 16,
    *,
    geometry: SkyGeometry | None = None,
    core_zenith: float = 60.0,
    min_contrast: float = 2.0,
) -> tuple[list[Frame], int]:
    """Daytime frames of a source and the total number of frames it holds.

    With a ``geometry``, a frame must show its fisheye circle: the sky within ``core_zenith``
    at least ``min_contrast`` times brighter than the pixels beyond the horizon. Without one,
    only the brightness rule applies.
    """
    frames = source.frames()
    if not frames:
        return [], 0
    size = frame_size(source, frames, _THUMBNAIL_SIDE)
    sky = beyond = None
    if geometry is not None:
        local = geometry.resized(size)
        sky = local.below(core_zenith)
        beyond = ~local.below(HORIZON_DEG)

    def measure(frame: Frame) -> tuple[float | None, float | None]:
        rgb = decode_rgb(source.read(frame), size)
        if rgb is None:
            return None, None
        contrast = None if sky is None or beyond is None else circle_contrast(rgb, sky, beyond)
        return central_luminance(rgb), contrast

    with ThreadPoolExecutor(workers) as pool:
        measured = list(bounded_map(pool, measure, frames, 2 * workers))
    keep = select_daytime(
        [m[0] for m in measured], relative, floor, [m[1] for m in measured], min_contrast
    )
    return [frame for frame, kept in zip(frames, keep, strict=True) if kept], len(frames)
