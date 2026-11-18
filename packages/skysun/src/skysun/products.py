"""Products of a run, one set per frame, and the run itself (streaming, resumable).

Per frame, under ``<output>/<YYYY-MM-DD>/``:

- ``zone_<stamp>.png``: the sun zone, 8-bit grayscale, code = round(255 x value), value =
  code / 255: the brightness the sun's glow adds, as a share of the headroom between the sky's
  own level and saturation (1 on the sun's saturated pixels).
- ``reflections_<stamp>.png``: the reflections of the optics on their own scale, same encoding:
  value x the table's reference contrast = the predicted contrast (1 = the strongest reflections
  of this optic, clipped).
- ``sun_<stamp>.json``: the frame's numbers (:class:`FrameInfo`), written last: a frame whose
  JSON exists is done and skipped when the run is started again.

Every pixel of the frame has a code in both layers; pixels without a sky direction are 0.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skysun.frames import FrameFile, read_rgb
from skysun.pipeline import (
    FrameInfo,
    FrameMeasure,
    SunCamera,
    SunMap,
    measure_frame,
    render,
    temporal_profile,
)

__all__ = ["RunResult", "decode", "encode", "frame_paths", "load_layer", "run_frames"]

LAYERS = ("zone", "reflections")


def encode(value: NDArray[np.float32]) -> NDArray[np.uint8]:
    """[0, 1] to the 8-bit code round(255 x value)."""
    return np.asarray(np.rint(np.clip(value, 0.0, 1.0) * 255), dtype=np.uint8)


def decode(code: NDArray[np.uint8]) -> NDArray[np.float32]:
    """The 8-bit code back to a value in [0, 1] (within 1/510 of the original)."""
    return np.asarray(np.asarray(code, dtype=np.float32) / np.float32(255), dtype=np.float32)


def frame_paths(output: Path, frame: FrameFile) -> dict[str, Path]:
    """Where the products of ``frame`` are written."""
    folder = output / f"{frame.time:%Y-%m-%d}"
    paths = {layer: folder / f"{layer}_{frame.stamp}.png" for layer in LAYERS}
    paths["json"] = folder / f"sun_{frame.stamp}.json"  # the frame's numbers, not a layer
    return paths


def load_layer(path: Path) -> NDArray[np.float32]:
    """One written layer as values in [0, 1]."""
    with Image.open(path) as image:
        return decode(np.asarray(image, dtype=np.uint8))


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}{path.suffix}")
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _png_writer(code: NDArray[np.uint8]) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        Image.fromarray(code).save(path, format="PNG", compress_level=6)

    return write


def write_products(result: SunMap, paths: dict[str, Path]) -> None:
    """The two layers, then the JSON (the mark that the frame is done)."""
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    for layer, value in zip(LAYERS, (result.zone, result.reflections), strict=True):
        _atomic(paths[layer], _png_writer(encode(value)))
    text = result.info.model_dump_json(indent=1) + "\n"

    def write_json(path: Path) -> None:
        path.write_text(text)

    _atomic(paths["json"], write_json)


@dataclass
class RunResult:
    """Frames mapped (or found already done) and frames that could not be read."""

    done: list[FrameInfo] = field(default_factory=list)
    unreadable: list[Path] = field(default_factory=list)


def _needed(frames: Sequence[FrameFile], done: list[bool], window: float) -> list[bool]:
    """Frames to measure: those not done, and the done ones within the window of one not done
    (their profiles enter its temporal median)."""
    todo = [f.time.timestamp() for f, d in zip(frames, done, strict=True) if not d]
    stamps = np.asarray(todo)
    return [
        not d or (stamps.size > 0 and bool((np.abs(stamps - f.time.timestamp()) <= window).any()))
        for f, d in zip(frames, done, strict=True)
    ]


def run_frames(
    frames: Sequence[FrameFile],
    camera: SunCamera,
    sky: Callable[[FrameFile], NDArray[np.bool_]],
    output: Path,
) -> RunResult:
    """Both layers of every frame, in time order; frames already done are skipped. A frame is
    rendered once the frames up to ``temporal_window_s`` after it are measured, with the median
    profile of its neighbours (``temporal_profile``): a few measured frames are held, never the
    pixels of more than one frame. ``sky(frame)``: the static mask's sky pixels for that frame.
    A file that cannot be decoded gets no product and is listed in the result."""
    result = RunResult()
    window = camera.params.zone.temporal_window_s
    ordered = sorted(frames, key=lambda f: f.time)
    paths = [frame_paths(output, f) for f in ordered]
    done = [p["json"].exists() for p in paths]
    needed = _needed(ordered, done, window)
    held: list[tuple[int, FrameMeasure]] = []  # measured frames still useful, in time order

    def flush(until: float | None) -> None:
        while held:
            pending = [(i, m) for i, m in held if not done[i]]
            if not pending:
                held.clear()
                return
            index, first = pending[0]
            if until is not None and first.time.timestamp() + window >= until:
                return
            position = next(k for k, (i, _) in enumerate(held) if i == index)
            measures = [m for _, m in held]
            value, used = temporal_profile(measures, position, camera)
            mapped = render(first, camera, value, used)
            write_products(mapped, paths[index])
            result.done.append(mapped.info)
            done[index] = True
            horizon = first.time.timestamp() - window
            held[:] = [(i, m) for i, m in held if m.time.timestamp() >= horizon or not done[i]]

    for index, frame in enumerate(ordered):
        if done[index] and not needed[index]:
            result.done.append(FrameInfo.model_validate_json(paths[index]["json"].read_text()))
            continue
        flush(frame.time.timestamp())
        try:
            rgb = read_rgb(frame.path)
        except OSError:  # PIL's UnidentifiedImageError and truncated files are OSErrors
            result.unreadable.append(frame.path)
            continue
        held.append((index, measure_frame(rgb, frame.time, camera, sky(frame))))
        if done[index]:
            result.done.append(FrameInfo.model_validate_json(paths[index]["json"].read_text()))
        del rgb
    flush(None)
    return result
