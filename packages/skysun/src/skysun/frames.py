"""Frames on disk: their UTC time from the file name, their pixels as 8-bit RGB."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

__all__ = ["FrameFile", "frame_time", "list_frames", "read_rgb"]

_STAMP = re.compile(r"(\d{14})")
_SUFFIXES = {".jp2", ".jpg", ".jpeg", ".png"}


def frame_time(path: Path) -> datetime:
    """UTC time of a frame from the last ``YYYYmmddHHMMSS`` in its file name."""
    stamps = _STAMP.findall(path.name)
    if not stamps:
        raise ValueError(f"{path.name}: no YYYYmmddHHMMSS timestamp in the file name")
    return datetime.strptime(stamps[-1], "%Y%m%d%H%M%S").replace(tzinfo=UTC)


@dataclass(frozen=True)
class FrameFile:
    path: Path
    time: datetime

    @property
    def stamp(self) -> str:
        return f"{self.time:%Y%m%d%H%M%S}"


def list_frames(sources: Iterable[Path]) -> list[FrameFile]:
    """Frames of files or folders (not recursive), in time order; duplicates refused."""
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(p for p in source.iterdir() if p.suffix.lower() in _SUFFIXES)
        elif source.is_file():
            files.append(source)
        else:
            raise FileNotFoundError(source)
    frames = sorted((FrameFile(p, frame_time(p)) for p in files), key=lambda f: f.time)
    stamps = [f.stamp for f in frames]
    if len(set(stamps)) != len(stamps):
        raise ValueError("Several frames have the same timestamp")
    return frames


def read_rgb(path: Path) -> NDArray[np.uint8]:
    """8-bit RGB pixels (the site camera's .jp2 files hold JPEG data; Pillow reads either)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
