"""Where frames come from: a directory of images or a zip archive of images.

A source lists its frames in time order and returns the encoded bytes of one frame. Decoding is
separate so that it can run in worker threads while archive access stays serialised.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import re
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

__all__ = [
    "DirectorySource",
    "Frame",
    "FrameSource",
    "ZipSource",
    "decode_rgb",
    "open_source",
    "parse_timestamp",
]

IMAGE_SUFFIXES = frozenset({".jp2", ".jpg", ".jpeg", ".png"})
_TIMESTAMP = re.compile(r"(\d{14})")


@dataclass(frozen=True, order=True)
class Frame:
    """One image of a source; sortable by time, then by name."""

    timestamp: datetime
    name: str


class FrameSource(Protocol):
    """A day (or any batch) of frames from one camera."""

    @property
    def label(self) -> str:
        """Short name used in reports and cache files."""

    def frames(self) -> list[Frame]:
        """Frames in time order."""

    def read(self, frame: Frame) -> bytes:
        """Encoded bytes of one frame."""


def parse_timestamp(name: str) -> datetime | None:
    """First ``YYYYMMDDhhmmss`` group found in a file name, if it is a valid date."""
    match = _TIMESTAMP.search(Path(name).name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _keep(name: str, pattern: str | None) -> bool:
    path = Path(name)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    return pattern is None or fnmatch.fnmatch(path.name, pattern)


def _to_frames(names: list[str], pattern: str | None) -> list[Frame]:
    frames = []
    for name in names:
        if not _keep(name, pattern):
            continue
        timestamp = parse_timestamp(name)
        if timestamp is not None:
            frames.append(Frame(timestamp, name))
    return sorted(frames)


def source_date(source: FrameSource) -> str:
    """ISO date of the images; an empty batch or several dates is not a day."""
    dates = sorted({frame.timestamp.date().isoformat() for frame in source.frames()})
    if len(dates) != 1:
        raise ValueError(f"Expected one image date, found {dates or 'no timestamped images'}")
    return dates[0]


def source_fingerprint(source: FrameSource) -> str:
    """Identify source, selection and file revisions independently of its display label."""
    digest = hashlib.sha256(b"skystatic-averages-v2")
    frames = source.frames()
    if isinstance(source, (DirectorySource, ZipSource)):
        digest.update(
            json.dumps([type(source).__name__, str(source.path.resolve()), source.pattern]).encode()
        )
        paths = (
            [source.path / frame.name for frame in frames]
            if isinstance(source, DirectorySource)
            else [source.path]
        )
        revisions = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        digest.update(json.dumps(revisions).encode())
    else:
        # A custom source need not expose filesystem paths.
        for frame in frames:
            digest.update(hashlib.sha256(source.read(frame)).digest())
    digest.update(json.dumps([(f.name, f.timestamp.isoformat()) for f in frames]).encode())
    return digest.hexdigest()


class DirectorySource:
    """Image files directly inside a directory."""

    def __init__(self, path: Path, pattern: str | None = None, label: str | None = None) -> None:
        self.path = path
        self.pattern = pattern
        self._label = label

    @property
    def label(self) -> str:
        return self._label or source_date(self)

    def frames(self) -> list[Frame]:
        return _to_frames([p.name for p in self.path.iterdir() if p.is_file()], self.pattern)

    def read(self, frame: Frame) -> bytes:
        return (self.path / frame.name).read_bytes()


class ZipSource:
    """Image files anywhere inside a zip archive."""

    def __init__(self, path: Path, pattern: str | None = None, label: str | None = None) -> None:
        self.path = path
        self.pattern = pattern
        self._label = label
        self._lock = threading.Lock()
        self._archive: zipfile.ZipFile | None = None

    @property
    def label(self) -> str:
        return self._label or source_date(self)

    def _open(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.path)
        return self._archive

    def frames(self) -> list[Frame]:
        with self._lock:
            names = self._open().namelist()
        return _to_frames(names, self.pattern)

    def read(self, frame: Frame) -> bytes:
        with self._lock:  # one file handle: member reads must not interleave
            return self._open().read(frame.name)


def open_source(path: Path, pattern: str | None = None, label: str | None = None) -> FrameSource:
    """Directory or ``.zip`` archive, chosen from the path."""
    if path.is_dir():
        return DirectorySource(path, pattern, label)
    if path.suffix.lower() == ".zip" and path.is_file():
        return ZipSource(path, pattern, label)
    raise FileNotFoundError(f"Not a directory of images nor a zip archive: {path}")


def decode_rgb(data: bytes, size: tuple[int, int]) -> NDArray[np.float32] | None:
    """RGB image as float32 in 0..255 at ``size`` (width, height); None if undecodable.

    Archives do contain empty or truncated files; they are skipped, not fatal.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format == "JPEG":
                image.draft("RGB", size)
            rgb = image.convert("RGB")
            if rgb.size != size:
                rgb = rgb.resize(size, Image.Resampling.BOX)
            return np.asarray(rgb, dtype=np.float32)
    except OSError:
        return None
