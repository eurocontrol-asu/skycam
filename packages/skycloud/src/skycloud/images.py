"""Frames: where they are, how they are read, their timestamps and the camera pedestal.

A frame is named by its UTC timestamp ``YYYYmmddHHMMSS`` and stored as
``ROOT/YYYY/MM/DD/image_<timestamp>.jp2``. Despite the extension the files hold 8-bit JPEG
data; Pillow decodes them by content (as skystatic does), never by extension.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

__all__ = [
    "frame_path",
    "pedestal",
    "read_rgb",
    "shift_stamp",
    "stamp_minutes",
    "when_of",
]

U8 = NDArray[np.uint8]


def when_of(ts: str) -> datetime:
    """The UTC instant of a frame timestamp."""
    return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def stamp_minutes(ts: str) -> float:
    """The timestamp in minutes since the epoch (the unit of every temporal kernel)."""
    return when_of(ts).timestamp() / 60.0


def shift_stamp(ts: str, seconds: int) -> str:
    """The timestamp ``seconds`` later (earlier if negative)."""
    t = datetime.strptime(ts, "%Y%m%d%H%M%S") + timedelta(seconds=seconds)
    return t.strftime("%Y%m%d%H%M%S")


def frame_path(root: Path, ts: str) -> Path:
    """``ROOT/YYYY/MM/DD/image_<ts>.jp2``."""
    return Path(root) / ts[:4] / ts[4:6] / ts[6:8] / f"image_{ts}.jp2"


def read_rgb(path: Path) -> U8:
    """The frame as an RGB uint8 array (height, width, 3), decoded by content."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def pedestal(img: U8) -> NDArray[np.float32]:
    """The camera pedestal (R, G, B): mean of the four 60 px corners, outside the fisheye."""
    c = np.concatenate(
        [
            img[:60, :60].reshape(-1, 3),
            img[:60, -60:].reshape(-1, 3),
            img[-60:, :60].reshape(-1, 3),
            img[-60:, -60:].reshape(-1, 3),
        ]
    )
    return np.asarray(c.astype(np.float32).mean(0))
