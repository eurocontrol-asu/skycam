"""Synthetic camera: smooth sky, a moving cloud, a fixed structure, optional dirt."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

SIZE = 96  # square images
STRUCTURE = (slice(60, 80), slice(70, 74))  # fixed dark mast inside the analysed disc
DIRT = (slice(44, 50), slice(30, 36))  # on the optics, only on "dirty" days


def zenith_map(size: int = SIZE) -> NDArray[np.float32]:
    """Equidistant fisheye: 0 degrees at the centre, 100 degrees at the image edge."""
    y, x = np.indices((size, size))
    radius = np.hypot(y - (size - 1) / 2, x - (size - 1) / 2)
    return (radius / (size / 2) * 100).astype(np.float32)


def scene(index: int, dirty: bool = False, night: bool = False) -> NDArray[np.uint8]:
    """One frame; the cloud moves with ``index``, the mast and the dirt do not."""
    y, x = np.indices((SIZE, SIZE))
    sky = 150 + 40 * (x / SIZE)  # smooth gradient
    cloud = 60 * np.exp(-(((x - (10 + 3 * index) % SIZE) ** 2) + (y - 30) ** 2) / 60)
    image = sky + cloud
    image[STRUCTURE] = 25
    if dirty:
        image[DIRT] = 90
    image[zenith_map() >= 99] = 12  # past the horizon a fisheye sees its dark housing
    if night:
        image = image * 0.02
    rgb = np.stack([image * 0.9, image * 0.95, image], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def night_frame() -> NDArray[np.uint8]:
    """What the GVCCS site camera writes at night: one flat magenta value, brighter than the day."""
    return np.broadcast_to(np.array([96, 51, 84], np.uint8), (SIZE, SIZE, 3)).copy()


def frame_names(count: int, day: datetime) -> list[str]:
    return [f"image_{(day + timedelta(minutes=i)):%Y%m%d%H%M%S}.png" for i in range(count)]


def encode(image: NDArray[np.uint8]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def write_day(folder: Path, day: datetime, count: int = 24, dirty: bool = False) -> Path:
    folder.mkdir(parents=True)
    for index, name in enumerate(frame_names(count, day)):
        (folder / name).write_bytes(encode(scene(index, dirty=dirty)))
    return folder


def write_zip_day(path: Path, day: datetime, count: int = 24, dirty: bool = False) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for index, name in enumerate(frame_names(count, day)):
            archive.writestr(f"CAM/{day:%Y/%m/%d}/{name}", encode(scene(index, dirty=dirty)))
    return path


@pytest.fixture
def zenith_file(tmp_path: Path) -> Path:
    path = tmp_path / "zenith.npz"
    np.savez(path, zenith_deg=zenith_map())
    return path
