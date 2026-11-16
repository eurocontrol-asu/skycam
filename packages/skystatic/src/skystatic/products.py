"""Self-describing daily masks and the consumer API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, ConfigDict

__all__ = ["DayMask", "MaskMetadata", "load_day_mask", "sky_mask"]


class MaskMetadata(BaseModel):
    """Quality and provenance embedded in a daily PNG, independent of report.json."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    day: date
    label: str
    skyline_visible: bool
    variable_reliable: bool
    installation_id: str | None = None


def png_metadata(metadata: BaseModel) -> PngInfo:
    """Encode product metadata as a PNG text field."""
    info = PngInfo()
    info.add_text("skystatic", metadata.model_dump_json())
    return info


def read_metadata(path: Path) -> MaskMetadata:
    """Read a daily PNG header; legacy masks without quality metadata are refused."""
    with Image.open(path) as image:
        raw = image.info.get("skystatic")
    if not isinstance(raw, str):
        raise ValueError(f"{path}: missing skystatic metadata; regenerate this legacy mask")
    return MaskMetadata.model_validate_json(raw)


def read_classes(path: Path) -> NDArray[np.uint8]:
    """Read class indices without converting RGB or silently changing label values."""
    with Image.open(path) as image:
        if image.mode not in {"L", "P"}:
            raise ValueError(f"{path}: expected an indexed or grayscale class PNG")
        classes = np.asarray(image, dtype=np.uint8).copy()
    if not np.isin(classes, [0, 1, 2, 3]).all():
        raise ValueError(f"{path}: unknown class indices")
    return classes


def resize_classes(classes: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    """Resize class indices with nearest neighbours, preserving the camera's field of view."""
    width, height = size
    rows, cols = classes.shape
    if width <= 0 or height <= 0:
        raise ValueError("Mask size must be positive (width, height)")
    if abs(cols / rows - width / height) > 0.01 * cols / rows:
        raise ValueError("Mask and requested image have different aspect ratios")
    return np.asarray(
        Image.fromarray(classes).resize(size, Image.Resampling.NEAREST), dtype=np.uint8
    )


@dataclass(frozen=True)
class DayMask:
    """Daily classes together with their quality; sky is class zero only."""

    metadata: MaskMetadata
    classes: NDArray[np.uint8]

    @property
    def sky(self) -> NDArray[np.bool_]:
        """True for usable sky; installation, variable static and outside-domain are False."""
        return np.asarray(self.classes == 0, dtype=np.bool_)


def load_day_mask(path: Path, size: tuple[int, int] | None = None) -> DayMask:
    """Load one self-describing PNG, optionally resizing its class labels."""
    metadata = read_metadata(path)
    classes = read_classes(path)
    if size is not None:
        classes = resize_classes(classes, size)
    return DayMask(metadata, classes)


def sky_mask(
    output: Path,
    timestamp: datetime,
    size: tuple[int, int],
    *,
    allow_unreliable: bool = False,
) -> NDArray[np.bool_]:
    """Sky for the exact image date and size; no nearest-day fallback.

    Dates use the same calendar/clock as the image filenames; no implicit timezone
    conversion is performed. By default a day without a visible skyline is refused.
    """
    day = timestamp.date()
    matches = [
        path for path in sorted(output.glob("classes_*.png")) if read_metadata(path).day == day
    ]
    if not matches:
        raise KeyError(f"No daily mask for {day}")
    if len(matches) != 1:
        raise ValueError(f"Several daily masks for {day}: {matches}")
    product = load_day_mask(matches[0], size)
    if not allow_unreliable and (
        not product.metadata.variable_reliable or not product.metadata.skyline_visible
    ):
        raise ValueError(f"{day}: unreliable daily mask (skyline not visible)")
    return product.sky
