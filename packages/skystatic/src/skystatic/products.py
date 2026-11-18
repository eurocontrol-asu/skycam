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
from pydantic import BaseModel, ConfigDict, Field

from skystatic.optics import RingFit, ring_mask

__all__ = [
    "DayMask",
    "MaskMetadata",
    "OpticsRings",
    "RingCircle",
    "load_day_mask",
    "load_optics",
    "sky_mask",
]


class RingCircle(BaseModel):
    """One ring of the optics: its own centre and radius, pixels of the product's image size."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    centre: tuple[float, float]  # x, y
    radius: float
    width: float = 0.0  # outer minus inner radius; 0 for a thin line (lines mode)
    contrast: float  # peak contrast on the mean of the reference days, log units


class OpticsRings(BaseModel):
    """The ring circles fitted by skystatic, with the image size their pixels refer to.

    Written as ``optics.json`` next to the masks and stored in a frozen installation; read with
    :func:`load_optics`. ``halfwidth`` is the half-width (pixels) of the band masked about each
    thin line; ``fit.eye`` the radius of the central bright disc (0 if none).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    size_wh: tuple[int, int]
    halfwidth: float = Field(ge=0)
    fit: RingFit

    @property
    def circles(self) -> list[RingCircle]:
        """Every ring as a circle about its own centre."""
        out = []
        for (inner, outer, peak), (centre, _) in zip(
            self.fit.rings, self.fit.circles(), strict=True
        ):
            out.append(
                RingCircle(
                    centre=centre, radius=(inner + outer) / 2, width=outer - inner, contrast=peak
                )
            )
        return out

    def scaled(self, size: tuple[int, int]) -> OpticsRings:
        """The same circles for an image of ``size`` (width, height) and the same view."""
        width, height = size
        sx, sy = width / self.size_wh[0], height / self.size_wh[1]
        if abs(sx - sy) > 0.01 * sx:
            raise ValueError("The optics and the requested image have different aspect ratios")

        def point(p: tuple[float, float]) -> tuple[float, float]:
            return ((p[0] + 0.5) * sx - 0.5, (p[1] + 0.5) * sy - 0.5)

        fit = self.fit
        scaled = RingFit(
            centre=point(fit.centre),
            noise=fit.noise,
            rings=[(a * sx, b * sx, c) for a, b, c in fit.rings],
            eye=fit.eye * sx,
            centres=[point(c) for c in fit.centres],
        )
        return OpticsRings(size_wh=(width, height), halfwidth=self.halfwidth * sx, fit=scaled)

    def band(self) -> NDArray[np.bool_]:
        """The ring bands and the eye as masked by skystatic, at ``size_wh``."""
        return ring_mask(self.fit, (self.size_wh[1], self.size_wh[0]), self.halfwidth)


def load_optics(path: Path, size: tuple[int, int] | None = None) -> OpticsRings | None:
    """The ring circles of a skystatic run folder (``optics.json``), of an ``optics.json``
    file, or of a frozen installation NPZ (checksum verified); None when no ring was fitted.
    ``size`` (width, height): scale them to that image size."""
    if path.is_dir():
        path = path / "optics.json"
    if path.suffix == ".npz":
        from skystatic.installation import FrozenInstallation  # avoids a cycle at import

        product = FrozenInstallation.from_file(path).optics
    else:
        product = OpticsRings.model_validate_json(path.read_text())
    if product is None or size is None:
        return product
    return product.scaled(size)


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
