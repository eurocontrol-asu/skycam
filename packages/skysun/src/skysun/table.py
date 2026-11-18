"""The reflection table of a camera: calibrated once, embedded, checksummed.

For every structure of the optics (ring or eye), every sun-elevation bin and every direction of
a sector relative to the sun's direction (seen from the structure's centre in the image), the
median contrast of its light on clear archive frames where the sun shone. With the table travel
the circles it was measured on, the visibility ramp (how "the sun shines" is read), the
reference contrast worth 1 on the reflections layer, the contrast below which a sector's light
is negligible, and the provenance
(frames, days, period, command); a SHA-256 over all of it and the arrays detects any change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "PACKAGED_TABLE",
    "ReflectionTable",
    "StructureInfo",
    "TableMetadata",
    "VisibilityRamp",
    "packaged_table",
]

F32 = NDArray[np.float32]
PACKAGED_TABLE = "gvccs_bretigny.npz"


class StructureInfo(BaseModel):
    """A structure the table was measured on: native pixels of the camera's frames."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    centre: tuple[float, float]
    radius: float
    disc: bool


class VisibilityRamp(BaseModel):
    """Whether the sun shines, 0..1.

    Read from the zone's core (the pixels of the field within ``core_deg`` of the sun, static or
    not): 0 when none is saturated, 1 from ``high`` (one value per sun-elevation bin, linear in
    elevation between bin centres), linear between. When most of the core is not sky (the sun
    behind the tree line or the installation), the core says nothing and the glow of the profile
    decides: 0 at ``glow_low`` or below, 1 at ``glow_high`` or above. ``origin`` says how all of
    them were measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    elevation_edges: list[float]
    high: list[float]
    glow_low: float = Field(ge=0, le=1)
    glow_high: float = Field(gt=0, le=1)
    origin: str

    @model_validator(mode="after")
    def consistent(self) -> VisibilityRamp:
        """One top per elevation bin, each in (0, 1]; an increasing glow ramp."""
        if len(self.high) != len(self.elevation_edges) - 1 or not self.high:
            raise ValueError("One ramp top per elevation bin")
        if min(self.high) <= 0 or max(self.high) > 1:
            raise ValueError("Ramp tops must lie in (0, 1]")
        if self.glow_high <= self.glow_low:
            raise ValueError("glow_high must exceed glow_low")
        return self

    def from_core(self, core_saturation: float, elevation_deg: float) -> float:
        """The core's reading: saturated share over the top at this elevation."""
        edges = np.asarray(self.elevation_edges)
        top = float(np.interp(elevation_deg, (edges[:-1] + edges[1:]) / 2, self.high))
        return float(np.clip(core_saturation / top, 0.0, 1.0))

    def from_glow(self, glow: float) -> float:
        """The profile's reading, for a core that is mostly not sky."""
        return float(np.clip((glow - self.glow_low) / (self.glow_high - self.glow_low), 0.0, 1.0))


class TableMetadata(BaseModel):
    """Everything but the arrays; ``sha256`` covers this (itself excluded) and the arrays."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    camera_id: str
    image_size_wh: tuple[int, int]
    structures: list[StructureInfo]
    elevation_edges: list[float]
    sectors: int
    saturation: int
    core_deg: float
    visibility: VisibilityRamp
    reference_contrast: float = Field(
        gt=0,
        description="Contrast worth 1 on the reflections layer (value x reference = contrast): "
        "the 99th percentile of single-frame sector contrasts over the clear calibration frames "
        "where the sun fully shines, rings pooled, the eye left out",
    )
    negligible_contrast: float = Field(
        ge=0,
        description="A sector's predicted contrast below this is negligible (skystatic's level "
        "of the clean optics' own texture and ring lines, 0.3 %)",
    )
    frames: int
    days: list[str]
    period: tuple[str, str]
    selection: str
    command: str
    sha256: str = ""

    @property
    def negligible_value(self) -> float:
        """The negligible contrast on the reflections layer's scale."""
        return self.negligible_contrast / self.reference_contrast


def _digest(metadata: TableMetadata, arrays: dict[str, NDArray[np.generic]]) -> str:
    data = metadata.model_dump(mode="json", exclude={"sha256"})
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReflectionTable:
    """contrast[structure, elevation bin, relative sector]: the median contrast (band over
    flanks minus 1) of the sector, and the number of frames per (structure, elevation bin)."""

    metadata: TableMetadata
    contrast: F32
    frames: NDArray[np.int32]

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        return {"contrast": self.contrast, "frames": self.frames}

    def sealed(self) -> ReflectionTable:
        """The same table with its checksum."""
        metadata = self.metadata.model_copy(
            update={"sha256": _digest(self.metadata, self.arrays())}
        )
        return ReflectionTable(metadata, self.contrast, self.frames)

    def save(self, path: Path) -> None:
        """Write a portable NPZ (no pickle); never overwrites."""
        if path.exists():
            raise FileExistsError(path)
        table = self.sealed()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.array(table.metadata.model_dump_json()),
                **table.arrays(),  # type: ignore[arg-type]
            )

    @classmethod
    def from_file(cls, path: Path) -> ReflectionTable:
        """Load and verify the checksum and the shapes."""
        with np.load(path, allow_pickle=False) as stored:
            metadata = TableMetadata.model_validate_json(str(stored["metadata"].item()))
            table = cls(
                metadata,
                stored["contrast"].astype(np.float32),
                stored["frames"].astype(np.int32),
            )
        shape = (len(metadata.structures), len(metadata.elevation_edges) - 1, metadata.sectors)
        if table.contrast.shape != shape:
            raise ValueError(f"{path}: table arrays do not match their metadata {shape}")
        if table.frames.shape != shape[:2]:
            raise ValueError(f"{path}: frame counts do not match their metadata")
        if _digest(metadata, table.arrays()) != metadata.sha256:
            raise ValueError(f"{path}: reflection table checksum mismatch")
        for array in (table.contrast, table.frames):
            array.flags.writeable = False
        return table

    def lookup(self, structure: int, elevation: float, relative_deg: F32) -> F32:
        """The table's contrast for one structure at the sun's ``elevation`` (deg) and sectors
        at ``relative_deg`` from the sun's direction: linear in both, the relative angle
        wrapping around, the elevation held at the first and last bin centres; never below 0."""
        edges = np.asarray(self.metadata.elevation_edges)
        centres = (edges[:-1] + edges[1:]) / 2
        position = float(np.interp(elevation, centres, np.arange(centres.size)))
        e0 = int(np.floor(position))
        e1 = min(e0 + 1, centres.size - 1)
        fe = position - e0
        rows = (1 - fe) * self.contrast[structure, e0] + fe * self.contrast[structure, e1]
        sectors = self.metadata.sectors
        step = 360.0 / sectors
        p = np.mod(relative_deg, 360.0) / step - 0.5
        k0 = np.floor(p).astype(np.int64)
        f = (p - k0).astype(np.float32)
        out = (1 - f) * rows[k0 % sectors] + f * rows[(k0 + 1) % sectors]
        return np.asarray(np.maximum(out, 0.0), dtype=np.float32)

    def layer_value(self, contrast: F32) -> F32:
        """Contrast on the reflections layer's own scale: contrast / reference, clipped to 1."""
        scaled = contrast / np.float32(self.metadata.reference_contrast)
        return np.asarray(np.clip(scaled, 0.0, 1.0), dtype=np.float32)

    def check_structures(self, structures: list[StructureInfo], tolerance_px: float = 2.0) -> None:
        """Refuse circles other than those the table was measured on (same order, centres and
        radii within ``tolerance_px``)."""
        mine = self.metadata.structures
        if [s.name for s in structures] != [s.name for s in mine]:
            raise ValueError(
                f"The optics have structures {[s.name for s in structures]}, the reflection table "
                f"{[s.name for s in mine]}"
            )
        for a, b in zip(structures, mine, strict=True):
            offset = max(abs(a.centre[0] - b.centre[0]), abs(a.centre[1] - b.centre[1]))
            if offset > tolerance_px or abs(a.radius - b.radius) > tolerance_px:
                raise ValueError(f"{a.name}: the optics differ from the reflection table's circle")


def packaged_table() -> ReflectionTable:
    """The calibrated table of the GVCCS site camera (Brétigny), embedded in the package."""
    resource = resources.files("skysun") / "tables" / PACKAGED_TABLE
    with resources.as_file(resource) as path:
        return ReflectionTable.from_file(Path(path))
