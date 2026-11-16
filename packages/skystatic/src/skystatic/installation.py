"""Explicitly frozen installation products, versioned by camera and validity period."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from skystatic.products import read_classes

__all__ = ["FrozenInstallation", "freeze_installation"]


class InstallationMetadata(BaseModel):
    """Provenance and inclusive/exclusive calendar validity of a frozen installation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    camera_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    valid_from: date
    valid_until: date
    geometry_sha256: str
    parameters: dict[str, JsonValue]
    reference_days: list[str]
    rejected_reference_days: list[str]
    sha256: str

    @model_validator(mode="after")
    def ordered_period(self) -> Self:
        """A frozen installation must have a nonempty validity period."""
        if self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from (exclusive end)")
        if not self.reference_days:
            raise ValueError("A frozen installation requires reference-day provenance")
        return self

    @property
    def artifact_id(self) -> str:
        """Camera, human version and content identifier."""
        return f"{self.camera_id}/{self.version}/{self.sha256}"


def installation_digest(
    metadata: InstallationMetadata, classes: NDArray[np.uint8], votes: NDArray[np.uint8]
) -> str:
    """Hash provenance and both pixel arrays to detect changed frozen products."""
    data = metadata.model_dump(mode="json", exclude={"sha256"})
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    digest.update(str(classes.shape).encode())
    digest.update(classes.tobytes())
    digest.update(votes.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenInstallation:
    """Read-only installation classes and original reference votes."""

    metadata: InstallationMetadata
    classes: NDArray[np.uint8]
    votes: NDArray[np.uint8]

    @classmethod
    def from_file(cls, path: Path) -> FrozenInstallation:
        """Load a portable NPZ without pickle and verify its content hash."""
        with np.load(path, allow_pickle=False) as stored:
            metadata = InstallationMetadata.model_validate_json(str(stored["metadata"].item()))
            classes = stored["classes"].copy()
            votes = stored["votes"].copy()
        if (
            classes.dtype != np.uint8
            or votes.dtype != np.uint8
            or classes.ndim != 2
            or votes.shape != classes.shape
            or not np.isin(classes, [0, 1, 3]).all()
        ):
            raise ValueError("Invalid frozen installation arrays")
        if installation_digest(metadata, classes, votes) != metadata.sha256:
            raise ValueError("Frozen installation checksum mismatch")
        classes.flags.writeable = False
        votes.flags.writeable = False
        return cls(metadata, classes, votes)


def freeze_installation(
    output: Path,
    destination: Path,
    *,
    camera_id: str,
    version: str,
    valid_from: date,
    valid_until: date,
) -> FrozenInstallation:
    """Freeze an existing run without recomputing it; never overwrite a destination.

    The validity interval is [valid_from, valid_until). Reference days, parameters,
    geometry identity, classes and votes travel together in one checksummed NPZ.
    """
    # Imported at call time to keep product readers independent of the pipeline.
    from skystatic.pipeline import RunReport

    report = RunReport.model_validate_json((output / "report.json").read_text())
    if report.installation_id is not None:
        raise ValueError("This run already uses a frozen installation; reuse its original artifact")
    classes = read_classes(output / "installation.png")
    # Vote images contain counts, not class labels.

    with Image.open(output / "reference_votes.png") as image:
        votes = np.asarray(image, dtype=np.uint8).copy()
    if classes.shape != votes.shape or not np.isin(classes, [0, 1, 3]).all():
        raise ValueError("Invalid installation or reference-vote image")
    metadata = InstallationMetadata(
        camera_id=camera_id,
        version=version,
        valid_from=valid_from,
        valid_until=valid_until,
        geometry_sha256=report.geometry_sha256,
        parameters=report.config.model_dump(mode="json"),
        reference_days=report.reference_days,
        rejected_reference_days=report.rejected_reference_days,
        sha256="",
    )
    metadata = metadata.model_copy(update={"sha256": installation_digest(metadata, classes, votes)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary, metadata=np.array(metadata.model_dump_json()), classes=classes, votes=votes
        )
        os.link(temporary, destination)  # atomic and exclusive, also between concurrent writers
    finally:
        temporary.unlink(missing_ok=True)
    return FrozenInstallation.from_file(destination)
