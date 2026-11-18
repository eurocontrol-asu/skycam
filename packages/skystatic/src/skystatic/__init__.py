"""Static parts of a fixed all-sky camera, from averaged images."""

from __future__ import annotations

from skystatic.frames import DirectorySource, Frame, FrameSource, ZipSource, open_source
from skystatic.geometry import SkyGeometry
from skystatic.installation import FrozenInstallation, freeze_installation
from skystatic.optics import RingFit
from skystatic.pipeline import DayResult, RunReport, StaticConfig, compute_day, run
from skystatic.products import (
    DayMask,
    MaskMetadata,
    OpticsRings,
    RingCircle,
    load_day_mask,
    load_optics,
    sky_mask,
)

__all__ = [
    "DayMask",
    "DayResult",
    "DirectorySource",
    "Frame",
    "FrameSource",
    "FrozenInstallation",
    "MaskMetadata",
    "OpticsRings",
    "RingCircle",
    "RingFit",
    "RunReport",
    "SkyGeometry",
    "StaticConfig",
    "ZipSource",
    "compute_day",
    "freeze_installation",
    "load_day_mask",
    "load_optics",
    "open_source",
    "run",
    "sky_mask",
]
