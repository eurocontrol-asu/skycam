"""Static parts of a fixed all-sky camera, from averaged images."""

from __future__ import annotations

from skystatic.frames import DirectorySource, Frame, FrameSource, ZipSource, open_source
from skystatic.geometry import SkyGeometry
from skystatic.pipeline import DayResult, RunReport, StaticConfig, compute_day, run

__all__ = [
    "DayResult",
    "DirectorySource",
    "Frame",
    "FrameSource",
    "RunReport",
    "SkyGeometry",
    "StaticConfig",
    "ZipSource",
    "compute_day",
    "open_source",
    "run",
]
