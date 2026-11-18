"""Building the reference tables of a camera from an archive of its frames.

``build_reference`` runs the steps of the c16 build (per-frame measurements, bin means,
label-free clear / overcast labels, raw tables, smoothing, spreads, post-fit, threshold tables)
in a resumable work folder and packs the result in the format ``skycloud.tables`` loads.
"""

from __future__ import annotations

from skycloud.reference_build.frames import ArchiveFrames, select_frames
from skycloud.reference_build.params import BuildParams
from skycloud.reference_build.pipeline import (
    STEPS,
    BuildInputs,
    BuildReport,
    CarriedTables,
    build_reference,
)

__all__ = [
    "STEPS",
    "ArchiveFrames",
    "BuildInputs",
    "BuildParams",
    "BuildReport",
    "CarriedTables",
    "build_reference",
    "select_frames",
]
