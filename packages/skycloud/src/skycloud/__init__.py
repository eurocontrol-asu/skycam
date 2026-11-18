"""Continuous cloud map of a fixed all-sky camera: value, confidence and an unknown state."""

from __future__ import annotations

from skycloud.engine import FrameInputs, FrameResult, render_frame
from skycloud.geometry import CameraGeometry, Site
from skycloud.ledger_build import build_ledgers
from skycloud.params import CloudMapParams
from skycloud.products import FrameProducts, load_products
from skycloud.render import RenderSetup, RunReport, render
from skycloud.sun import sun_position
from skycloud.tables import ReferenceTables, pack_reference_tables

__all__ = [
    "CameraGeometry",
    "CloudMapParams",
    "FrameInputs",
    "FrameProducts",
    "FrameResult",
    "ReferenceTables",
    "RenderSetup",
    "RunReport",
    "Site",
    "build_ledgers",
    "load_products",
    "pack_reference_tables",
    "render",
    "render_frame",
    "sun_position",
]
