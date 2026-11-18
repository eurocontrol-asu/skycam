"""Shared fixtures: a tiny synthetic reference archive and a tiny synthetic camera."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pytest

from skycloud.tables import JSON_FILES, REF_FIELDS, SPREAD_FIELDS, ReferenceTables

ENODES = np.array([-4.0, 0.0, 10.0, 30.0, 65.0])
NG, NZ = 6, 5  # angle-to-sun bins of 2 deg, zenith bins of 1.5 deg


def table_value(e: Any, g: Any, z: Any, k: int) -> Any:
    """A linear field per table so that trilinear lookups have a closed form."""
    return 0.01 * k + 0.1 * e + 0.02 * g + 0.003 * z


def thresholds() -> dict[str, Any]:
    return {
        "c4_consistency": {"R0": 0.04, "D0": 0.15},
        "c5_ledger_params": {"T_STEP": 0.1, "R_STEP": 0.03, "N_NOISE": 0.006, "SIG_D": 2.0},
        "c6_midscale": {"elev": [0.0, 10.0], "sd": [0.003, 0.004], "K": [5.0, 6.0]},
        "c7_shape": {"elev": [0.0, 10.0], "R0": [0.3, 0.2], "R1": [0.6, 1.0]},
        "c8_release": {"T_o": 0.53, "T_c": 0.91},
        "c9_knowable": {"E_K": 6.5},
        "c11_lift_colour": {"e": [1.0, 10.0], "clear_p95": [0.4, 0.2], "ov_p5": [0.5, 0.85]},
    }


def synthetic_npz() -> dict[str, Any]:
    e, g, z = np.meshgrid(np.arange(len(ENODES)), np.arange(NG), np.arange(NZ), indexing="ij")
    arrays: dict[str, Any] = {}
    for prefix, names in (("ref", REF_FIELDS), ("spread", SPREAD_FIELDS)):
        for k, name in enumerate(names):
            arrays[f"{prefix}/{name}"] = table_value(e, g, z, k).astype(np.float32)
        arrays[f"{prefix}/enodes"] = ENODES
        arrays[f"{prefix}/GB"] = np.float64(2.0)
        arrays[f"{prefix}/ZB"] = np.float64(1.5)
    arrays["grid/gp"] = np.array([0, 5, 10], np.int64)
    arrays["grid/az"] = np.array([0.0, 90.0, 180.0], np.float32)
    arrays["grid/zen"] = np.array([10.0, 20.0, 30.0], np.float32)
    for prefix, top in (("lowcount", 400.0), ("lowcount_ped", 12.0)):
        arrays[f"{prefix}/S"] = np.array([10.0, 50.0, 100.0, top])
        arrays[f"{prefix}/e"] = np.array([np.nan, 0.03, 0.04, 0.01])
        arrays[f"{prefix}/q_step"] = np.array([np.nan, 1.0, 1.0, 1.0])
        arrays[f"{prefix}/q_none"] = np.array([np.nan, 1.0, 1.0, 1.0])
    meta = {"camera": "synthetic", "thresholds": thresholds()}
    assert set(meta["thresholds"]) == set(JSON_FILES)
    arrays["__metadata__"] = np.array(json.dumps(meta))
    return arrays


def npz_bytes(arrays: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


@pytest.fixture
def tiny_tables() -> ReferenceTables:
    """Synthetic reference tables, built in memory."""
    return ReferenceTables.from_npz(np.load(io.BytesIO(npz_bytes(synthetic_npz()))), "synthetic")


SYN_SIZE, SYN_RADIUS = 261, 95.0


def write_synthetic_archive(root: Any) -> dict[str, Any]:
    """A tiny camera on disk: an equidistant fisheye geometry, its sky mask and two archive
    days of uniform skies (2024-03-10 clear blue, 2024-03-11 grey overcast), a frame every
    15 min from 05:45 to 11:30 UTC at Brétigny's latitude (sun from about -3 to +38 deg)."""
    from pathlib import Path

    from PIL import Image

    root = Path(root)
    n, c = SYN_SIZE, SYN_SIZE // 2
    y, x = np.mgrid[0:n, 0:n].astype(float)
    r = np.hypot(x - c, y - c)
    inside = r <= SYN_RADIUS
    zen = np.where(inside, 90.0 * r / SYN_RADIUS, np.nan)
    az = np.where(inside, np.degrees(np.arctan2(x - c, c - y)) % 360, np.nan)
    np.savez(root / "geo.npz", azimuth_deg=az, zenith_deg=zen)
    Image.fromarray(np.where(inside, 0, 255).astype(np.uint8)).save(root / "static_mask.png")
    stamps = []
    for day, colour in (("20240310", (60, 90, 90)), ("20240311", (120, 120, 120))):
        folder = root / "img" / day[:4] / day[4:6] / day[6:]
        folder.mkdir(parents=True)
        for k in range(24):
            minutes = 5 * 60 + 45 + 15 * k
            ts = f"{day}{minutes // 60:02d}{minutes % 60:02d}00"
            img = np.full((n, n, 3), 10, np.uint8)
            img[inside] = colour
            Image.fromarray(img).save(folder / f"image_{ts}.jp2", format="PNG")
            stamps.append(ts)
    return {"root": root, "stamps": stamps}


SYN_PARAMS = {
    "labels": {"node_min_frames": 2, "node_min_bins": 5},
    "smooth": {"table_min_frames": 2},
    "thresholds": {
        "shape_min_frames": 1, "lift_min_frames": 1, "frame_min_dirs": 50,
        "midscale_frames": 24, "midscale_min_pixels": 100,
    },
}  # fmt: skip
