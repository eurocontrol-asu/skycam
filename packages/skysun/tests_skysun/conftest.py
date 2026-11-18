"""A tiny synthetic camera: equidistant fisheye, a sun with its glow, one ring of the optics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from skystatic import OpticsRings, RingFit
from skysun import CameraGeometry, Site, sun_position
from skysun.table import ReflectionTable, StructureInfo, TableMetadata, VisibilityRamp

SIZE = 400  # square frames, 0.45 deg per pixel
SITE = Site(latitude_deg=48.6005, longitude_deg=2.3468)
NOON = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)  # sun ~ +64 deg, south
MORNING = datetime(2024, 3, 18, 9, 0, tzinfo=UTC)  # sun ~ +28 deg, south-east
NIGHT = datetime(2024, 1, 15, 2, 0, tzinfo=UTC)
CENTRE = (SIZE - 1) / 2
RING_RADIUS = 75.0


def maps(size: int = SIZE) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Azimuth (north up, east left as seen from below) and zenith (90 deg at radius size/2)."""
    y, x = np.indices((size, size), dtype=np.float32)
    dx, dy = x - (size - 1) / 2, y - (size - 1) / 2
    zenith = np.hypot(dx, dy) / (size / 2) * 90
    azimuth = np.degrees(np.arctan2(-dx, -dy)) % 360
    zenith[zenith > 90] = np.nan
    azimuth[~np.isfinite(zenith)] = np.nan
    return azimuth.astype(np.float32), zenith.astype(np.float32)


@pytest.fixture
def geometry() -> CameraGeometry:
    return CameraGeometry(*maps())


def frame(
    geometry: CameraGeometry, time: datetime, *, sun_visible: bool = True
) -> NDArray[np.uint8]:
    """Sky at 80, a glow 170 x exp(-angle / 5 deg) with a saturated disc of 2 deg when the
    sun shines; a ring 6 % brighter than the sky on the sun's side; black outside."""
    sun = sun_position(time, SITE)
    angle = geometry.angle_to(sun)
    lum = np.full(geometry.shape, 80.0, dtype=np.float32)
    if sun.elevation_deg > 0 and sun_visible:
        lum += 170.0 * np.exp(-angle / 5.0)
        lum[angle < 2.0] = 255.0
        y, x = np.indices(geometry.shape)
        sx, sy = geometry.sun_pixel(angle)
        r = np.hypot(x - CENTRE, y - CENTRE)
        facing = np.cos(np.arctan2(y - CENTRE, x - CENTRE) - np.arctan2(sy - CENTRE, sx - CENTRE))
        lum[(np.abs(r - RING_RADIUS) <= 1.5) & (facing > 0)] *= 1.06
    lum[~geometry.inside] = 0.0
    return np.repeat(np.clip(lum, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def optics(size: int = SIZE) -> OpticsRings:
    fit = RingFit(
        centre=(CENTRE, CENTRE),
        noise=1e-4,
        rings=[(RING_RADIUS, RING_RADIUS, 0.004)],
        eye=0.0,
        centres=[(CENTRE, CENTRE)],
    )
    return OpticsRings(size_wh=(size, size), halfwidth=1.5, fit=fit)


REFERENCE = 0.08  # the synthetic optic's reference contrast


def table(contrast: float = 0.04, *, high: float = 0.6) -> ReflectionTable:
    """One ring; ``contrast`` on the half of its sectors facing the sun, 0 elsewhere."""
    sectors = 36
    relative = (np.arange(sectors) + 0.5) * 10
    row = np.where(np.cos(np.radians(relative)) > 0, contrast, 0.0).astype(np.float32)
    edges = [0.0, 20.0, 40.0, 70.0]
    values = np.broadcast_to(row, (1, len(edges) - 1, sectors)).copy()
    metadata = TableMetadata(
        camera_id="synthetic",
        image_size_wh=(SIZE, SIZE),
        structures=[
            StructureInfo(name="ring0", centre=(CENTRE, CENTRE), radius=RING_RADIUS, disc=False)
        ],
        elevation_edges=edges,
        sectors=sectors,
        saturation=250,
        core_deg=2.0,
        visibility=VisibilityRamp(
            elevation_edges=edges,
            high=[high] * (len(edges) - 1),
            glow_low=0.10,
            glow_high=0.25,
            origin="synthetic",
        ),
        reference_contrast=REFERENCE,
        negligible_contrast=0.003,
        frames=12,
        days=["20240101"],
        period=("20240101100000", "20240101120000"),
        selection="synthetic",
        command="tests",
    )
    return ReflectionTable(
        metadata, values, np.full((1, len(edges) - 1), 12, dtype=np.int32)
    ).sealed()


@pytest.fixture
def table_file(tmp_path: Path) -> Path:
    path = tmp_path / "table.npz"
    table().save(path)
    return path
