"""The table build from per-day measurement files: ramp, relative sectors, elevation fill."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skysun import ReflectionTable
from skysun.calibrate import BuildConfig, build_table
from skysun.table import StructureInfo

pytestmark = pytest.mark.integration

RING = StructureInfo(name="ring0", centre=(100.0, 100.0), radius=50.0, disc=False)


def write_day(
    folder: Path, day: str, elevations: list[float], labels: list[int], cores: list[float]
) -> None:
    """Frames whose sun lies right of the ring centre (image angle 0); sector 0-10 lit 10 %
    above flanks of 100 when the sun shines (core >= 0.5), nothing lit otherwise."""
    n = len(elevations)
    counts = np.zeros((n, 90), dtype=np.int64)
    counts[:, :4] = 100
    saturated_counts = np.zeros((n, 90), dtype=np.int64)
    saturated_counts[:, :4] = (np.array(cores)[:, None] * 100).astype(np.int64)
    flank = np.full((n, 1, 36), 100.0)
    band = flank.copy()
    shine = np.array(cores) >= 0.5
    band[shine, 0, 0] = 110.0
    np.savez(
        folder / f"{day}.npz",
        structures=np.array(json.dumps([RING.model_dump(mode="json")])),
        stamp=np.array([f"{day}{100000 + 1000 * i:06d}" for i in range(n)]),
        label=np.array(labels, dtype=np.int8),
        elevation=np.array(elevations),
        azimuth=np.full(n, 180.0),
        sun_xy=np.tile([300.0, 100.0], (n, 1)),
        # a glow of 170 exp(-a / 5 deg) above a sky of 80 where the sun shines
        level=80.0 + 170.0 * np.exp(-((np.arange(90) + 0.5) * 0.5) / 5.0)[None, :] * shine[:, None],
        saturated=saturated_counts / np.maximum(counts, 1),
        counts=counts,
        saturated_counts=saturated_counts,
        sky_level=np.full(n, 80.0),
        core_saturated=np.array(cores),
        core_sky=np.ones(n),
        band=band,
        flank=flank,
        usable=np.ones((n, 1, 36), dtype=bool),
    )


def test_build_reads_the_ramp_and_the_lit_sector(tmp_path: Path) -> None:
    folder = tmp_path / "measure"
    folder.mkdir()
    # 40 clear frames at 15 deg (cores 0.6..0.98), 5 overcast frames with a hidden sun
    cores = list(np.linspace(0.6, 0.98, 40))
    write_day(folder, "20240601", [15.0] * 40, [1] * 40, cores)
    write_day(folder, "20240602", [15.0] * 5, [2] * 5, [0.0] * 5)
    config = BuildConfig(camera_id="test", min_frames=10, ramp_min_frames=30)
    table = build_table(folder, config, image_size=(200, 200))
    meta = table.metadata
    top = float(np.quantile(cores, 0.05))
    assert meta.visibility.high == pytest.approx([top] * 13)  # every bin from the 15-deg one
    assert meta.days == ["20240601"]
    assert meta.frames == sum(c >= top for c in cores)  # the clear frames at or above the top
    e = int(np.digitize(15.0, meta.elevation_edges) - 1)
    assert table.contrast[0, e, 0] == pytest.approx(0.10)
    assert table.contrast[0, e, 1:].max() == 0.0
    assert (table.contrast[0, 0] == table.contrast[0, e]).all()  # empty bins: nearest filled
    assert meta.reference_contrast == pytest.approx(0.10)  # p99 of 0.10 and zeros
    assert meta.visibility.glow_low == 0.0  # hidden: no glow
    # mean of exp(-a/5) over 5-10 deg = e^-1 - e^-2
    assert meta.visibility.glow_high == pytest.approx(np.exp(-1) - np.exp(-2), abs=0.01)
    table.save(tmp_path / "table.npz")
    assert ReflectionTable.from_file(tmp_path / "table.npz").metadata.sha256 == meta.sha256
