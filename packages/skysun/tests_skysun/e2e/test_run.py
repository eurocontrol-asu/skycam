"""``skysun run`` end to end on a tiny synthetic camera."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from skystatic import SkyGeometry, StaticConfig, open_source, run
from skysun import CameraGeometry
from tests_skysun.conftest import MORNING, frame, maps, optics

pytestmark = pytest.mark.e2e


def skysun(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "skysun.cli", *args], capture_output=True, text=True, check=False
    )


def test_run_writes_layers_and_json_and_night_is_zero(
    tmp_path: Path, geometry: CameraGeometry, table_file: Path
) -> None:
    day = tmp_path / "raw"
    day.mkdir()
    for i in range(12):
        time = MORNING + timedelta(minutes=20 * i)
        Image.fromarray(frame(geometry, time)).save(day / f"image_{time:%Y%m%d%H%M%S}.png")
    run(
        [open_source(day)],
        SkyGeometry.from_array(maps()[1]),
        StaticConfig(max_side=200),
        tmp_path / "static",
    )
    (tmp_path / "static" / "optics.json").write_text(optics().model_dump_json())
    night = tmp_path / "night"
    night.mkdir()
    Image.fromarray(np.full((*geometry.shape, 3), 90, np.uint8)).save(
        night / f"image_{MORNING:%Y%m%d}020000.png"  # same date, sun at -30 deg
    )
    azimuth, zenith = maps()
    np.savez(tmp_path / "geometry.npz", azimuth_deg=azimuth, zenith_deg=zenith)
    stamp = f"{MORNING:%Y%m%d%H%M%S}"
    common = [
        "--geometry",
        str(tmp_path / "geometry.npz"),
        "--static",
        str(tmp_path / "static"),
        "--latitude",
        "48.6005",
        "--longitude",
        "2.3468",
        "--table",
        str(table_file),
        "--output",
        str(tmp_path / "out"),
    ]
    result = skysun("run", str(day / f"image_{stamp}.png"), *common)
    assert result.returncode == 0, result.stderr
    assert "1 frames, 1 with the sun above the horizon" in result.stdout
    info = json.loads((tmp_path / "out/2024-03-18" / f"sun_{stamp}.json").read_text())
    assert info["visibility"] == 1.0
    assert 20 < info["sun_elevation_deg"] < 35
    code = np.asarray(Image.open(tmp_path / "out/2024-03-18" / f"zone_{stamp}.png"))
    assert code.dtype == np.uint8 and code.max() == 255
    assert not (tmp_path / "out/2024-03-18" / f"sun_{stamp}.png").exists()  # no combined map
    dark = skysun("run", str(night), *common)
    assert dark.returncode == 0, dark.stderr
    for layer in ("zone", "reflections"):
        code = np.asarray(Image.open(tmp_path / "out/2024-03-18" / f"{layer}_20240318020000.png"))
        assert code.shape == geometry.shape and (code == 0).all()


def test_run_reports_a_missing_geometry(tmp_path: Path) -> None:
    result = skysun(
        "run",
        str(tmp_path),
        "--geometry",
        str(tmp_path / "none.npz"),
        "--static",
        str(tmp_path),
        "--latitude",
        "48.6",
        "--longitude",
        "2.3",
        "--output",
        str(tmp_path / "o"),
    )
    assert result.returncode == 1
    assert result.stderr.startswith("skysun:")
