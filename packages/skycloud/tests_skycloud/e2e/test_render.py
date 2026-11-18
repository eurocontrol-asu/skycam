from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.e2e


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    code = "from skycloud.cli import main; main()"
    cmd = [sys.executable, "-c", code, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603


def test_help_lists_the_commands() -> None:
    r = _run("--help")
    assert r.returncode == 0 and "render" in r.stdout and "ledger" in r.stdout


def test_missing_static_mask_is_an_error_not_a_fallback(tmp_path: Path) -> None:
    size = 64
    y, x = np.indices((size, size), dtype=np.float64)
    zen = np.hypot(x - 31.5, y - 31.5) / 32 * 90
    zen[zen > 90] = np.nan
    np.savez(
        tmp_path / "geo.npz",
        zenith_deg=zen,
        azimuth_deg=np.degrees(np.arctan2(x - 31.5, 31.5 - y)) % 360,
    )
    (tmp_path / "masks").mkdir()
    day = tmp_path / "img" / "2024" / "01" / "01"
    day.mkdir(parents=True)
    from PIL import Image

    Image.fromarray(np.full((size, size, 3), 120, np.uint8)).save(
        day / "image_20240101120000.jp2", format="JPEG"
    )
    r = _run(
        "render", "20240101120000", "--out", str(tmp_path / "out"),
        "--images", str(tmp_path / "img"),
        "--mask-dir", str(tmp_path / "masks"), "--ledger-dir", str(tmp_path / "ledgers"),
        "--azimuth", str(tmp_path / "geo.npz"), "--zenith", str(tmp_path / "geo.npz"),
        "--latitude", "48.6", "--longitude", "2.35",
    )  # fmt: skip
    assert r.returncode == 0  # the frame fails, the run reports it
    assert "1 errors" in r.stdout
    frames = (tmp_path / "out" / "frames.json").read_text()
    assert "No daily mask" in frames
