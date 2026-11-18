from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tests_skystatic.conftest import write_day

pytestmark = pytest.mark.e2e


def skystatic(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "skystatic.cli", *args], capture_output=True, text=True, check=False
    )


def test_run_writes_masks_and_report(tmp_path: Path, zenith_file: Path) -> None:
    folders = [
        write_day(tmp_path / f"2024-01-0{i}", datetime(2024, 1, i, 9), dirty=i == 2)
        for i in (1, 2, 3)
    ]
    result = skystatic(
        "run",
        *map(str, folders),
        "--zenith",
        str(zenith_file),
        "--output",
        str(tmp_path / "out"),
        "--max-side",
        "0",
        "--min-days",
        "0.6",
    )
    assert result.returncode == 0, result.stderr
    assert "installation:" in result.stdout
    assert "2024-01-02: " in result.stdout
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [d["label"] for d in report["days"]] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert report["days"][1]["variable_components"] >= 1
    assert (tmp_path / "out" / "installation.png").exists()


def test_freeze_and_reuse_from_cli(tmp_path: Path, zenith_file: Path) -> None:
    reference = write_day(tmp_path / "raw/reference", datetime(2024, 1, 1, 9))
    first = skystatic(
        "run",
        str(reference),
        "--zenith",
        str(zenith_file),
        "--output",
        str(tmp_path / "baseline"),
        "--max-side",
        "0",
    )
    assert first.returncode == 0, first.stderr
    frozen = skystatic(
        "freeze",
        str(tmp_path / "baseline"),
        str(tmp_path / "camera-v1.npz"),
        "--camera-id",
        "test-camera",
        "--version",
        "v1",
        "--valid-from",
        "2024-01-01",
        "--valid-until",
        "2025-01-01",
    )
    assert frozen.returncode == 0, frozen.stderr
    day = write_day(tmp_path / "raw/day", datetime(2024, 5, 10, 9), dirty=True)
    reused = skystatic(
        "run",
        str(day),
        "--zenith",
        str(zenith_file),
        "--output",
        str(tmp_path / "daily"),
        "--max-side",
        "0",
        "--installation",
        str(tmp_path / "camera-v1.npz"),
    )
    assert reused.returncode == 0, reused.stderr
    report = json.loads((tmp_path / "daily/report.json").read_text())
    assert report["installation_id"].startswith("test-camera/v1/")
    assert report["reference_days"] == ["2024-01-01"]
    assert report["days"][0]["variable_components"] >= 1


def test_missing_input_fails_with_a_message(tmp_path: Path, zenith_file: Path) -> None:
    result = skystatic(
        "run", str(tmp_path / "nowhere"), "--zenith", str(zenith_file), "--output", str(tmp_path)
    )
    assert result.returncode == 1
    assert "skystatic:" in result.stderr


def test_reference_takes_several_days_and_they_are_not_processed(
    tmp_path: Path, zenith_file: Path
) -> None:
    day = write_day(tmp_path / "2024-01-01", datetime(2024, 1, 1, 9), dirty=True)
    references = [write_day(tmp_path / f"2024-06-0{i}", datetime(2024, 6, i, 9)) for i in (1, 2, 3)]
    result = skystatic(
        "run",
        str(day),
        "--reference",
        *map(str, references),
        "--zenith",
        str(zenith_file),
        "--output",
        str(tmp_path / "out"),
        "--max-side",
        "0",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [d["label"] for d in report["days"]] == ["2024-01-01"]
    assert report["reference_days"] == ["2024-06-01", "2024-06-02", "2024-06-03"]
    assert report["days"][0]["variable_components"] == 1
