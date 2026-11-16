from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e
EXE = [sys.executable, "-m", "skywind.cli"]


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*EXE, *args], capture_output=True, text=True, timeout=900, check=False, env={**os.environ, **(env or {})}
    )


def test_help_lists_commands() -> None:
    res = _run("--help")
    assert res.returncode == 0
    for cmd in ("acquire", "decode", "observe", "calibrate", "fit", "export", "advect", "run", "evaluate"):
        assert cmd in res.stdout


@pytest.mark.requires_store
def test_days_lists_store_days() -> None:
    res = _run("days")
    if res.returncode != 0 or res.stdout.strip() == "[]":
        pytest.skip("input store not reachable")
    assert "20240206" in {d["day"] for d in json.loads(res.stdout)}


def test_bad_day_fails_cleanly() -> None:
    res = _run("export", "not-a-day")
    assert res.returncode != 0


@pytest.mark.requires_store
def test_decode_then_observe(tmp_path: Path) -> None:
    env = {"SKYWIND_DATA": str(tmp_path)}
    if _run("days").stdout.strip() in ("", "[]"):
        pytest.skip("input store not reachable")
    assert _run("decode", "20240206", env=env).returncode == 0
    res = _run("observe", "20240206", env=env)
    assert res.returncode == 0
    assert json.loads(res.stdout)[0]["observations"] == 6946
