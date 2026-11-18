from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests_skycloud.conftest import SYN_PARAMS, npz_bytes, synthetic_npz, write_synthetic_archive

pytestmark = pytest.mark.e2e


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    code = "from skycloud.cli import main; main()"
    cmd = [sys.executable, "-c", code, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603


def test_build_reference_writes_an_archive_the_renderer_loads(tmp_path: Path) -> None:
    write_synthetic_archive(tmp_path)
    (tmp_path / "params.json").write_text(json.dumps(SYN_PARAMS))
    carry = tmp_path / "carry.npz"
    carry.write_bytes(npz_bytes(synthetic_npz()))
    (tmp_path / "eval_days.txt").write_text("20991231\n")
    common = [
        "build-reference", "--out", str(tmp_path / "ref.npz"), "--work", str(tmp_path / "work"),
        "--camera", "synthetic", "--images", str(tmp_path / "img"),
        "--sky-mask", str(tmp_path / "static_mask.png"),
        "--azimuth", str(tmp_path / "geo.npz"), "--zenith", str(tmp_path / "geo.npz"),
        "--latitude", "48.6", "--longitude", "2.35", "--params", str(tmp_path / "params.json"),
        "--exclude-days", str(tmp_path / "eval_days.txt"),
    ]  # fmt: skip
    r = _run(*common)
    assert r.returncode == 1 and "not measured by the build" in r.stderr
    sha = hashlib.sha256(carry.read_bytes()).hexdigest()
    r = _run(*common, "--carry", str(carry), "--carry-sha256", sha)
    assert r.returncode == 0, r.stderr
    digest = r.stdout.split("SHA-256 ")[1].strip()
    from skycloud.tables import ReferenceTables

    t = ReferenceTables.load(tmp_path / "ref.npz", sha256=digest)
    assert t.metadata["archive"]["frames"] == 48 and "step labels: running" in r.stderr
