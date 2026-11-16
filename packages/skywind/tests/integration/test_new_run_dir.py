from __future__ import annotations

from pathlib import Path

import pytest

from skywind.runs import new_run_dir

pytestmark = pytest.mark.integration


def test_run_dir_readme_first_line(tmp_path: Path) -> None:
    a = new_run_dir(tmp_path, "eval", "a test", "scratch")
    b = new_run_dir(tmp_path, "eval", "another", "reference")
    assert a != b
    assert (a / "README.md").read_text().splitlines()[0] == "status: scratch — a test"
    assert (b / "README.md").read_text().startswith("status: reference — another")
