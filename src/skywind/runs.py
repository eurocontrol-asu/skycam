"""Run directories: ``<runs>/<YYYYMMDD>-<slug>/`` with a README whose first line states the status and purpose."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

__all__ = ["new_run_dir", "write_json"]

type Status = Literal["reference", "archived", "scratch"]


def new_run_dir(root: Path, slug: str, purpose: str, status: Status = "scratch", details: str = "") -> Path:
    """Create ``root/<today>-<slug>`` (suffixing -2, -3... if it exists) and its README.md."""
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    base = root / f"{stamp}-{slug}"
    path, k = base, 2
    while path.exists():
        path, k = Path(f"{base}-{k}"), k + 1
    path.mkdir(parents=True)
    text = f"status: {status} — {purpose}\n\ncreated: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
    (path / "README.md").write_text(text + (f"\n{details}\n" if details else ""))
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
