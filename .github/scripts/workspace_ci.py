"""Discover uv workspace members for CI and run member-specific type checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import TypedDict

__all__ = ["discover", "main"]


class Member(TypedDict):
    """Package metadata consumed by the GitHub Actions matrix."""

    name: str
    path: str


def discover(root: Path) -> list[Member]:
    """Read member globs and exclusions from the root uv configuration."""
    config = tomllib.loads((root / "pyproject.toml").read_text())
    workspace = config["tool"]["uv"]["workspace"]
    excluded = {
        path.resolve()
        for pattern in workspace.get("exclude", [])
        for path in root.glob(pattern)
    }
    paths = {
        path.resolve()
        for pattern in workspace["members"]
        for path in root.glob(pattern)
        if path.resolve() not in excluded and (path / "pyproject.toml").is_file()
    }
    members: list[Member] = []
    for path in sorted(paths):
        package = tomllib.loads((path / "pyproject.toml").read_text())
        members.append(
            {"name": package["project"]["name"], "path": str(path.relative_to(root))}
        )
    if not members:
        raise ValueError("No uv workspace members found")
    return members


def main() -> None:
    """Emit the package matrix, or check each member with its own mypy config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mypy", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    members = discover(root)
    if args.mypy:
        for member in members:
            subprocess.run(  # noqa: S603
                ["uv", "run", "--locked", "--package", member["name"], "mypy", "src"],  # noqa: S607
                cwd=root / member["path"],
                check=True,
            )
        return
    encoded = json.dumps(members)
    print(encoded)  # noqa: T201
    if output := os.getenv("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(f"packages={encoded}\n")


if __name__ == "__main__":
    main()
