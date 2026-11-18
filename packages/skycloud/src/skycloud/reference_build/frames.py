"""The archive frames of a build: from a folder or from a list, grouped by day.

A frame is named by its UTC timestamp; its file is ``image_<YYYYmmddHHMMSS>.jp2`` (the content
is decoded whatever the extension). The fetch from a remote archive is not part of the library:
the build reads frames already on disk.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = ["ArchiveFrames", "select_frames"]


@dataclass(frozen=True)
class ArchiveFrames:
    """The frames of a build, per day (``YYYYmmdd``), each day's in time order; ``listed``:
    the same frames in the order they were given (the random draws of a build read it)."""

    days: dict[str, list[tuple[str, Path]]]
    listed: list[tuple[str, Path]]

    @property
    def n_frames(self) -> int:
        return sum(len(v) for v in self.days.values())

    @property
    def stamps(self) -> list[str]:
        return [ts for d in sorted(self.days) for ts, _ in self.days[d]]


def _stamp(path: Path) -> str:
    name = path.stem
    if not name.startswith("image_") or len(name) < 20 or not name[6:20].isdigit():
        raise ValueError(f"{path}: not a frame name image_<YYYYmmddHHMMSS>")
    return name[6:20]


def _paths(images: Path | None, list_file: Path | None) -> list[Path]:
    if list_file is not None:
        out = []
        for line in Path(list_file).read_text().split():
            p = Path(line)
            if p.is_absolute():
                out.append(p)
            elif images is not None and line.isdigit() and len(line) == 14:
                out.append(images / line[:4] / line[4:6] / line[6:8] / f"image_{line}.jp2")
            elif images is not None:
                out.append(images / p)
            else:
                raise ValueError(f"{list_file}: '{line}' is relative and no image folder is given")
        return out
    if images is None:
        raise ValueError("give an image folder or a frame list")
    return sorted(Path(images).rglob("image_*.jp2"))


def select_frames(
    *,
    images: Path | None = None,
    list_file: Path | None = None,
    exclude_days: Iterable[str] = (),
    duplicates: Literal["error", "keep"] = "error",
) -> ArchiveFrames:
    """The frames of ``list_file`` (absolute paths, paths relative to ``images`` or
    timestamps) or every ``image_*.jp2`` under ``images``; days of ``exclude_days`` (the
    evaluation days) are dropped. A timestamp met twice is refused, or kept twice with
    ``duplicates='keep'`` (the c16 build)."""
    excluded = frozenset(exclude_days)
    days: dict[str, list[tuple[str, Path]]] = {}
    listed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for p in _paths(images, list_file):
        ts = _stamp(p)
        if ts[:8] in excluded:
            continue
        if ts in seen and duplicates == "error":
            raise ValueError(f"frame {ts} is listed twice (e.g. {p}); use duplicates='keep'")
        seen.add(ts)
        days.setdefault(ts[:8], []).append((ts, p))
        listed.append((ts, p))
    if not days:
        raise ValueError("no frame selected")
    # time order; equal stamps (duplicates kept) by path, as the c16 build sorted them
    ordered = {d: sorted(v, key=lambda x: (x[0], str(x[1]))) for d, v in sorted(days.items())}
    return ArchiveFrames(ordered, listed)
