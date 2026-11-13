"""Source interfaces. A source fetches one day of one kind of input into the durable input store."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

__all__ = ["CommBSource", "FileSource", "SourceError"]


class SourceError(RuntimeError):
    """A source could not deliver (missing file, rejected credentials...)."""


class FileSource(Protocol):
    """Copies the file of ``day`` (YYYYMMDD) to ``dest``."""

    def fetch(self, day: str, dest: Path) -> Path: ...


class CommBSource(Protocol):
    """Raw Comm-B (DF20/21) replies of the given aircraft in [start, stop): columns mintime, rawmsg, icao24, serials."""

    def fetch(self, start: pd.Timestamp, stop: pd.Timestamp, icao24: list[str]) -> tuple[pd.DataFrame, dict[str, object]]: ...
