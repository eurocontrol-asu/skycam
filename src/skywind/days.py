"""Study days and their time windows (from ``days.csv`` in the input store, else from the Comm-B acquisition logs)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from skywind.config import Paths

__all__ = ["DayInfo", "available_days", "day_info", "parse_day", "parse_windows"]

_DAY = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")
_WINDOW = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-/]\s*(\d{1,2}:\d{2}(?::\d{2})?)")


def parse_day(text: str) -> str:
    """'2024-03-18' or '20240318' -> '20240318'."""
    m = _DAY.match(text.strip())
    if not m:
        raise ValueError(f"not a day: {text!r} (expected YYYYMMDD or YYYY-MM-DD)")
    return "".join(m.groups())


def parse_windows(day: str, text: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """'15:30-17:30;18:00-18:45' (any separator between windows) -> UTC timestamps."""
    d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    out = [(pd.Timestamp(f"{d}T{a}", tz="UTC"), pd.Timestamp(f"{d}T{b}", tz="UTC")) for a, b in _WINDOW.findall(text)]
    if not out:
        raise ValueError(f"no HH:MM-HH:MM window in {text!r}")
    return out


@dataclass(frozen=True)
class DayInfo:
    day: str
    windows: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    set: str = "dev"

    @property
    def start(self) -> pd.Timestamp:
        return min(w[0] for w in self.windows)

    @property
    def stop(self) -> pd.Timestamp:
        return max(w[1] for w in self.windows)


def _from_csv(path: Path) -> dict[str, DayInfo]:
    if not path.is_file():
        return {}
    table = pd.read_csv(path, dtype=str)
    out = {}
    for _, r in table.iterrows():
        day = parse_day(str(r["day"]))
        out[day] = DayInfo(day, tuple(parse_windows(day, str(r["windows"]))), str(r.get("set", "dev") or "dev"))
    return out


def _from_log(paths: Paths, day: str) -> DayInfo | None:
    log = paths.commb_log(day)
    if not log.is_file():
        return None
    meta = json.loads(log.read_text())
    return DayInfo(day, ((pd.Timestamp(meta["start"]), pd.Timestamp(meta["stop"])),))


def day_info(paths: Paths, day: str) -> DayInfo:
    day = parse_day(day)
    info = _from_csv(paths.days_csv()).get(day) or _from_log(paths, day)
    if info is None:
        raise FileNotFoundError(f"no time window for {day}: not in {paths.days_csv()} and no {paths.commb_log(day)}")
    return info


def available_days(paths: Paths) -> list[str]:
    """Days with Comm-B, state vectors and ERA5 in the input store."""
    days = {p.stem.removeprefix("commb_") for p in (paths.inputs / "commb").glob("commb_*.parquet")}
    return sorted(d for d in days if paths.opensky(d).is_file() and paths.era5(d).is_file())
