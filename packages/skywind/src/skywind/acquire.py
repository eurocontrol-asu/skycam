"""Acquisition of a new day into the durable input store: state vectors and ERA5 (file sources), then the raw Comm-B
replies of the aircraft at cruise near the site (Comm-B source), and its time windows in ``days.csv``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from skywind.config import Settings
from skywind.sources.base import CommBSource, FileSource, SourceError
from skywind.sources.remote import RemoteCopySource
from skywind.statevectors import load_state_vectors

__all__ = ["acquire_day", "cruise_aircraft", "default_file_sources", "register_day"]


def default_file_sources(settings: Settings) -> tuple[FileSource, FileSource]:
    """State-vector and ERA5 sources from ``[sources]`` of the configuration."""
    src = settings.sources
    if not (src.remote_host and src.opensky_pattern and src.era5_pattern):
        raise SourceError("configure [sources] remote_host, opensky_pattern and era5_pattern (skywind.toml)")
    return RemoteCopySource(src.remote_host, src.opensky_pattern), RemoteCopySource(src.remote_host, src.era5_pattern)


def cruise_aircraft(
    sv_path: Path, settings: Settings, start: pd.Timestamp, stop: pd.Timestamp, min_samples: int = 12
) -> list[str]:
    """Aircraft with at least ``min_samples`` state vectors at cruise inside the site box in [start, stop]."""
    sv = load_state_vectors(sv_path, ["flight_id", "timestamp", "icao24", "latitude", "longitude", "altitude"])
    site = settings.site
    sel = (
        sv["timestamp"].between(start, stop)
        & ((sv["latitude"] - site.lat).abs() <= site.box_deg)
        & ((sv["longitude"] - site.lon).abs() <= site.box_deg)
        & (sv["altitude"] >= site.min_alt_ft)
    )
    n = sv[sel].groupby("icao24").size()
    return sorted(str(a) for a in n[n >= min_samples].index)


def _fetch_file(source: FileSource, day: str, dest: Path, overwrite: bool) -> str:
    if dest.is_file() and not overwrite:
        return "present"
    source.fetch(day, dest)
    return "fetched"


def register_day(
    settings: Settings, day: str, windows: list[tuple[pd.Timestamp, pd.Timestamp]], set_name: str = "new"
) -> bool:
    """Append the day to ``days.csv`` (keeping its existing columns) unless already listed. True if a row was added."""
    path = settings.paths.days_csv()
    row = {
        "day": day,
        "commb_start": min(w[0] for w in windows).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commb_stop": max(w[1] for w in windows).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "set": set_name,
        "notes": "registered by skywind acquire",
    }
    columns = list(row)
    if path.is_file():
        table = pd.read_csv(path, dtype=str)
        if (table["day"].str.replace("-", "") == day).any():
            return False
        columns = list(table.columns)
    pd.DataFrame([row]).reindex(columns=columns).to_csv(path, mode="a", header=not path.is_file(), index=False)
    return True


def acquire_day(
    settings: Settings,
    day: str,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    files: tuple[FileSource, FileSource],
    commb: CommBSource,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch everything needed for ``day``; Comm-B covers the windows padded by ``commb_padding_min`` before."""
    paths = settings.paths
    report: dict[str, Any] = {"day": day}
    report["state_vectors"] = _fetch_file(files[0], day, paths.opensky(day), overwrite)
    report["era5"] = _fetch_file(files[1], day, paths.era5(day), overwrite)
    if paths.commb(day).is_file() and not overwrite:
        report["commb"] = "present"
        return report
    start = min(w[0] for w in windows) - pd.Timedelta(minutes=settings.sources.commb_padding_min)
    stop = max(w[1] for w in windows)
    icao = cruise_aircraft(paths.opensky(day), settings, start, stop)
    raw, log = commb.fetch(start, stop, icao)
    paths.commb(day).parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(paths.commb(day), index=False)
    log.update(
        day=day,
        start=str(start),
        stop=str(stop),
        box=settings.site.box_deg,
        n_aircraft=len(icao),
        rows=len(raw),
        aircraft_with_commb=int(raw["icao24"].nunique()),
    )
    paths.commb_log(day).write_text(json.dumps(log, indent=2))
    report.update(commb="fetched", rows=len(raw), aircraft=len(icao))
    return report
