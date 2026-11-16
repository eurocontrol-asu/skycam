"""End-to-end processing of one day: (acquire) -> decode -> observe -> export -> advect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from skywind.config import Settings
from skywind.days import day_info, parse_windows
from skywind.pipeline import (
    advect_step,
    cruise_emission_points,
    decode_step,
    export_step,
    fitted_field,
    gp_params,
    observe_step,
)

__all__ = ["run_day"]


def _inputs_present(settings: Settings, day: str) -> bool:
    p = settings.paths
    return p.commb(day).is_file() and p.opensky(day).is_file() and p.era5(day).is_file()


def _acquire(settings: Settings, day: str, windows: str) -> dict[str, Any]:
    from skywind.acquire import acquire_day, default_file_sources, register_day
    from skywind.sources.opensky_trino import TrinoCommB

    rep = acquire_day(
        settings,
        day,
        parse_windows(day, windows),
        default_file_sources(settings),
        TrinoCommB(settings.paths.cache / "opensky_query_cache"),
    )
    rep["registered"] = register_day(settings, day, parse_windows(day, windows))
    return rep


def run_day(
    settings: Settings,
    day: str,
    windows: str | None = None,
    params: Path | None = None,
    acquire_missing: bool = True,
    max_age_min: float = 40.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {"day": day}
    if not _inputs_present(settings, day):
        if not (acquire_missing and windows):
            raise FileNotFoundError(f"inputs of {day} missing in {settings.paths.inputs}; give --windows to acquire")
        report["acquire"] = _acquire(settings, day, windows)
    report["decode"] = decode_step(settings, day)
    report["observe"] = observe_step(settings, day)
    info = day_info(settings.paths, day)
    field = fitted_field(settings, day, gp_params(settings, params))
    path = export_step(settings, day, field, info.start, info.stop)
    report["field"] = str(path)
    pts = cruise_emission_points(settings, day, info.start, info.stop)
    until = info.stop + pd.Timedelta(minutes=max_age_min)
    for winds in ("era5", "corrected"):
        adv = advect_step(settings, path, pts, until, winds, max_age_min)
        dest = settings.paths.advected(day, winds)
        dest.parent.mkdir(parents=True, exist_ok=True)
        adv.to_parquet(dest, index=False)
        report[f"advected_{winds}"] = str(dest)
    return report
