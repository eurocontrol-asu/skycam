"""Validation of the raw-GVCCS truth source against the prototype's calibration artefacts (same days, same flights):
per-flight drift and offset of the ERA5 advection, and the distance by age, from both sources."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from skywind.advect import advect, met_dataset
from skywind.config import Settings
from skywind.era5 import load_era5
from skywind.evaluate.contrails import TruthSource, age_table, cross_track_errors, flight_fits

__all__ = ["compare_truth_sources"]

log = logging.getLogger(__name__)


def _era5_errors(settings: Settings, source: TruthSource, day: str) -> pd.DataFrame:
    truth = source.load(day)
    adv = advect(truth.emissions, met_dataset(load_era5(settings.paths.era5(day))), truth.last_time, max_age_min=45)
    return cross_track_errors(truth, adv, settings.site)


def compare_truth_sources(
    settings: Settings,
    a: TruthSource,
    b: TruthSource,
    days: Sequence[str],
    out: Path,
    names: tuple[str, str] = ("raw", "calib"),
) -> pd.DataFrame:
    """ERA5 drift / offset per flight from two truth sources, paired by flight; writes CSVs, returns the pairs."""
    errs: dict[str, list[pd.DataFrame]] = {names[0]: [], names[1]: []}
    for day in days:
        log.info("truth validation, day %s", day)
        for name, src in zip(names, (a, b), strict=True):
            errs[name].append(_era5_errors(settings, src, day))
    e = {k: pd.concat(v, ignore_index=True) for k, v in errs.items()}
    fits = {k: flight_fits(v) for k, v in e.items()}
    pairs = fits[names[0]].merge(fits[names[1]], on=["day", "flight"], suffixes=(f"_{names[0]}", f"_{names[1]}"))
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "truth_pairs.csv", index=False)
    for k, v in e.items():
        v.to_parquet(out / f"errors_era5_{k}.parquet", index=False)
    age_table(e, fits).to_csv(out / "truth_age_table.csv", index=False)
    return pairs
