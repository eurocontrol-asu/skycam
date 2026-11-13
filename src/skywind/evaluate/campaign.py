"""Evaluation campaigns as one call each: create the run directory (with its README), run, summarise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from skywind.config import Settings
from skywind.days import available_days
from skywind.evaluate.contrails import TruthSource
from skywind.evaluate.gvccs_calib import CalibArtefacts
from skywind.evaluate.gvccs_raw import GvccsRaw
from skywind.evaluate.runner import (
    VARIANTS,
    lodo_estimators,
    run_aircraft,
    run_calibration,
    run_contrails,
    run_diagnostics,
    run_ridge,
    run_transfer,
)
from skywind.field.gp import GPParams
from skywind.runs import new_run_dir

__all__ = ["aircraft_campaign", "contrail_campaign", "diagnostics_campaign"]


def _ensure_calibration(settings: Settings, days: list[str]) -> None:
    path = settings.paths.heading_table()
    have = set(pd.read_parquet(path)["day"].astype(str)) if path.is_file() else set()
    missing = [d for d in days if d not in have]
    if missing:
        run_calibration(settings, missing, GPParams.from_config(settings.gp))


def aircraft_campaign(
    settings: Settings,
    variants: list[str],
    days: list[str] | None = None,
    ridge: bool = True,
    diagnostics: bool = True,
    transfer: bool = True,
    fits_from: list[Path] | None = None,
) -> dict[str, Any]:
    days = days or available_days(settings.paths, "dev")  # holdout days are never used for tuning
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
    out = new_run_dir(
        settings.paths.runs,
        "eval-aircraft",
        "evaluation A (leave-one-aircraft-out, leave-one-day-out "
        f"hyperparameters) on {', '.join(days)}; variants {', '.join(variants)}",
    )
    if any(VARIANTS[v].use_cross for v in variants):
        _ensure_calibration(settings, days)
    scores = [run_aircraft(settings, days, v, out, fits_from or []) for v in variants]
    if ridge:
        scores.append(run_ridge(settings, days, out))
    if transfer:
        run_transfer(settings, days, out)
    if diagnostics:
        run_diagnostics(settings, days, GPParams.from_config(settings.gp), out)
    table = pd.concat(scores, ignore_index=True)
    table.to_csv(out / "summary.csv", index=False)
    pooled = table[table["day"] == "pooled"][["variant", "rmse_era5", "rmse_field", "skill", "z_sd", "cover68"]]
    return {"run": str(out), "pooled": pooled.to_dict("records")}


def contrail_campaign(
    settings: Settings,
    aircraft_runs: list[Path],
    variants: list[str],
    days: list[str] | None = None,
    truth: str = "raw",
) -> dict[str, Any]:
    """Evaluation B. ``truth``: ``raw`` (GVCCS annotations, any day) or ``calib`` (prototype artefacts, six days)."""
    source: TruthSource = GvccsRaw(settings) if truth == "raw" else CalibArtefacts(settings)
    days = days or source.days()
    out = new_run_dir(
        settings.paths.runs,
        "eval-contrails",
        f"evaluation B (advection vs GVCCS, truth {truth}) "
        f"on {', '.join(days)}; variants {', '.join(variants)}; "
        f"hyperparameters from {', '.join(map(str, aircraft_runs))}",
    )
    res = run_contrails(
        settings,
        source,
        lodo_estimators(settings, aircraft_runs, [v for v in variants if v != "ridge"], "ridge" in variants),
        out,
        days,
    )
    pooled = res["drifts"][res["drifts"]["day"] == "pooled"]
    return {"run": str(out), "pooled_drift": pooled.to_dict("records")}


def diagnostics_campaign(settings: Settings, params: GPParams, days: list[str] | None = None) -> dict[str, Any]:
    """Hollingsworth-Lonnberg and Desroziers diagnostics of the given hyperparameters on the dev days."""
    days = days or available_days(settings.paths, "dev")
    out = new_run_dir(settings.paths.runs, "diagnostics", f"innovation diagnostics on {', '.join(days)} for {params}")
    return {"run": str(out), **run_diagnostics(settings, days, params, out)}
