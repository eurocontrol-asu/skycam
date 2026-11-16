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
    frozen_estimators,
    lodo_estimators,
    run_aircraft,
    run_calibration,
    run_contrails,
    run_diagnostics,
    run_ridge,
    run_transfer,
)
from skywind.field.gp import GPEstimator, GPParams
from skywind.runs import new_run_dir, write_json

__all__ = [
    "aircraft_campaign",
    "candidates_campaign",
    "contrail_campaign",
    "diagnostics_campaign",
    "holdout_aircraft_campaign",
]


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
    estimators = (
        frozen_estimators(settings)
        if not aircraft_runs
        else lodo_estimators(settings, aircraft_runs, [v for v in variants if v != "ridge"], "ridge" in variants)
    )
    res = run_contrails(settings, source, estimators, out, days)
    pooled = res["drifts"][res["drifts"]["day"] == "pooled"]
    return {"run": str(out), "pooled_drift": pooled.to_dict("records")}


def diagnostics_campaign(settings: Settings, params: GPParams, days: list[str] | None = None) -> dict[str, Any]:
    """Hollingsworth-Lonnberg and Desroziers diagnostics of the given hyperparameters on the dev days."""
    days = days or available_days(settings.paths, "dev")
    out = new_run_dir(settings.paths.runs, "diagnostics", f"innovation diagnostics on {', '.join(days)} for {params}")
    return {"run": str(out), **run_diagnostics(settings, days, params, out)}


def holdout_aircraft_campaign(settings: Settings, days: list[str] | None = None) -> dict[str, Any]:
    """Evaluation A on held-out days with the frozen configuration, no hyperparameter fitted on them:
    ``gp_frozen`` (shipped GP hyperparameters as is), ``gp_frozen_dayvar`` (shipped procedure: variances re-estimated
    on the day's own aircraft data) and ``ridge_dev`` (ridge with its dev-fitted kernel)."""
    from skywind.evaluate.aircraft import loao_frame, score_predictions
    from skywind.field.hyper import fit_gp
    from skywind.field.ridge import RidgeEstimator
    from skywind.pipeline import DAY_VARIANCES, decode_step, observation_set, observe_step

    days = days or available_days(settings.paths, "holdout")
    out = new_run_dir(
        settings.paths.runs,
        "holdout-aircraft",
        f"evaluation A on holdout days {', '.join(days)} with the frozen configuration (no refit)",
    )
    p = GPParams.from_config(settings.gp)
    kw: dict[str, Any] = {
        "cutoff": settings.gp.cutoff,
        "tile_minutes": settings.gp.tile_minutes,
        "max_neighbours": settings.gp.max_neighbours,
    }
    preds: dict[str, list[pd.DataFrame]] = {"gp_frozen": [], "gp_frozen_dayvar": [], "ridge_dev": []}
    for day in days:
        if not settings.paths.observations(day).is_file():
            decode_step(settings, day)
            observe_step(settings, day)
        obs = observation_set(settings, day)
        preds["gp_frozen"].append(loao_frame(GPEstimator(obs, p, **kw), obs, p.effective().mean_bias))
        pd_ = fit_gp([obs], p, only=DAY_VARIANCES, workers=settings.max_workers).params
        preds["gp_frozen_dayvar"].append(loao_frame(GPEstimator(obs, pd_, **kw), obs, pd_.effective().mean_bias))
        preds["ridge_dev"].append(loao_frame(RidgeEstimator(obs, settings.ridge), obs, 0.0))
    scores = []
    for name, frames in preds.items():
        table = pd.concat(frames, ignore_index=True)
        table.to_parquet(out / f"loao_{name}.parquet", index=False)
        scores.append(score_predictions(table).assign(variant=name))
    summary = pd.concat(scores, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    write_json(out / "frozen_config.json", settings.model_dump(mode="json"))
    pooled = summary[summary["day"] == "pooled"][["variant", "rmse_era5", "rmse_field", "skill", "z_sd", "cover95"]]
    return {"run": str(out), "pooled": pooled.to_dict("records")}


def _pooled_skill(table: pd.DataFrame, variant: str) -> float:
    return float(table[(table["variant"] == variant) & (table["day"] == "pooled")]["skill"].iloc[0])


def candidates_campaign(
    settings: Settings, twoscale_run: Path | None, days: list[str] | None = None, skip_b: bool = False
) -> dict[str, Any]:
    """Milestone 3 candidates on the dev days: across-heading checks (A, full vector) and evaluation B with frozen-v1,
    the across-heading candidate and the two-scale candidate (raw GVCCS truth); adoption rule applied."""
    from skywind.evaluate.candidates import adoption, cross_check, estimator_factory

    days = days or available_days(settings.paths, "dev")
    out = new_run_dir(settings.paths.runs, "candidates", f"milestone 3 candidates on the dev days {', '.join(days)}")
    a_cross, across = cross_check(settings, days)
    a_cross.to_csv(out / "a_cross.csv", index=False)
    across.to_csv(out / "across_check.csv", index=False)
    result: dict[str, Any] = {"run": str(out)}
    if skip_b:
        return result
    fits = twoscale_run / "fits_gp_twoscale.csv" if twoscale_run else None
    res = run_contrails(settings, GvccsRaw(settings), estimator_factory(settings, fits), out, days)
    pooled_a = a_cross[a_cross["day"] == "pooled"].set_index("variant")["skill"]
    rules = [
        adoption(res["paired"], "gp_cross_emaddc", float(pooled_a["gp_frozen"]), float(pooled_a["gp_cross_emaddc"]))
    ]
    if twoscale_run is not None:
        ts = pd.read_csv(twoscale_run / "summary.csv", dtype={"day": str})
        ref = pd.read_csv(settings.paths.runs / "20261113-eval-aircraft-4" / "summary.csv", dtype={"day": str})
        rules.append(
            adoption(
                res["paired"], "gp_twoscale", _pooled_skill(ref, "gp_matern_flow"), _pooled_skill(ts, "gp_twoscale")
            )
        )
    table = pd.DataFrame(rules)
    table.to_csv(out / "adoption.csv", index=False)
    result["adoption"] = table.to_dict("records")
    return result
