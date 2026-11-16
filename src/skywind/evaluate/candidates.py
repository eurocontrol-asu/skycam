"""Milestone 3 candidates, developed and judged on the dev days only.

* ``gp_twoscale``: two-scale field, hyperparameters from its leave-one-day-out evaluation A run.
* ``gp_cross_emaddc``: frozen-v1 GP plus across-heading observations for airframes with a reliable heading offset
  (EMADDC way: offsets estimated against ERA5 only, pooled over the OTHER days of the store, at least ``MIN_OBS``
  observations on ``MIN_DAYS`` days), with across-heading noise fixed from the day-to-day scatter of the offsets and
  the heading resolution (not fitted).

Adoption rule (declared before the results): adopted if, on the dev days, the median absolute drift error of evaluation
B improves by at least 2 m/min with a paired one-sided Wilcoxon p < 0.05 against frozen-v1, and the pooled skill of
evaluation A does not drop by more than 0.01.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from skywind.config import Settings
from skywind.evaluate.aircraft import loao_frame, score_predictions
from skywind.field.data import ObsSet
from skywind.field.gp import GPEstimator, GPParams
from skywind.field.heading import combine_offsets, cross_noise_terms, day_to_day_scatter_deg
from skywind.geo import unit_vectors
from skywind.observations import apply_heading_offsets
from skywind.pipeline import observation_set
from skywind.windfield import ResidualEstimator

__all__ = ["MIN_DAYS", "MIN_OBS", "adoption", "cross_check", "cross_params", "emaddc_obs", "estimator_factory"]

MIN_OBS, MIN_DAYS = 40, 2


def _kw(settings: Settings) -> dict[str, Any]:
    g = settings.gp
    return {"cutoff": g.cutoff, "tile_minutes": g.tile_minutes, "max_neighbours": g.max_neighbours}


def _table(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.paths.heading_table("era5"))


def cross_params(settings: Settings, day: str) -> GPParams:
    """Frozen-v1 parameters with the across-heading noise fixed from the offsets of the days other than ``day``."""
    t = _table(settings)
    scatter = day_to_day_scatter_deg(t, (day,))
    off = combine_offsets(t, (day,), MIN_OBS, MIN_DAYS)
    p = GPParams.from_config(settings.gp)
    noise, bias = cross_noise_terms(scatter, float(off["n_days"].median()), p.sigma_noise)
    return replace(p, sigma_cross_noise=noise, sigma_cross_bias=bias, fit_cross_noise=False)


def emaddc_obs(settings: Settings, day: str) -> ObsSet:
    return observation_set(settings, day, True, (day,), "era5", MIN_OBS, MIN_DAYS)


def estimator_factory(settings: Settings, twoscale_fits: Path | None) -> Callable[[str], dict[str, ResidualEstimator]]:
    """Per dev day: frozen-v1, the across-heading candidate and (if fitted) the two-scale candidate."""

    def build(day: str) -> dict[str, ResidualEstimator]:
        p = GPParams.from_config(settings.gp)
        out: dict[str, ResidualEstimator] = {
            "gp_frozen": GPEstimator(observation_set(settings, day), p, **_kw(settings)),
            "gp_cross_emaddc": GPEstimator(emaddc_obs(settings, day), cross_params(settings, day), **_kw(settings)),
        }
        if twoscale_fits is not None:
            fits = pd.read_csv(twoscale_fits, dtype={"held_out": str})
            row = fits[fits["held_out"] == day].iloc[0].to_dict()
            q = replace(p, **{k: row[k] for k in p.as_dict() if k in row and pd.notna(row[k])})
            out["gp_twoscale"] = GPEstimator(observation_set(settings, day), q, **_kw(settings))
        return out

    return build


def cross_check(settings: Settings, days: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluation A (along-heading LOAO) with and without the across-heading observations (same frozen-v1 field
    parameters), and the image-independent check of the full vector: the calibrated across-heading residual of each
    held-out aircraft with a reliable offset, predicted from the other aircraft with and without the across
    observations. Returns (scores A, scores of the across check)."""
    a_rows, c_rows = [], []
    for day in days:
        p0 = GPParams.from_config(settings.gp)
        base = GPEstimator(observation_set(settings, day), p0, **_kw(settings))
        cand = GPEstimator(emaddc_obs(settings, day), cross_params(settings, day), **_kw(settings))
        for name, est in (("gp_frozen", base), ("gp_cross_emaddc", cand)):
            a_rows.append(loao_frame(est, est.obs, p0.effective().mean_bias).assign(variant=name))
            c_rows.append(_across(settings, day, est).assign(variant=name))
    a = pd.concat(a_rows, ignore_index=True)
    sa = pd.concat([score_predictions(g).assign(variant=v) for v, g in a.groupby("variant")], ignore_index=True)
    c = pd.concat(c_rows, ignore_index=True)
    return sa, _across_scores(c)


def _across(settings: Settings, day: str, est: GPEstimator) -> pd.DataFrame:
    """Calibrated across-heading residual vs the field predicted from the other aircraft, at reliable airframes."""
    w = pd.read_parquet(settings.paths.observations(day))
    off = combine_offsets(_table(settings), (day,), MIN_OBS, MIN_DAYS)["offset_deg"]
    w = apply_heading_offsets(w, off)
    loo = est.loao(vectors=True)
    rows = loo["obs_index"].to_numpy()  # along rows come first, in the order of the observation table
    ok = np.isfinite(w["hdg_offset"].to_numpy()[rows])
    _, _, qx, qy = unit_vectors(w["hdg"].to_numpy()[rows])
    pred = loo["du"].to_numpy() * qx + loo["dv"].to_numpy() * qy
    frame = pd.DataFrame(
        {"day": day, "icao24": w["icao24"].to_numpy()[rows], "y": w["d_right"].to_numpy()[rows], "pred": pred}
    )
    result: pd.DataFrame = frame[ok].reset_index(drop=True)
    return result


def _across_scores(c: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = [*c.groupby(["variant", "day"]), *(((k, "pooled"), g) for k, g in c.groupby("variant"))]
    for (v, day), g in groups:
        mse0, mse1 = float((g["y"] ** 2).mean()), float(((g["y"] - g["pred"]) ** 2).mean())
        out.append(
            {
                "variant": v,
                "day": day,
                "n_obs": len(g),
                "rmse_era5": np.sqrt(mse0),
                "rmse_field": np.sqrt(mse1),
                "skill": 1 - mse1 / mse0,
            }
        )
    return pd.DataFrame(out)


def adoption(
    paired: pd.DataFrame, candidate: str, skill_reference: float, skill_candidate: float, reference: str = "gp_frozen"
) -> dict[str, Any]:
    """Apply the declared adoption rule to one candidate (paired per-flight drifts of B; pooled A skills of the
    candidate and of frozen-v1 under the same protocol)."""
    d = (paired[f"drift_{reference}"].abs() - paired[f"drift_{candidate}"].abs()).to_numpy()
    d = d[np.isfinite(d)]
    med_ref = float(paired[f"drift_{reference}"].abs().median())
    med_cand = float(paired[f"drift_{candidate}"].abs().median())
    p = float(wilcoxon(d[d != 0], alternative="greater").pvalue)
    drop = skill_reference - skill_candidate
    adopted = (med_ref - med_cand >= 2.0) and p < 0.05 and drop <= 0.01
    return {
        "candidate": candidate,
        "n_flights": len(d),
        "median_drift_reference": med_ref,
        "median_drift_candidate": med_cand,
        "median_improvement": med_ref - med_cand,
        "wilcoxon_p": p,
        "skill_A_reference": skill_reference,
        "skill_A_candidate": skill_candidate,
        "skill_drop": drop,
        "adopted": bool(adopted),
    }
