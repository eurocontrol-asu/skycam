"""Evaluation A (independent of images): leave-one-aircraft-out prediction of the along-heading residual.

For every along-heading observation, the field is estimated from the OTHER aircraft only and projected on the heading.
ERA5 alone predicts a zero residual. Scores: RMSE of ERA5 and of the field, skill = 1 - MSE / MSE_ERA5, and the
calibration of the predictive uncertainty (normalised residual z = (y - mean bias - prediction) / predictive sd, with
its mean, sd and the coverage of the 68 % and 95 % intervals). With leave-one-day-out, the hyperparameters used for a
day are fitted on the other days only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Protocol

import numpy as np
import pandas as pd

from skywind.config import RidgeConfig
from skywind.field.data import ObsSet
from skywind.field.gp import GPEstimator, GPParams
from skywind.field.hyper import fit_gp, fit_ridge
from skywind.field.ridge import RidgeEstimator

__all__ = ["loao_frame", "lodo_gp", "lodo_ridge", "score_predictions"]


class _Loao(Protocol):
    def loao(self, vectors: bool = False) -> pd.DataFrame: ...


def loao_frame(est: _Loao, obs: ObsSet, mean_bias: float) -> pd.DataFrame:
    """LOAO predictions joined with the observations: day, icao24, y, pred, var_field, var_total, z."""
    t = est.loao()
    i = t["obs_index"].to_numpy(int)
    out = pd.DataFrame(
        {
            "day": obs.day[i],
            "icao24": obs.icao24[i],
            "y": obs.y[i],
            "pred": t["pred"].to_numpy(float),
            "var_field": t["var_field"].to_numpy(float),
            "var_total": t["var_total"].to_numpy(float),
        }
    )
    out["z"] = (out["y"] - mean_bias - out["pred"]) / np.sqrt(out["var_total"])
    return out


def _scores(d: pd.DataFrame) -> dict[str, float]:
    mse0, mse1 = float((d["y"] ** 2).mean()), float(((d["y"] - d["pred"]) ** 2).mean())
    z = d["z"].to_numpy(float)
    return {
        "n_obs": float(len(d)),
        "n_aircraft": float(d["icao24"].nunique()),
        "rmse_era5": np.sqrt(mse0),
        "rmse_field": np.sqrt(mse1),
        "skill": 1 - mse1 / mse0,
        "z_mean": float(z.mean()),
        "z_sd": float(z.std()),
        "cover68": float((np.abs(z) <= 1).mean()),
        "cover95": float((np.abs(z) <= 1.96).mean()),
        "sd_field_median": float(np.sqrt(d["var_field"]).median()),
    }


def score_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Scores per day and pooled."""
    rows = [{"day": day, **_scores(g)} for day, g in pred.groupby("day")]
    rows.append({"day": "pooled", **_scores(pred)})
    return pd.DataFrame(rows)


def lodo_gp(
    days: Sequence[str],
    obs_provider: Callable[[str], Mapping[str, ObsSet]],
    base: GPParams,
    fit_kwargs: dict[str, Any] | None = None,
    est_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GP hyperparameters fitted (marginal likelihood) on all days but one, LOAO on the held-out day.

    ``obs_provider(held_out)`` returns the observation sets of every day for that fold (so that a calibration table can
    exclude the held-out day).
    """
    fk, ek = fit_kwargs or {}, est_kwargs or {}
    preds, fits = [], []
    for day in days:
        sets = obs_provider(day)
        res = fit_gp([sets[d] for d in days if d != day], base, **fk)
        fits.append({"held_out": day, **res.summary()})
        preds.append(loao_frame(GPEstimator(sets[day], res.params, **ek), sets[day], res.params.effective().mean_bias))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(fits)


def ridge_grid_tables(
    obs_by_day: Mapping[str, ObsSet],
    base: RidgeConfig,
    grid: dict[str, Sequence[float]] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """LOAO squared-error sums of every grid combination, per day (computed once, reused by every fold)."""
    days = list(obs_by_day)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(days)))) as pool:
        tables = list(pool.map(fit_ridge, [[obs_by_day[d]] for d in days], [base] * len(days), [grid] * len(days)))
    return pd.concat(
        [t.assign(day=d, sse=t["mse"] * t["n"]) for d, t in zip(days, tables, strict=False)], ignore_index=True
    )


def lodo_ridge(
    obs_by_day: Mapping[str, ObsSet],
    base: RidgeConfig,
    grid: dict[str, Sequence[float]] | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ridge kernel and prior chosen by pooled LOAO error on all days but one, LOAO on the held-out day."""
    table = ridge_grid_tables(obs_by_day, base, grid, workers)
    keys = [k for k in table.columns if k in RidgeConfig.model_fields]
    preds, fits = [], []
    for day in obs_by_day:
        train = table[table["day"] != day].groupby(keys, as_index=False)[["sse", "n"]].sum()
        best = train.assign(mse=train["sse"] / train["n"]).sort_values("mse").iloc[0]
        cfg = base.model_copy(update={k: float(best[k]) for k in keys})
        fits.append({"held_out": day, **cfg.model_dump(), "train_mse": float(best["mse"])})
        preds.append(loao_frame(RidgeEstimator(obs_by_day[day], cfg), obs_by_day[day], 0.0))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(fits)
