"""Evaluation A (independent of images): leave-one-aircraft-out prediction of the along-heading residual.

For every along-heading observation, the field is estimated from the OTHER aircraft only and projected on the heading.
ERA5 alone predicts a zero residual. Scores: RMSE of ERA5 and of the field, skill = 1 - MSE / MSE_ERA5, and the
calibration of the predictive uncertainty (normalised residual z = (y - mean bias - prediction) / predictive sd, with
its mean, sd and the coverage of the 68 % and 95 % intervals). With leave-one-day-out, the hyperparameters used for a
day are fitted on the other days only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    out = pd.DataFrame({"day": obs.day[i], "icao24": obs.icao24[i], "y": obs.y[i], "pred": t["pred"].to_numpy(float),
                        "var_field": t["var_field"].to_numpy(float), "var_total": t["var_total"].to_numpy(float)})
    out["z"] = (out["y"] - mean_bias - out["pred"]) / np.sqrt(out["var_total"])
    return out


def _scores(d: pd.DataFrame) -> dict[str, float]:
    mse0, mse1 = float((d["y"] ** 2).mean()), float(((d["y"] - d["pred"]) ** 2).mean())
    z = d["z"].to_numpy(float)
    return {"n_obs": float(len(d)), "n_aircraft": float(d["icao24"].nunique()), "rmse_era5": np.sqrt(mse0),
            "rmse_field": np.sqrt(mse1), "skill": 1 - mse1 / mse0, "z_mean": float(z.mean()), "z_sd": float(z.std()),
            "cover68": float((np.abs(z) <= 1).mean()), "cover95": float((np.abs(z) <= 1.96).mean()),
            "sd_field_median": float(np.sqrt(d["var_field"]).median())}


def score_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Scores per day and pooled."""
    rows = [{"day": day, **_scores(g)} for day, g in pred.groupby("day")]
    rows.append({"day": "pooled", **_scores(pred)})
    return pd.DataFrame(rows)


def _lodo(days: Sequence[str], fit: Callable[[list[str]], tuple[Any, dict[str, Any]]],
          predict: Callable[[str, Any], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, fits = [], []
    for day in days:
        params, summary = fit([d for d in days if d != day])
        fits.append({"held_out": day, **summary})
        preds.append(predict(day, params))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(fits)


def lodo_gp(obs_by_day: Mapping[str, ObsSet], base: GPParams, fit_kwargs: dict[str, Any] | None = None,
            est_kwargs: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GP hyperparameters fitted (marginal likelihood) on all days but one, LOAO on the held-out day."""
    fk, ek = fit_kwargs or {}, est_kwargs or {}

    def fit(train: list[str]) -> tuple[GPParams, dict[str, Any]]:
        res = fit_gp([obs_by_day[d] for d in train], base, **fk)
        return res.params, res.summary()

    def predict(day: str, p: GPParams) -> pd.DataFrame:
        return loao_frame(GPEstimator(obs_by_day[day], p, **ek), obs_by_day[day], p.effective().mean_bias)

    return _lodo(list(obs_by_day), fit, predict)


def lodo_ridge(obs_by_day: Mapping[str, ObsSet], base: RidgeConfig,
               grid: dict[str, Sequence[float]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ridge kernel and prior fitted by LOAO grid search on all days but one, LOAO on the held-out day."""

    def fit(train: list[str]) -> tuple[RidgeConfig, dict[str, Any]]:
        table = fit_ridge([obs_by_day[d] for d in train], base, grid)
        best = table.iloc[0].to_dict()
        cfg = base.model_copy(update={k: float(v) for k, v in best.items() if k in RidgeConfig.model_fields})
        return cfg, {**cfg.model_dump(), "train_mse": best["mse"]}

    def predict(day: str, cfg: RidgeConfig) -> pd.DataFrame:
        return loao_frame(RidgeEstimator(obs_by_day[day], cfg), obs_by_day[day], 0.0)

    return _lodo(list(obs_by_day), fit, predict)
