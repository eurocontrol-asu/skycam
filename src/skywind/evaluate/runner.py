"""Evaluation campaigns A (aircraft, leave-one-aircraft-out / leave-one-day-out) and B (contrails), written to a run
directory. Estimator variants are named; every variant is evaluated with the same folds on the same data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from skywind.config import Settings
from skywind.era5 import Era5
from skywind.evaluate.aircraft import loao_frame, lodo_gp, lodo_ridge, score_predictions
from skywind.evaluate.contrails import (
    ContrailTruth,
    TruthSource,
    age_table,
    cross_track_errors,
    drift_table,
    flight_fits,
)
from skywind.field.data import ObsSet
from skywind.field.diagnostics import desroziers, fit_correlation_length, hollingsworth_lonnberg
from skywind.field.gp import GPEstimator, GPParams
from skywind.field.hyper import fit_gp
from skywind.field.ridge import RidgeEstimator
from skywind.pipeline import advect_step, calibrate_step, export_step, observation_set
from skywind.runs import write_json
from skywind.windfield import CorrectedWind, ResidualEstimator

__all__ = ["VARIANTS", "Variant", "run_aircraft", "run_calibration", "run_contrails", "run_diagnostics", "run_ridge"]


@dataclass(frozen=True)
class Variant:
    """A GP configuration: parameter overrides, cross-heading observations, strict (leave-day-out) offset table."""

    overrides: dict[str, Any] = field(default_factory=dict)
    use_cross: bool = False
    strict_offsets: bool = False


VARIANTS: dict[str, Variant] = {
    "gp": Variant(),
    "gp_nobias": Variant({"use_bias": False}),
    "gp_flow": Variant({"flow_following": True}),
    "gp_matern": Variant({"kernel": "matern32"}),
    "gp_cross": Variant(use_cross=True),
    "gp_cross_strict": Variant(use_cross=True, strict_offsets=True),
}


def _est_kwargs(settings: Settings) -> dict[str, Any]:
    g = settings.gp
    return {"cutoff": g.cutoff, "tile_minutes": g.tile_minutes, "max_neighbours": g.max_neighbours}


def _provider(settings: Settings, days: Sequence[str], v: Variant) -> Callable[[str], Mapping[str, ObsSet]]:
    cache: dict[tuple[str, ...], dict[str, ObsSet]] = {}

    def provide(held_out: str) -> Mapping[str, ObsSet]:
        excl = (held_out,) if v.strict_offsets else ()
        if excl not in cache:
            cache[excl] = {d: observation_set(settings, d, v.use_cross, excl) for d in days}
        return cache[excl]

    return provide


def run_aircraft(settings: Settings, days: Sequence[str], variant: str, out: Path) -> pd.DataFrame:
    """Evaluation A for one GP variant: LODO fits, LOAO predictions and scores (+ all-days fit for reference)."""
    v = VARIANTS[variant]
    base = replace(GPParams.from_config(settings.gp), **v.overrides)
    provider = _provider(settings, days, v)
    fit_kwargs: dict[str, Any] = {"workers": settings.max_workers}
    preds, fits = lodo_gp(days, provider, base, fit_kwargs, _est_kwargs(settings))
    sets = provider("")
    all_fit = fit_gp([sets[d] for d in days], base, workers=settings.max_workers)
    fits = pd.concat([fits, pd.DataFrame([{"held_out": "none (all days)", **all_fit.summary()}])], ignore_index=True)
    scores = score_predictions(preds).assign(variant=variant)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / f"loao_{variant}.parquet", index=False)
    fits.to_csv(out / f"fits_{variant}.csv", index=False)
    scores.to_csv(out / f"scores_{variant}.csv", index=False)
    write_json(out / f"params_{variant}_alldays.json", all_fit.params.as_dict())
    return scores


def run_transfer(settings: Settings, days: Sequence[str], out: Path) -> pd.DataFrame:
    """LOAO of each day with the all-days parameters and with the parameters fitted on that day alone (in-sample),
    to compare with the leave-one-day-out scores."""
    base = GPParams.from_config(settings.gp)
    sets = {d: observation_set(settings, d) for d in days}
    p_all = fit_gp(list(sets.values()), base, workers=settings.max_workers).params
    rows = []
    for d in days:
        own = fit_gp([sets[d]], base, workers=settings.max_workers)
        for label, p in (("all_days", p_all), ("own_day", own.params)):
            s = score_predictions(
                loao_frame(GPEstimator(sets[d], p, **_est_kwargs(settings)), sets[d], p.effective().mean_bias)
            )
            row: dict[str, Any] = {"day": d, "params": label}
            row.update({str(k): v for k, v in s.iloc[0].drop("day").items()})
            row.update({f"p_{k}": val for k, val in p.as_dict().items() if isinstance(val, float)})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "transfer.csv", index=False)
    return table


def run_ridge(settings: Settings, days: Sequence[str], out: Path) -> pd.DataFrame:
    """Evaluation A for the ridge baseline (LODO grid search) and for its fixed prototype kernel."""
    sets = {d: observation_set(settings, d) for d in days}
    preds, fits = lodo_ridge(sets, settings.ridge, workers=settings.max_workers)
    proto = pd.concat([loao_frame(RidgeEstimator(sets[d], settings.ridge), sets[d], 0.0) for d in days])
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "loao_ridge.parquet", index=False)
    fits.to_csv(out / "fits_ridge.csv", index=False)
    scores = pd.concat(
        [
            score_predictions(preds).assign(variant="ridge"),
            score_predictions(proto).assign(variant="ridge_prototype_75km_500m_20min"),
        ]
    )
    scores.to_csv(out / "scores_ridge.csv", index=False)
    return scores


def run_calibration(settings: Settings, days: Sequence[str], params: GPParams) -> pd.DataFrame:
    """(Re)compute the heading-offset table rows of ``days``."""
    return pd.DataFrame([calibrate_step(settings, d, params) for d in days])


def run_diagnostics(settings: Settings, days: Sequence[str], params: GPParams, out: Path) -> dict[str, Any]:
    """Hollingsworth-Lonnberg binned covariances (+ Gaussian fits) and Desroziers ratios per day."""
    sets = [observation_set(settings, d) for d in days]
    hl = hollingsworth_lonnberg(sets, params.effective().mean_bias)
    hl.to_csv(out / "hollingsworth_lonnberg.csv", index=False)
    fitsd = {dim: fit_correlation_length(hl, dim) for dim in ("horizontal_km", "vertical_m", "time_s")}
    des = [
        {"day": d, **desroziers(GPEstimator(o, params, **_est_kwargs(settings)))}
        for d, o in zip(days, sets, strict=False)
    ]
    pd.DataFrame(des).to_csv(out / "desroziers.csv", index=False)
    summary = {"hl_gaussian_fits": fitsd, "desroziers": des}
    write_json(out / "diagnostics.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ contrails (B)
def _normal_std(ds: xr.Dataset, adv: pd.DataFrame, truth: ContrailTruth) -> pd.DataFrame:
    """Residual-wind posterior sd along the contrail normal at the annotated (point, frame) positions."""
    m = truth.centres.merge(adv, on=["flight_id", "waypoint", "time"])
    sel = {
        "longitude": xr.DataArray(m["longitude"].to_numpy(), dims="p"),
        "latitude": xr.DataArray(m["latitude"].to_numpy(), dims="p"),
        "level": xr.DataArray(m["level"].to_numpy(), dims="p"),
        "time": xr.DataArray(m["time"].dt.tz_localize(None).to_numpy(), dims="p"),
    }
    names = ["residual_eastward_wind_std", "residual_northward_wind_std", "residual_wind_covariance"]
    su, sv, cuv = (ds[n].interp(sel).values for n in names)
    var = su**2 * m["normal_e"] ** 2 + 2 * cuv * m["normal_e"] * m["normal_n"] + sv**2 * m["normal_n"] ** 2
    m["std_normal"] = np.sqrt(np.maximum(var, 0))
    return m.groupby("flight", as_index=False).agg(std_normal=("std_normal", "mean"))


def _contrail_day(
    settings: Settings, truth: ContrailTruth, fields: Mapping[str, ResidualEstimator], out: Path, era: Era5
) -> dict[str, Any]:
    start = pd.Timestamp(truth.emissions["emission_time"].min())
    stop = truth.last_time
    s = settings.model_copy(deep=True)
    s.export.pad_before_min, s.export.pad_after_min = 5.0, 5.0
    errors, stds = {}, {}
    for name, est in fields.items():
        path = export_step(
            s,
            truth.day,
            CorrectedWind(era, est, s),
            start,
            stop,
            out / "fields" / f"{name}_{truth.day}.nc",
            {"variant": name},
        )
        if "era5" not in errors:
            adv = advect_step(s, path, truth.emissions, stop, winds="era5", max_age_min=45)
            errors["era5"] = cross_track_errors(truth, adv, s.site)
        adv = advect_step(s, path, truth.emissions, stop, winds="corrected", max_age_min=45)
        errors[name] = cross_track_errors(truth, adv, s.site)
        stds[name] = _normal_std(xr.open_dataset(path), adv, truth).assign(day=truth.day)
    return {"errors": errors, "stds": stds}


def run_contrails(
    settings: Settings,
    source: TruthSource,
    estimators: Callable[[str], Mapping[str, ResidualEstimator]],
    out: Path,
    days: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluation B: for every day, export each named estimator, advect, score against the truth source."""
    errors: dict[str, list[pd.DataFrame]] = {}
    stds: dict[str, list[pd.DataFrame]] = {}
    for day in days or source.days():
        truth = source.load(day)
        res = _contrail_day(settings, truth, estimators(day), out, Era5.from_file(settings.paths.era5(day)))
        for k, e in res["errors"].items():
            errors.setdefault(k, []).append(e)
        for k, s in res["stds"].items():
            stds.setdefault(k, []).append(s)
    return summarise_contrails(
        {k: pd.concat(v) for k, v in errors.items()}, {k: pd.concat(v) for k, v in stds.items()}, out
    )


def summarise_contrails(
    errors: Mapping[str, pd.DataFrame], stds: Mapping[str, pd.DataFrame], out: Path
) -> dict[str, pd.DataFrame]:
    fits = {k: flight_fits(e) for k, e in errors.items()}
    paired, drifts = drift_table(fits)
    ages = age_table(errors, fits)
    calib = []
    for k, sd in stds.items():
        f = fits[k].merge(sd, on=["day", "flight"]).dropna()
        z = f["drift_m_per_min"] / (60 * f["std_normal"])
        calib.append(
            {
                "variant": k,
                "n_flights": len(f),
                "median_std_normal": float(f["std_normal"].median()),
                "z_sd": float(z.std()),
                "cover68": float((z.abs() <= 1).mean()),
                "cover95": float((z.abs() <= 1.96).mean()),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    for k, e in errors.items():
        e.to_parquet(out / f"errors_{k}.parquet", index=False)
    paired.to_csv(out / "flights_paired.csv", index=False)
    drifts.to_csv(out / "drift_summary.csv", index=False)
    ages.to_csv(out / "age_summary.csv", index=False)
    pd.DataFrame(calib).to_csv(out / "uncertainty_calibration.csv", index=False)
    return {"paired": paired, "drifts": drifts, "ages": ages, "calibration": pd.DataFrame(calib)}


# ------------------------------------------------------------------------------------------------ campaigns
def _fold_params(fits: pd.DataFrame, day: str, base: GPParams) -> GPParams:
    row = fits[fits["held_out"].astype(str) == day].iloc[0].to_dict()
    return replace(base, **{k: row[k] for k in base.as_dict() if k in row and pd.notna(row[k])})


def lodo_estimators(
    settings: Settings, aircraft_run: Path, variants: Sequence[str], with_ridge: bool = True
) -> Callable[[str], dict[str, ResidualEstimator]]:
    """Per day, the estimators of each variant with the hyperparameters fitted WITHOUT that day (evaluation A run)."""

    def build(day: str) -> dict[str, ResidualEstimator]:
        out: dict[str, ResidualEstimator] = {}
        for name in variants:
            v = VARIANTS[name]
            base = replace(GPParams.from_config(settings.gp), **v.overrides)
            p = _fold_params(pd.read_csv(aircraft_run / f"fits_{name}.csv"), day, base)
            obs = observation_set(settings, day, v.use_cross, (day,) if v.strict_offsets else ())
            out[name] = GPEstimator(obs, p, **_est_kwargs(settings))
        if with_ridge:
            fits = pd.read_csv(aircraft_run / "fits_ridge.csv", dtype={"held_out": str})
            row = fits[fits["held_out"] == day].iloc[0].to_dict()
            cfg = settings.ridge.model_copy(
                update={k: float(row[k]) for k in ("length_h_km", "length_z_m", "length_t_min", "sigma_prior")}
            )
            out["ridge"] = RidgeEstimator(observation_set(settings, day), cfg)
        return out

    return build
