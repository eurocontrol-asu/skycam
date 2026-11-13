"""Per-day processing steps on the input store: decode -> observe -> fit -> export -> advect.

Each step reads its inputs from the configured paths, writes its output to the working-data cache and returns a small
summary. The command-line interface only calls these functions.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from skywind import __version__
from skywind.advect import advect, emission_points, met_dataset
from skywind.config import Settings
from skywind.decode import decode_payloads
from skywind.era5 import Era5
from skywind.export import era5_on_grid, export_grid, residual_dataset, to_met_dataset, write_netcdf
from skywind.field.data import ObsSet
from skywind.field.gp import GPEstimator, GPParams
from skywind.field.heading import combine_offsets, offset_terms, update_table
from skywind.field.hyper import fit_gp
from skywind.identify import decode_day
from skywind.observations import add_residuals, apply_heading_offsets, bin_observations
from skywind.runs import write_json
from skywind.statevectors import add_position_track, load_state_vectors, sanitise
from skywind.windfield import CorrectedWind, ResidualEstimator, build_estimator

__all__ = [
    "DAY_VARIANCES",
    "advect_step",
    "calibrate_step",
    "cruise_emission_points",
    "decode_step",
    "export_step",
    "fit_step",
    "fitted_field",
    "gp_params",
    "load_observations",
    "observation_set",
    "observe_step",
    "state_vectors",
]


def state_vectors(settings: Settings, day: str, clean: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    """State vectors of the day with position-derived track; ``clean`` applies the label-free sanitiser.

    Decoding uses the raw extract (register identification and the receiver clock check compare replies with the
    reported speeds and the position track, as validated in the prototype); emission points use the sanitised one.
    """
    sv = load_state_vectors(settings.paths.opensky(day))
    report: dict[str, float] = {}
    if clean:
        sv, report = sanitise(sv)
    return add_position_track(sv), report


def decode_step(settings: Settings, day: str) -> dict[str, Any]:
    """Raw Comm-B -> identified BDS 5,0 / 6,0 messages with their nearest state vector."""
    paths = settings.paths
    raw = pd.read_parquet(paths.commb(day))
    decoded = decode_payloads(raw)
    sv, report = state_vectors(settings, day, clean=False)
    msgs, receivers, stats = decode_day(decoded, sv, settings.decode, settings.site.min_alt_ft)
    out = paths.decoded(day)
    out.parent.mkdir(parents=True, exist_ok=True)
    msgs.to_parquet(out, index=False)
    receivers.to_csv(paths.receivers(day))
    return {"day": day, "raw": len(raw), **stats, **{f"sanitise_{k}": v for k, v in report.items()}, "output": str(out)}


def observe_step(settings: Settings, day: str) -> dict[str, Any]:
    """Identified messages -> 30 s wind observations with ERA5 residuals."""
    paths = settings.paths
    msgs = pd.read_parquet(paths.decoded(day))
    w = bin_observations(msgs, day, settings.observation, settings.site.min_alt_ft)
    w = add_residuals(w, Era5.from_file(paths.era5(day)))
    out = paths.observations(day)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(out, index=False)
    return {
        "day": day,
        "observations": len(w),
        "aircraft": int(w["icao24"].nunique()),
        "along_bias": float(w["d_along"].mean()),
        "along_sd": float(w["d_along"].std()),
        "right_bias": float(w["d_right"].mean()),
        "right_sd": float(w["d_right"].std()),
        "output": str(out),
    }


def load_observations(settings: Settings, days: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(settings.paths.observations(d)) for d in days]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def observation_set(
    settings: Settings, day: str, use_cross: bool = False, exclude_offset_days: tuple[str, ...] = ()
) -> ObsSet:
    """Observations of a day as an :class:`ObsSet`; with ``use_cross``, heading offsets from the persistent table
    (optionally excluding days) are applied and cross-heading rows added for the calibrated airframes."""
    w = pd.read_parquet(settings.paths.observations(day))
    if use_cross:
        table = pd.read_parquet(settings.paths.heading_table())
        offsets = combine_offsets(table, exclude_offset_days)["offset_deg"]
        w = apply_heading_offsets(w, offsets)
    return ObsSet.from_frame(w, settings.site, use_cross=use_cross)


def gp_params(settings: Settings, params_file: Path | None = None) -> GPParams:
    """GP hyperparameters from the configuration, overridden by a fitted-parameter JSON file if given."""
    base = GPParams.from_config(settings.gp)
    if params_file is None:
        return base
    values = json.loads(params_file.read_text())
    return replace(base, **{k: v for k, v in values.items() if k in GPParams.__dataclass_fields__})


def calibrate_step(settings: Settings, day: str, params: GPParams, reference: str = "field") -> dict[str, Any]:
    """Heading-offset statistics of a day against ERA5 (+ the along-only field of the other aircraft) -> table."""
    w = pd.read_parquet(settings.paths.observations(day))
    u_ref, v_ref = w["u_era"].to_numpy(float).copy(), w["v_era"].to_numpy(float).copy()
    if reference == "field":
        obs = ObsSet.from_frame(w, settings.site)
        loo = GPEstimator(
            obs,
            replace(params, flow_following=False),
            settings.gp.cutoff,
            settings.gp.tile_minutes,
            settings.gp.max_neighbours,
        ).loao(vectors=True)
        idx = loo["obs_index"].to_numpy(int)
        u_ref[idx] += loo["du"].to_numpy(float)
        v_ref[idx] += loo["dv"].to_numpy(float)
    terms = offset_terms(w, u_ref, v_ref)
    table = update_table(settings.paths.heading_table(), day, terms, reference)
    return {"day": day, "airframes": len(terms), "table_rows": len(table), "table_days": int(table["day"].nunique())}


def fit_step(settings: Settings, days: list[str], base: GPParams, out: Path, use_cross: bool = False) -> dict[str, Any]:
    """Marginal-likelihood fit of the GP hyperparameters on ``days``; written as JSON."""
    sets = [observation_set(settings, d, use_cross) for d in days]
    res = fit_gp(sets, base, workers=settings.max_workers)
    summary = {**res.summary(), "days": days}
    write_json(out, summary)
    return summary


DAY_VARIANCES = ("log_sigma_field", "log_sigma_bias", "log_sigma_noise")


def fitted_field(
    settings: Settings,
    day: str,
    params: GPParams,
    method: str = "gp",
    use_cross: bool = False,
    day_variances: bool = True,
) -> CorrectedWind:
    """ERA5 + residual estimator built from the day's observations.

    With ``day_variances`` (GP only), the field, bias and noise variances are re-estimated by marginal likelihood on
    the day's own aircraft data (length scales unchanged): the residual amplitude varies from day to day and this keeps
    the posterior uncertainty calibrated (evaluation A, variant gp_matern_dayvar)."""
    obs = observation_set(settings, day, use_cross)
    if method == "gp" and day_variances and len(obs) >= 50:
        params = fit_gp([obs], params, only=DAY_VARIANCES, workers=settings.max_workers).params
    if method == "gp":
        est: ResidualEstimator = GPEstimator(
            obs, params, settings.gp.cutoff, settings.gp.tile_minutes, settings.gp.max_neighbours
        )
    else:
        est = build_estimator(obs, settings, method)
    return CorrectedWind(Era5.from_file(settings.paths.era5(day)), est, settings)


def export_step(
    settings: Settings,
    day: str,
    field: CorrectedWind,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    out: Path | None = None,
    attrs: dict[str, Any] | None = None,
) -> Path:
    """Gridded corrected wind (NetCDF, pycontrails-ready) over [start, stop] (+ configured padding)."""
    era = field.era5.ds
    grid = export_grid(era, settings.site, settings.export, start, stop)
    on_grid = era5_on_grid(era, grid)
    residual = residual_dataset(on_grid, field.estimator, settings.site)
    meta = {"day": day, "estimator": field.estimator.name, "skywind_version": __version__, **(attrs or {})}
    ds = to_met_dataset(on_grid, residual, meta)
    return write_netcdf(ds, out or settings.paths.fields(day))


def advect_step(
    settings: Settings,
    field_path: Path,
    points: pd.DataFrame,
    until: pd.Timestamp,
    winds: str = "corrected",
    max_age_min: float = 90.0,
) -> pd.DataFrame:
    """Advect emission points on an exported dataset (``winds`` = corrected | era5)."""
    ds = xr.open_dataset(field_path)
    return advect(points, met_dataset(ds, "era5" if winds == "era5" else "corrected"), until, max_age_min=max_age_min)


def cruise_emission_points(
    settings: Settings, day: str, start: pd.Timestamp, stop: pd.Timestamp, every_s: float = 30.0
) -> pd.DataFrame:
    """Default emission points of a day: cruise state vectors inside the site box during [start, stop]."""
    sv, _ = state_vectors(settings, day)
    site = settings.site
    inside = ((sv["latitude"] - site.lat).abs() <= site.box_deg) & ((sv["longitude"] - site.lon).abs() <= site.box_deg)
    sel = inside & (sv["altitude"] >= site.min_alt_ft) & sv["timestamp"].between(start, stop)
    sel &= (sv["t"] % every_s) < 1e-6
    return emission_points(sv[sel])
