"""Command-line interface. Every command is a thin wrapper around a library function."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import cyclopts

from skywind import _threads  # noqa: F401  (caps BLAS threads before numpy is imported)
from skywind.config import Settings, load_settings
from skywind.days import day_info, parse_day, parse_windows

__all__ = ["app", "main"]

app = cyclopts.App(name="skywind", help="ERA5 wind corrected with Mode S Comm-B winds, above a ground camera.")
evaluate_app = cyclopts.App(name="evaluate", help="Evaluations A (aircraft) and B (contrails).")
app.command(evaluate_app)

Config = Annotated[Path | None, cyclopts.Parameter(help="extra TOML configuration file")]
Days = Annotated[list[str] | None, cyclopts.Parameter(consume_multiple=True, help="days (default: all available)")]


def _settings(config: Path | None) -> Settings:
    return load_settings(config)


def _print(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")


def _fail(exc: Exception) -> None:
    sys.stderr.write(f"error: {exc}\n")
    raise SystemExit(1)


@app.command
def days(set_name: str | None = None, config: Config = None) -> None:
    """List the days available in the input store with their windows and set (dev / holdout)."""
    from skywind.days import available_days

    s = _settings(config)
    infos = [day_info(s.paths, d) for d in available_days(s.paths, set_name)]
    _print([{"day": i.day, "set": i.set, "windows": [(str(a), str(b)) for a, b in i.windows]} for i in infos])


@app.command
def acquire(day: str, windows: str, overwrite: bool = False, config: Config = None) -> None:
    """Fetch state vectors, ERA5 and Comm-B of a new DAY for WINDOWS ('HH:MM-HH:MM[;...]') into the input store."""
    from skywind.acquire import acquire_day, default_file_sources, register_day
    from skywind.sources.opensky_trino import TrinoCommB

    s = _settings(config)
    d = parse_day(day)
    try:
        rep = acquire_day(
            s,
            d,
            parse_windows(d, windows),
            default_file_sources(s),
            TrinoCommB(s.paths.cache / "opensky_query_cache"),
            overwrite,
        )
    except Exception as exc:
        _fail(exc)
    rep["registered"] = register_day(s, d, parse_windows(d, windows))
    _print(rep)


@app.command
def decode(*day: str, config: Config = None) -> None:
    """Decode and identify BDS 5,0 / 6,0 messages of each DAY."""
    from skywind.pipeline import decode_step

    s = _settings(config)
    _print([decode_step(s, parse_day(d)) for d in day])


@app.command
def observe(*day: str, config: Config = None) -> None:
    """Build the 30 s wind observations and their ERA5 residuals for each DAY."""
    from skywind.pipeline import observe_step

    s = _settings(config)
    _print([observe_step(s, parse_day(d)) for d in day])


@app.command
def calibrate(*day: str, params: Path | None = None, reference: str = "field", config: Config = None) -> None:
    """Add each DAY to the persistent per-airframe heading-offset table."""
    from skywind.pipeline import calibrate_step, gp_params

    s = _settings(config)
    p = gp_params(s, params)
    _print([calibrate_step(s, parse_day(d), p, reference) for d in day])


@app.command
def fit(*day: str, out: Path, cross: bool = False, config: Config = None) -> None:
    """Fit the GP hyperparameters (marginal likelihood) on the DAYs; write them as JSON to OUT."""
    from skywind.pipeline import fit_step, gp_params

    s = _settings(config)
    _print(fit_step(s, [parse_day(d) for d in day], gp_params(s), out, cross))


@app.command
def export(
    day: str,
    method: str = "gp",
    params: Path | None = None,
    start: str | None = None,
    stop: str | None = None,
    out: Path | None = None,
    cross: bool = False,
    config: Config = None,
) -> None:
    """Export the corrected wind of DAY as a pycontrails-ready NetCDF (default window: the day's windows)."""
    import pandas as pd

    from skywind.pipeline import export_step, fitted_field, gp_params

    s = _settings(config)
    d = parse_day(day)
    info = day_info(s.paths, d)
    field = fitted_field(s, d, gp_params(s, params), method, cross)
    t0 = pd.Timestamp(start, tz="UTC") if start else info.start
    t1 = pd.Timestamp(stop, tz="UTC") if stop else info.stop
    _print({"output": str(export_step(s, d, field, t0, t1, out))})


@app.command
def advect(
    day: str,
    field: Path | None = None,
    points: Path | None = None,
    winds: str = "corrected",
    out: Path | None = None,
    max_age_min: float = 40.0,
    config: Config = None,
) -> None:
    """Advect emission points (default: cruise state vectors in the site box) on an exported field."""
    import pandas as pd

    from skywind.pipeline import advect_step, cruise_emission_points

    s = _settings(config)
    d = parse_day(day)
    info = day_info(s.paths, d)
    pts = pd.read_parquet(points) if points else cruise_emission_points(s, d, info.start, info.stop)
    adv = advect_step(
        s, field or s.paths.fields(d), pts, info.stop + pd.Timedelta(minutes=max_age_min), winds, max_age_min
    )
    dest = out or s.paths.advected(d, winds)
    dest.parent.mkdir(parents=True, exist_ok=True)
    adv.to_parquet(dest, index=False)
    _print({"points": len(pts), "rows": len(adv), "output": str(dest)})


@app.command
def run(
    day: str,
    windows: str | None = None,
    params: Path | None = None,
    acquire_missing: bool = True,
    config: Config = None,
) -> None:
    """End-to-end for one DAY: (acquire) -> decode -> observe -> export -> advect (ERA5 and corrected)."""
    from skywind.workflow import run_day

    s = _settings(config)
    try:
        _print(run_day(s, parse_day(day), windows, params, acquire_missing))
    except Exception as exc:
        _fail(exc)


@evaluate_app.command
def aircraft(
    *variant: str,
    days: Days = None,
    ridge: bool = True,
    diagnostics: bool = True,
    transfer: bool = True,
    config: Config = None,
) -> None:
    """Evaluation A: leave-one-aircraft-out skill with leave-one-day-out hyperparameters, per VARIANT (default gp)."""
    from skywind.evaluate.campaign import aircraft_campaign

    s = _settings(config)
    _print(aircraft_campaign(s, list(variant) or ["gp"], days, ridge, diagnostics, transfer))


@evaluate_app.command
def contrails(
    *variant: str,
    aircraft_runs: Annotated[list[Path], cyclopts.Parameter(consume_multiple=True)],
    days: Days = None,
    config: Config = None,
) -> None:
    """Evaluation B: advection vs the GVCCS calibration artefacts, with the LODO fits of the evaluation A runs."""
    from skywind.evaluate.campaign import contrail_campaign

    s = _settings(config)
    _print(contrail_campaign(s, aircraft_runs, list(variant) or ["gp"], days))


@evaluate_app.command
def diagnostics(params: Path | None = None, days: Days = None, config: Config = None) -> None:
    """Hollingsworth-Lonnberg covariances and Desroziers ratios for the GP hyperparameters (default: configured)."""
    from skywind.evaluate.campaign import diagnostics_campaign
    from skywind.pipeline import gp_params

    s = _settings(config)
    _print(diagnostics_campaign(s, gp_params(s, params), days))


@evaluate_app.command
def report(aircraft_run: Path | None = None, contrail_run: Path | None = None) -> None:
    """Markdown tables of an evaluation A and / or B run directory."""
    from skywind.evaluate.report import aircraft_report, contrail_report

    parts = [aircraft_report(aircraft_run)] if aircraft_run else []
    parts += [contrail_report(contrail_run)] if contrail_run else []
    sys.stdout.write("\n".join(parts) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    app()


if __name__ == "__main__":
    main()
