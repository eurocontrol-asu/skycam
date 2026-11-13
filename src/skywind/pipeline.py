"""Per-day processing steps on the input store: decode -> observe -> fit -> export -> advect.

Each step reads its inputs from the configured paths, writes its output to the working-data cache and returns a small
summary. The command-line interface only calls these functions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from skywind.config import Settings
from skywind.decode import decode_payloads
from skywind.era5 import Era5
from skywind.identify import decode_day
from skywind.observations import add_residuals, bin_observations
from skywind.statevectors import add_position_track, load_state_vectors, sanitise

__all__ = ["decode_step", "load_observations", "observe_step", "state_vectors"]


def state_vectors(settings: Settings, day: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Sanitised state vectors of the day with position-derived track."""
    sv = load_state_vectors(settings.paths.opensky(day))
    sv, report = sanitise(sv)
    return add_position_track(sv), report


def decode_step(settings: Settings, day: str) -> dict[str, Any]:
    """Raw Comm-B -> identified BDS 5,0 / 6,0 messages with their nearest state vector."""
    paths = settings.paths
    raw = pd.read_parquet(paths.commb(day))
    decoded = decode_payloads(raw)
    sv, report = state_vectors(settings, day)
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
