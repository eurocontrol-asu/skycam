"""Per-airframe heading offset calibration (option for the cross-heading observations).

The cross-heading residual is dominated by a per-airframe constant heading offset (each aircraft flies its own magnetic
declination table epoch; EMADDC, de Haan et al. 2025). As in EMADDC, the offset h_c of an airframe minimises
sum_i [1 - cos(h_ref,i - h_mag,i - decl_i - h_c)], where h_ref is the heading implied by a reference wind (ground vector
minus reference wind): the closed form is the circular mean of h_ref - (h_mag + decl). Pooling many days makes the
leak of reference-wind errors into the offset small.

The table is persistent and accumulates over days: one row per (airframe, day) with the sufficient statistics
(n, sum of sin, sum of cos), the reference used and the library version. Offsets are combined at use time, optionally
excluding days (leave-day-out evaluation).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from skywind.geo import KT, angle_diff_deg, circular_mean_deg, circular_sd_deg

__all__ = [
    "TABLE_VERSION",
    "combine_offsets",
    "cross_noise_terms",
    "day_to_day_scatter_deg",
    "offset_terms",
    "update_table",
]

TABLE_VERSION = "skywind-heading-offsets/1"


def offset_terms(w: pd.DataFrame, u_ref: np.ndarray, v_ref: np.ndarray) -> pd.DataFrame:
    """Per airframe of one day: n, sum sin / cos of (reference heading - IGRF true heading)."""
    tr = np.radians(w["trk"].to_numpy(float))
    gs = w["gs"].to_numpy(float) * KT
    air_u, air_v = gs * np.sin(tr) - u_ref, gs * np.cos(tr) - v_ref
    h_ref = np.degrees(np.arctan2(air_u, air_v))
    phi = np.radians(angle_diff_deg(h_ref, w["hdg_mag"].to_numpy(float) + w["decl"].to_numpy(float)))
    d = pd.DataFrame(
        {
            "icao24": w["icao24"].astype(str).to_numpy(),
            "typecode": w["typecode"].astype(str).to_numpy(),
            "s": np.sin(phi),
            "c": np.cos(phi),
        }
    )
    d = d[np.isfinite(d["s"])]
    out = d.groupby("icao24").agg(
        n=("s", "size"), sum_sin=("s", "sum"), sum_cos=("c", "sum"), typecode=("typecode", "first")
    )
    result: pd.DataFrame = out.reset_index()
    return result


def update_table(path: Path, day: str, terms: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Insert or replace the rows of ``day`` in the persistent table and return the whole table."""
    new = terms.assign(
        day=day, reference=reference, version=TABLE_VERSION, computed_at=datetime.now(UTC).isoformat(timespec="seconds")
    )
    old = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=new.columns)
    table = pd.concat([old[old["day"].astype(str) != day], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def combine_offsets(
    table: pd.DataFrame, exclude_days: tuple[str, ...] = (), min_obs: int = 20, min_days: int = 1
) -> pd.DataFrame:
    """Offset per airframe (deg, to add to the IGRF true heading) from the days not excluded, with n and day span."""
    t = table[~table["day"].astype(str).isin(exclude_days)]
    if t.empty:
        return pd.DataFrame(columns=["offset_deg", "sd_deg", "n_obs", "n_days", "first_day", "last_day"])
    g = t.groupby("icao24").agg(
        n_obs=("n", "sum"),
        ss=("sum_sin", "sum"),
        sc=("sum_cos", "sum"),
        n_days=("day", "nunique"),
        first_day=("day", "min"),
        last_day=("day", "max"),
    )
    g = g[(g["n_obs"] >= min_obs) & (g["n_days"] >= min_days)]
    off = angle_diff_deg(circular_mean_deg(g["ss"] / g["n_obs"], g["sc"] / g["n_obs"]), 0.0)
    sd = circular_sd_deg(g["ss"] / g["n_obs"], g["sc"] / g["n_obs"])
    return g.assign(offset_deg=off, sd_deg=sd)[["offset_deg", "sd_deg", "n_obs", "n_days", "first_day", "last_day"]]


HEADING_RESOLUTION_DEG = 0.352  # BDS 6,0 magnetic heading resolution (90 / 256 deg)


def day_to_day_scatter_deg(table: pd.DataFrame, exclude_days: tuple[str, ...] = (), min_obs_day: int = 10) -> float:
    """Robust sd (deg) of the per-day offset of an airframe around its own multi-day mean, pooled over airframes seen
    on at least two days with ``min_obs_day`` observations each (1.4826 x median absolute deviation)."""
    t = table[~table["day"].astype(str).isin(exclude_days) & (table["n"] >= min_obs_day)]
    t = t.assign(off=angle_diff_deg(circular_mean_deg(t["sum_sin"] / t["n"], t["sum_cos"] / t["n"]), 0.0))
    multi = t[t.groupby("icao24")["day"].transform("nunique") >= 2]
    dev = multi["off"] - multi.groupby("icao24")["off"].transform("mean")
    n_days = multi.groupby("icao24")["day"].transform("nunique")
    dev = dev * np.sqrt(n_days / (n_days - 1))  # unbiased for the deviation from a mean of n days
    return float(1.4826 * np.median(np.abs(dev)))


def cross_noise_terms(
    scatter_deg: float, median_days: float, sigma_noise: float, tas_ms: float = 230.0
) -> tuple[float, float]:
    """Fixed (not fitted) noise of the across-heading observations, in m/s.

    Per-aircraft bias: error of the offset applied on a new day = its day-to-day change plus the error of the
    multi-day mean, sd x sqrt(1 + 1 / n_days). Per-observation noise: the along-heading noise combined with the heading
    quantisation (resolution / sqrt(12))."""
    bias = tas_ms * np.radians(scatter_deg * np.sqrt(1 + 1 / max(median_days, 1.0)))
    quant = tas_ms * np.radians(HEADING_RESOLUTION_DEG / np.sqrt(12))
    return float(np.hypot(sigma_noise, quant)), float(bias)
