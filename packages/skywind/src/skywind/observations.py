"""Wind observations from identified BDS 5,0 / 6,0 messages: one per (flight, 30 s bin), and residuals against ERA5.

wind = ground vector (BDS 5,0 ground speed + true track) - air vector (BDS 5,0 true airspeed + BDS 6,0 magnetic heading
+ IGRF declination). Quality filters: at least 2 messages of each register in the bin, wings level (|roll| <= 3 deg),
level-ish (|baro vertical rate| <= 1000 ft/min), heading, track and speeds steady inside the bin (rejects turns and
mis-timed messages).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from skywind.config import ObservationConfig
from skywind.declination import declination_deg
from skywind.era5 import Era5
from skywind.geo import FT, KT, circular_mean_deg, circular_sd_deg, isa_pressure_pa, unit_vectors

__all__ = ["add_residuals", "apply_heading_offsets", "bin_observations", "wind_from_vectors"]


def _sin_cos(m: pd.DataFrame, col: str, name: str) -> None:
    r = np.radians(m[col].to_numpy(dtype=float))
    m[name + "_s"], m[name + "_c"] = np.sin(r), np.cos(r)


def _aggregate(m: pd.DataFrame) -> pd.DataFrame:
    key = ["flight_id", "b"]
    is50, is60 = m["bds"] == "50", m["bds"] == "60"
    a = (
        m[is50]
        .groupby(key)
        .agg(
            n50=("t", "size"),
            gs=("gs50", "median"),
            tas=("tas50", "median"),
            roll=("abs_roll", "max"),
            trk_s=("trk_s", "mean"),
            trk_c=("trk_c", "mean"),
            gs_sd=("gs50", "std"),
            tas_sd=("tas50", "std"),
        )
    )
    b = (
        m[is60]
        .groupby(key)
        .agg(
            n60=("t", "size"),
            mach=("mach60", "median"),
            ias=("ias60", "median"),
            hdg_s=("hdg_s", "mean"),
            hdg_c=("hdg_c", "mean"),
            vrb=("abs_vrb", "max"),
        )
    )
    c = m.groupby(key).agg(
        t=("t", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        altitude=("altitude", "median"),
        geoaltitude=("geoaltitude", "median"),
        icao24=("icao24", "first"),
        callsign=("callsign", "first"),
        typecode=("typecode", "first"),
        gs_sv=("groundspeed", "median"),
        sv_s=("sv_s", "mean"),
        sv_c=("sv_c", "mean"),
    )
    return a.join(b, how="inner").join(c, how="inner").reset_index()


def _filter(w: pd.DataFrame, cfg: ObservationConfig) -> pd.DataFrame:
    w = w.assign(trk_sd=circular_sd_deg(w["trk_s"], w["trk_c"]), hdg_sd=circular_sd_deg(w["hdg_s"], w["hdg_c"]))
    keep = (
        (w["n50"] >= cfg.min_msgs_per_register)
        & (w["n60"] >= cfg.min_msgs_per_register)
        & (w["roll"] <= cfg.max_roll_deg)
        & (w["vrb"] <= cfg.max_vrate_fpm)
        & (w["trk_sd"] <= cfg.max_dir_sd_deg)
        & (w["hdg_sd"] <= cfg.max_dir_sd_deg)
        & (w["gs_sd"].fillna(0) <= cfg.max_speed_sd_kt)
        & (w["tas_sd"].fillna(0) <= cfg.max_speed_sd_kt)
    )
    return w[keep].copy()


def wind_from_vectors(gs_kt: object, trk_deg: object, tas_kt: object, hdg_deg: object) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) wind in m/s = ground vector - air vector."""
    tr, hd = np.radians(np.asarray(trk_deg, float)), np.radians(np.asarray(hdg_deg, float))
    gs, tas = np.asarray(gs_kt, float), np.asarray(tas_kt, float)
    return KT * (gs * np.sin(tr) - tas * np.sin(hd)), KT * (gs * np.cos(tr) - tas * np.cos(hd))


def bin_observations(msgs: pd.DataFrame, day: str, cfg: ObservationConfig, min_alt_ft: float) -> pd.DataFrame:
    """One wind observation per (flight, bin) from identified messages (:func:`skywind.identify.decode_day`)."""
    m = msgs[msgs["altitude"] >= min_alt_ft].copy()
    m["b"] = np.floor(m["t"] / cfg.bin_s).astype("int64")
    m["abs_roll"], m["abs_vrb"] = m["roll"].abs(), m["vrb60"].abs()
    _sin_cos(m, "trk50", "trk")
    _sin_cos(m, "hdg60", "hdg")
    _sin_cos(m, "trk_sv", "sv")
    w = _filter(_aggregate(m), cfg)
    w["trk"] = circular_mean_deg(w["trk_s"], w["trk_c"])
    w["trk_sv"] = circular_mean_deg(w["sv_s"], w["sv_c"])
    w["hdg_mag"] = circular_mean_deg(w["hdg_s"], w["hdg_c"])
    when = datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 12, tzinfo=UTC).replace(tzinfo=None)
    w["decl"] = declination_deg(w["latitude"], w["longitude"], w["altitude"] * FT / 1000, when)
    w["hdg_offset"] = 0.0
    w["hdg"] = (w["hdg_mag"] + w["decl"]) % 360
    w["p_hpa"] = isa_pressure_pa(w["altitude"].to_numpy() * FT) / 100
    w["u_obs"], w["v_obs"] = wind_from_vectors(w["gs"], w["trk"], w["tas"], w["hdg"])
    w["day"] = day
    drop = ["trk_s", "trk_c", "sv_s", "sv_c", "hdg_s", "hdg_c"]
    return w.drop(columns=drop).reset_index(drop=True)


def _decompose(w: pd.DataFrame) -> pd.DataFrame:
    hx, hy, qx, qy = unit_vectors(w["hdg"])
    w["du"], w["dv"] = w["u_obs"] - w["u_era"], w["v_obs"] - w["v_era"]
    w["d_along"] = w["du"] * hx + w["dv"] * hy
    w["d_right"] = w["du"] * qx + w["dv"] * qy
    return w


def add_residuals(w: pd.DataFrame, era: Era5) -> pd.DataFrame:
    """ERA5 at the observation (time, pressure from barometric altitude, lat, lon) and the residual decomposed along /
    to the right of the heading. Observations outside the ERA5 domain are dropped."""
    w = w.copy()
    args = (w["t"], w["p_hpa"], w["latitude"], w["longitude"])
    w["u_era"], w["v_era"] = era.wind(*args)
    w["T_era"] = era("t", *args)
    w = _decompose(w)
    w["tas_mach"] = w["mach"] * np.sqrt(1.4 * 287.053 * w["T_era"]) / KT
    return w.dropna(subset=["u_era", "v_era"]).reset_index(drop=True)


def apply_heading_offsets(w: pd.DataFrame, offsets_deg: pd.Series) -> pd.DataFrame:
    """Correct the true heading by a per-airframe offset (deg, indexed by icao24; missing -> NaN) and recompute the wind
    and residuals. Rows of airframes without an offset get ``hdg_offset`` NaN (cross-heading unusable)."""
    w = w.copy()
    off = w["icao24"].map(offsets_deg).astype(float)
    w["hdg_offset"] = off
    w["hdg"] = (w["hdg_mag"] + w["decl"] + off.fillna(0.0)) % 360
    w["u_obs"], w["v_obs"] = wind_from_vectors(w["gs"], w["trk"], w["tas"], w["hdg"])
    return _decompose(w)
