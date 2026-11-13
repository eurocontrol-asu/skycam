"""OpenSky state vectors: loading, label-free sanitising of degraded extracts, track and speed from positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from skywind.geo import KT, R_EARTH

__all__ = ["SV_COLUMNS", "add_position_track", "load_state_vectors", "sanitise", "step_inconsistency"]

SV_COLUMNS = [
    "flight_id",
    "timestamp",
    "icao24",
    "typecode",
    "callsign",
    "latitude",
    "longitude",
    "altitude",
    "geoaltitude",
    "groundspeed",
]
_FLOATS = ("latitude", "longitude", "altitude", "geoaltitude", "groundspeed")
_STRINGS = ("flight_id", "callsign", "icao24", "typecode")


def load_state_vectors(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a daily OpenSky extract; plain float / object dtypes, ``t`` in epoch seconds, sorted by flight and time."""
    sv = pd.read_parquet(path, columns=columns or SV_COLUMNS)
    sv["timestamp"] = pd.to_datetime(sv["timestamp"].astype("int64"), utc=True)
    sv["t"] = sv["timestamp"].astype("int64").to_numpy() / 1e9
    for c in _FLOATS:
        if c in sv:
            sv[c] = sv[c].astype(float)
    for c in _STRINGS:
        if c in sv:
            sv[c] = sv[c].astype(object).where(sv[c].notna(), None)
    sv = sv.dropna(subset=["latitude", "longitude", "altitude"])
    return sv.sort_values(["flight_id", "t"]).reset_index(drop=True)


def _step_distance_m(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("flight_id", sort=False)
    la, lo = np.radians(df["latitude"]), np.radians(df["longitude"])
    dla = la - np.radians(g["latitude"].shift())
    dlo = (lo - np.radians(g["longitude"].shift())) * np.cos(la)
    return pd.Series(R_EARTH * np.hypot(dla, dlo), index=df.index)


def step_inconsistency(df: pd.DataFrame, rel_tol: float = 0.2) -> tuple[pd.Series, pd.Series]:
    """Per row: (5 s step present, step consistent with the reported ground speed within ``rel_tol``)."""
    dt = df.groupby("flight_id", sort=False)["t"].diff()
    speed = _step_distance_m(df) / dt
    has = dt == 5
    ok = has & ((speed / (df["groundspeed"] * KT) - 1).abs() <= rel_tol)
    return has, ok


def _reinterpolate(df: pd.DataFrame, good: npt.NDArray[np.bool_]) -> int:
    lat, lon, t = df["latitude"].to_numpy().copy(), df["longitude"].to_numpy().copy(), df["t"].to_numpy()
    fixed = 0
    for idx in df.groupby("flight_id", sort=False).indices.values():
        gd = good[idx]
        if gd.sum() >= 10 and (~gd).any():
            lat[idx[~gd]] = np.interp(t[idx[~gd]], t[idx[gd]], lat[idx[gd]])
            lon[idx[~gd]] = np.interp(t[idx[~gd]], t[idx[gd]], lon[idx[gd]])
            fixed += int((~gd).sum())
    df["latitude"], df["longitude"] = lat, lon
    return fixed


def _despike(df: pd.DataFrame, column: str, max_jump_ft: float = 500.0) -> int:
    a = df[column].astype(float)
    med = a.groupby(df["flight_id"], sort=False).transform(lambda x: x.rolling(5, center=True, min_periods=3).median())
    bad = (a - med).abs() > max_jump_ft
    df[column] = a.where(~bad, med)
    return int(bad.sum())


def sanitise(df: pd.DataFrame, max_bad_share: float = 0.10) -> tuple[pd.DataFrame, dict[str, float]]:
    """Label-free repair of a degraded extract (ported from the prototype ``osclean``).

    Applied only when more than ``max_bad_share`` of the 5 s steps contradict the reported ground speed (by more than
    20 %). Positions of inconsistent rows are then re-interpolated in time from the consistent ones, and altitude
    spikes (> 500 ft from a 5-point running median) are replaced. Requires ``flight_id``, ``t``, ``groundspeed``.
    """
    df = df.sort_values(["flight_id", "t"]).reset_index(drop=True)
    has, ok = step_inconsistency(df)
    rate = float(1 - ok[has].mean()) if has.any() else 0.0
    report = {"bad_step_share": rate, "repositioned": 0.0, "altitude_spikes": 0.0}
    if rate <= max_bad_share:
        return df, report
    ok_next = ok.groupby(df["flight_id"], sort=False).shift(-1)
    first = df.groupby("flight_id", sort=False).cumcount() == 0
    good = (ok | first) & ok_next.fillna(True).astype(bool) & ~(first & ok_next.isna())
    report["repositioned"] = float(_reinterpolate(df, good.to_numpy(dtype=bool)))
    report["altitude_spikes"] = float(sum(_despike(df, c) for c in ("altitude", "geoaltitude") if c in df))
    has, ok = step_inconsistency(df)
    report["bad_step_share_after"] = float(1 - ok[has].mean()) if has.any() else 0.0
    return df, report


def add_position_track(sv: pd.DataFrame, half_window: int = 3, max_span_s: float = 90.0) -> pd.DataFrame:
    """Track (deg) and ground speed (kt) from positions ``half_window`` samples before / after (centred), per flight."""
    g = sv.groupby("flight_id", sort=False)
    la0, la1 = g["latitude"].shift(half_window), g["latitude"].shift(-half_window)
    lo0, lo1 = g["longitude"].shift(half_window), g["longitude"].shift(-half_window)
    span = g["t"].shift(-half_window) - g["t"].shift(half_window)
    dn = np.radians(la1 - la0) * R_EARTH
    de = np.radians(lo1 - lo0) * R_EARTH * np.cos(np.radians(sv["latitude"]))
    sv = sv.copy()
    sv["trk_sv"] = np.degrees(np.arctan2(de, dn)) % 360
    sv["gs_sv_pos"] = np.hypot(de, dn) / span / KT
    sv.loc[span > max_span_s, ["trk_sv", "gs_sv_pos"]] = np.nan
    return sv
