"""Evaluation B: advected emission points against annotated contrail centres (cross-track distance and drift).

The scoring is independent of where the annotations come from: a :class:`TruthSource` yields, per day, the emission
points to advect and the annotated contrail centre (geographic position + unit right-of-track normal) for each
(emission point, frame time). The study-folder artefacts are one source (:mod:`skywind.evaluate.gvccs_calib`); a source
reading the raw GVCCS annotations can be plugged in the same way.

For a method M (ERA5, corrected...), the signed cross-track error of a point is e = (annotated centre - advected
position) . normal (m, positive = contrail to the right). Per flight, a straight line e(age) = offset + drift * age is
fitted to the per-minute medians (ages 0.5-40 min): ``drift`` (m/min) is the wind effect, ``offset`` (m) the error at
emission, which is not a wind effect (altitude / wake / anchoring) and is reported separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from skywind.config import Site
from skywind.geo import local_xy_km

__all__ = ["AGE_BINS", "ContrailTruth", "TruthSource", "age_table", "cross_track_errors", "drift_table", "flight_fits"]

AGE_BINS = (0.5, 5.0, 10.0, 20.0, 40.0)


@dataclass(frozen=True)
class ContrailTruth:
    """``emissions``: flight, flight_id, waypoint, emission_time, latitude, longitude, level (hPa).
    ``centres``: flight, flight_id, waypoint, time, age_min, centre_lat, centre_lon, normal_e, normal_n."""

    day: str
    emissions: pd.DataFrame
    centres: pd.DataFrame

    @property
    def last_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.centres["time"].max())


class TruthSource(Protocol):
    def days(self) -> list[str]: ...

    def load(self, day: str) -> ContrailTruth: ...


def cross_track_errors(truth: ContrailTruth, advected: pd.DataFrame, site: Site) -> pd.DataFrame:
    """Signed cross-track error (m) of every annotated (point, frame) found in ``advected``."""
    m = truth.centres.merge(
        advected[["flight_id", "waypoint", "time", "latitude", "longitude"]],
        on=["flight_id", "waypoint", "time"],
        how="inner",
    )
    cx, cy = local_xy_km(m["centre_lat"], m["centre_lon"], site.lat, site.lon)
    px, py = local_xy_km(m["latitude"], m["longitude"], site.lat, site.lon)
    m["error_m"] = 1000 * ((cx - px) * m["normal_e"] + (cy - py) * m["normal_n"])
    m["day"] = truth.day
    return m[["day", "flight", "flight_id", "waypoint", "time", "age_min", "error_m"]]


def _line(g: pd.DataFrame, min_ages: int) -> pd.Series:
    med = g.groupby(g["age_min"].round())["error_m"].median()
    if len(med) < min_ages:
        return pd.Series({"offset_m": np.nan, "drift_m_per_min": np.nan, "n_points": len(g), "n_ages": len(med)})
    slope, intercept = np.polyfit(med.index.to_numpy(float), med.to_numpy(float), 1)
    return pd.Series({"offset_m": intercept, "drift_m_per_min": slope, "n_points": len(g), "n_ages": len(med)})


def flight_fits(errors: pd.DataFrame, min_ages: int = 8, max_age: float = 40.0) -> pd.DataFrame:
    """Per flight: offset at emission (m) and drift (m/min) of the cross-track error."""
    e = errors[(errors["age_min"] >= AGE_BINS[0]) & (errors["age_min"] <= max_age)]
    rows = [{"day": day, "flight": fl, **_line(g, min_ages).to_dict()} for (day, fl), g in e.groupby(["day", "flight"])]
    return pd.DataFrame(rows)


def _drift_summary(d: pd.DataFrame, methods: list[str], ref: str) -> dict[str, float]:
    out: dict[str, float] = {"n_flights": float(len(d))}
    for m in methods:
        a = d[f"drift_{m}"].abs()
        out[f"drift_med_{m}"], out[f"drift_p90_{m}"] = float(a.median()), float(a.quantile(0.9))
        out[f"offset_med_{m}"] = float(d[f"offset_{m}"].median())
        out[f"offset_abs_med_{m}"] = float(d[f"offset_{m}"].abs().median())
        if m != ref:
            out[f"improved_{m}"] = float((a < d[f"drift_{ref}"].abs()).mean())
    return out


def drift_table(fits: Mapping[str, pd.DataFrame], ref: str = "era5") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired per-flight table (flights fitted by every method) and its summary per day and pooled."""
    methods = list(fits)
    paired: pd.DataFrame | None = None
    for m, f in fits.items():
        part = f.rename(columns={"offset_m": f"offset_{m}", "drift_m_per_min": f"drift_{m}"})
        part = part[["day", "flight", f"offset_{m}", f"drift_{m}"]]
        paired = part if paired is None else paired.merge(part, on=["day", "flight"])
    if paired is None:
        raise ValueError("no method to compare")
    paired = paired.dropna()
    rows = [{"day": day, **_drift_summary(g, methods, ref)} for day, g in paired.groupby("day")]
    rows.append({"day": "pooled", **_drift_summary(paired, methods, ref)})
    return paired, pd.DataFrame(rows)


def age_table(errors: Mapping[str, pd.DataFrame], fits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Median and p90 of |cross-track error| by age bin, raw and after removing each flight's own offset."""
    rows = []
    for m, e in errors.items():
        f = fits[m][["day", "flight", "offset_m"]]
        e = e.merge(f, on=["day", "flight"], how="inner").dropna(subset=["offset_m"])
        e = e.assign(
            bin=pd.cut(e["age_min"], list(AGE_BINS)),
            raw=e["error_m"].abs(),
            drift_only=(e["error_m"] - e["offset_m"]).abs(),
        )
        for (day, b), g in pd.concat([e, e.assign(day="pooled")]).groupby(["day", "bin"], observed=True):
            rows.append(
                {
                    "method": m,
                    "day": day,
                    "age_bin": str(b),
                    "n_points": len(g),
                    "n_flights": g["flight"].nunique(),
                    "abs_med": g["raw"].median(),
                    "abs_p90": g["raw"].quantile(0.9),
                    "drift_only_med": g["drift_only"].median(),
                    "drift_only_p90": g["drift_only"].quantile(0.9),
                }
            )
    return pd.DataFrame(rows)


def paired_tests(paired: pd.DataFrame, ref: str = "era5") -> pd.DataFrame:
    """Per day and pooled, for each method: Wilcoxon signed-rank and sign tests of |drift_ref| - |drift_method| > 0
    (one-sided: the method reduces the drift error)."""
    from scipy.stats import binomtest, wilcoxon

    methods = [c.removeprefix("drift_") for c in paired.columns if c.startswith("drift_") and c != f"drift_{ref}"]
    rows = []
    for day, g in [*paired.groupby("day"), ("pooled", paired)]:
        for m in methods:
            d = (g[f"drift_{ref}"].abs() - g[f"drift_{m}"].abs()).to_numpy()
            d = d[np.isfinite(d) & (d != 0)]
            n_pos = int((d > 0).sum())
            w = wilcoxon(d, alternative="greater").pvalue if len(d) >= 5 else np.nan
            b = binomtest(n_pos, len(d), 0.5, alternative="greater").pvalue if len(d) else np.nan
            rows.append(
                {
                    "day": day,
                    "method": m,
                    "n_flights": len(d),
                    "n_improved": n_pos,
                    "median_reduction_m_per_min": float(np.median(d)) if len(d) else np.nan,
                    "wilcoxon_p": float(w),
                    "sign_test_p": float(b),
                }
            )
    return pd.DataFrame(rows)
