"""Contrail truth from the calibration artefacts of the prototype study (six days, GVCCS train split only).

Artefacts used (read-only), under ``<prototype_dir>/calib``:
  ``flights/matches_<day>.csv``         GVCCS flight -> OpenSky flight;
  ``dataset/meta/<day>_<flight>.json``  emission points in view (waypoint, emission time, ribbon row) and frames;
  ``segments/<day>_<flight>.npz``       per point and frame: presence, centre (ribbon pixel, 25 m, row 160 = ribbon
                                         axis, decreasing to the right of track), interior gaps, age;
  ``flights/advected_<day>_dz777.parquet`` the path the ribbon was drawn on (axis positions per waypoint and frame).
The annotated centre is placed back on the ground: axis position + offset x right-of-track normal of the axis
(tangent over +-500 m of emission arc length, as when the ribbons were built). Emission positions come from the input
store (sanitised state vectors), never from the annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skywind.config import Settings
from skywind.evaluate.contrails import ContrailTruth
from skywind.geo import FT, offset_latlon
from skywind.pipeline import state_vectors

__all__ = ["CalibArtefacts", "ribbon_normals"]

_RES_M, _AXIS_ROW, _HALF_TANGENT_M = 25.0, 160.0, 500.0
_EXCLUDED = {"34693e4f"}  # reference ribbon of the prototype, not a target


def _metres(lat: np.ndarray, lon: np.ndarray, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.pi / 180 * 6371e3
    return lon * np.cos(np.radians(lat0)) * k, lat * k


def ribbon_normals(s: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit right-of-track normal at each node of a polyline parameterised by emission arc length ``s`` (sorted)."""
    lo, hi = np.maximum(s - _HALF_TANGENT_M, s[0]), np.minimum(s + _HALF_TANGENT_M, s[-1])
    tx, ty = np.interp(hi, s, x) - np.interp(lo, s, x), np.interp(hi, s, y) - np.interp(lo, s, y)
    n = np.hypot(tx, ty)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 50, ty / n, np.nan), np.where(n > 50, -tx / n, np.nan)


class CalibArtefacts:
    """:class:`~skywind.evaluate.contrails.TruthSource` over the prototype calibration artefacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.paths.prototype_dir / "calib"

    def days(self) -> list[str]:
        return sorted({p.name[:8] for p in (self.root / "segments").glob("*.npz")})

    def _flights(self, day: str) -> list[tuple[dict[str, Any], Path]]:
        out = []
        for meta_path in sorted((self.root / "dataset" / "meta").glob(f"{day}_*.json")):
            seg = self.root / "segments" / meta_path.with_suffix(".npz").name
            meta = json.loads(meta_path.read_text())
            if seg.is_file() and meta["flight"] not in _EXCLUDED:
                out.append((meta, seg))
        return out

    def _emissions(self, day: str, metas: list[dict[str, Any]]) -> pd.DataFrame:
        from pycontrails.physics import units

        pts = pd.DataFrame(
            [
                {
                    "flight": m["flight"],
                    "flight_id": m["os_flight"],
                    "waypoint": p["waypoint"],
                    "emission_time": pd.Timestamp(p["emission"], tz="UTC"),
                }
                for m in metas
                for p in m["points"]
            ]
        )
        pts["emission_time"] = pd.to_datetime(pts["emission_time"], utc=True).dt.as_unit("ns")
        sv, _ = state_vectors(self.settings, day)
        sv = sv[sv["flight_id"].isin(set(pts["flight_id"]))]
        sv = sv[["flight_id", "timestamp", "latitude", "longitude", "altitude"]].rename(
            columns={"timestamp": "emission_time"}
        )
        sv["flight_id"] = sv["flight_id"].astype(str)
        sv["emission_time"] = pd.to_datetime(sv["emission_time"], utc=True).dt.as_unit("ns")
        e = pts.merge(sv, on=["flight_id", "emission_time"], how="inner")
        e["level"] = units.m_to_pl(e["altitude"].to_numpy(float) * FT)
        return e.drop(columns=["altitude"])

    def _axis(self, day: str, emissions: pd.DataFrame) -> pd.DataFrame:
        cols = ["flight_id", "waypoint", "time", "latitude", "longitude"]
        a = pd.read_parquet(self.root / "flights" / f"advected_{day}_dz777.parquet", columns=cols)
        a["flight_id"] = a["flight_id"].astype(str)
        a["time"] = pd.to_datetime(a["time"], utc=True).dt.as_unit("ns")
        return a[a["waypoint"].isin(set(emissions["waypoint"]))]

    def load(self, day: str) -> ContrailTruth:
        flights = self._flights(day)
        emissions = self._emissions(day, [m for m, _ in flights])
        axis = self._axis(day, emissions)
        frames = [self._centres(day, meta, seg, emissions, axis) for meta, seg in flights]
        centres = pd.concat([f for f in frames if len(f)], ignore_index=True)
        return ContrailTruth(day, emissions, centres)

    def _centres(
        self, day: str, meta: dict[str, Any], seg: Path, emissions: pd.DataFrame, axis: pd.DataFrame
    ) -> pd.DataFrame:
        z = np.load(seg)
        pts = sorted(meta["points"], key=lambda p: p["row"])
        waypoints = np.array([p["waypoint"] for p in pts])
        times = pd.to_datetime([day + f for f in z["frames"]], format="%Y%m%d%H%M%S").tz_localize("UTC").as_unit("ns")
        ok = z["present"] & ~z["gap"] & np.isfinite(z["centre"])
        k, t = np.nonzero(ok)
        obs = pd.DataFrame(
            {
                "waypoint": waypoints[k],
                "time": times[t],
                "age_min": z["age"][k, t].astype(float),
                "offset_m": (_AXIS_ROW - z["centre"][k, t].astype(float)) * _RES_M,
            }
        )
        e = emissions[emissions["flight"] == meta["flight"]]
        lat0 = self.settings.site.lat
        ex, ey = _metres(e["latitude"].to_numpy(), e["longitude"].to_numpy(), lat0)
        s_map = pd.Series(np.r_[0, np.cumsum(np.hypot(np.diff(ex), np.diff(ey)))], index=e["waypoint"].to_numpy())
        ax = axis[axis["waypoint"].isin(s_map.index) & axis["time"].isin(set(obs["time"]))].copy()
        ax["s"] = ax["waypoint"].map(s_map)
        ax = ax.sort_values(["time", "s"])
        ax["x"], ax["y"] = _metres(ax["latitude"].to_numpy(), ax["longitude"].to_numpy(), lat0)
        normals = [self._frame_normals(g) for _, g in ax.groupby("time", sort=False)]
        if not normals:
            return pd.DataFrame()
        ax = pd.concat(normals)
        out = obs.merge(
            ax[["waypoint", "time", "latitude", "longitude", "normal_e", "normal_n"]], on=["waypoint", "time"]
        )
        out = out.dropna(subset=["normal_e"])
        out["centre_lat"], out["centre_lon"] = offset_latlon(
            out["latitude"], out["longitude"], out["offset_m"] * out["normal_e"], out["offset_m"] * out["normal_n"]
        )
        out["flight"], out["flight_id"] = str(meta["flight"]), str(meta["os_flight"])
        return out.drop(columns=["latitude", "longitude"])

    @staticmethod
    def _frame_normals(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < 2:
            return g.assign(normal_e=np.nan, normal_n=np.nan)
        ne, nn = ribbon_normals(g["s"].to_numpy(float), g["x"].to_numpy(float), g["y"].to_numpy(float))
        return g.assign(normal_e=ne, normal_n=nn)
