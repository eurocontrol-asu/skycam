"""Contrail truth for any GVCCS day, from the raw dataset (train split): annotations, Flightradar24 tracks and the
camera geometry. No dependency on the prototype study folder.

Per day:
1. each GVCCS flight with an attributed contrail is matched to an OpenSky flight (same timestamps, median distance
   < 2 km over >= 12 samples, as in the prototype);
2. its emission points are the (sanitised) OpenSky state vectors from 45 min before the first frame to the last frame,
   at 150-400 hPa (pressure from barometric altitude);
3. they are advected with ERA5 (reference line, fixes the geometry; any method is scored against the same truth);
4. for each frame where the flight's contrail is annotated, and each advected point (every ``stride`` waypoints, age
   0.5-40 min), a line is drawn through the point along the right-of-track normal of the advected line (tangent over
   +-500 m of emission arc length), +-4 km in 25 m steps, at the point's geometric altitude (pressure altitude +
   the flight's GNSS - barometric altitude difference). Each sample is projected into the image (camera geometry of
   ``azimuth_zenith_grid.npy``) and looked up in the flight's annotation mask; the contiguous annotated run nearest to
   the point gives the annotated centre (mean offset of the run). Lines with less than half their samples inside the
   image (zenith <= 80 deg) are skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from skywind.advect import advect, emission_points, met_dataset
from skywind.config import Settings
from skywind.days import available_days
from skywind.era5 import load_era5
from skywind.evaluate.contrails import ContrailTruth
from skywind.geo import FT, R_EARTH, isa_altitude_m, offset_latlon
from skywind.gvccs.camera import Camera
from skywind.gvccs.dataset import GvccsDay, load_day, train_days
from skywind.pipeline import state_vectors

__all__ = ["GvccsRaw", "annotation_mask", "match_flights"]

_OFFSETS = np.arange(-4000.0, 4000.1, 25.0)
_HALF_TANGENT_M = 500.0
_MAX_ZENITH = np.radians(80.0)


def _haversine(la1: Any, lo1: Any, la2: Any, lo2: Any) -> Any:
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(h))


def match_flights(
    tracks: pd.DataFrame, sv: pd.DataFrame, flights: list[str], max_median_m: float = 2000.0
) -> pd.DataFrame:
    """GVCCS flight -> OpenSky flight_id by position at identical timestamps (median distance, >= 12 samples)."""
    rows = []
    osv = sv[["timestamp", "flight_id", "callsign", "latitude", "longitude"]]
    for f in flights:
        fr = tracks[tracks["FLIGHT_ID"] == f]
        mg = fr.merge(osv, on="timestamp")
        rec: dict[str, Any] = {
            "flight": f,
            "callsign": str(fr["CALL_SIGN"].iloc[0]) if len(fr) else None,
            "typecode": str(fr["ICAO_TYPE"].iloc[0]) if len(fr) else None,
        }
        if len(mg):
            mg = mg.assign(d=_haversine(mg["LATITUDE"], mg["LONGITUDE"], mg["latitude"], mg["longitude"]))
            g = mg.groupby("flight_id").agg(d=("d", "median"), n=("d", "size"))
            g = g[g["n"] >= 12].sort_values("d")
            if len(g):
                rec.update(flight_id=str(g.index[0]), match_median_m=float(g["d"].iloc[0]), match_n=int(g["n"].iloc[0]))
        rec["matched"] = bool(rec.get("match_median_m", np.inf) < max_median_m)
        rows.append(rec)
    return pd.DataFrame(rows)


def annotation_mask(polygons: list[list[float]], size: int = 1024) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=1)
    return np.asarray(img, dtype=bool)


def _metres(lat: np.ndarray, lon: np.ndarray, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.pi / 180 * R_EARTH
    return lon * np.cos(np.radians(lat0)) * k, lat * k


def _normals(s: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.maximum(s - _HALF_TANGENT_M, s[0]), np.minimum(s + _HALF_TANGENT_M, s[-1])
    tx, ty = np.interp(hi, s, x) - np.interp(lo, s, x), np.interp(hi, s, y) - np.interp(lo, s, y)
    n = np.hypot(tx, ty)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 50, ty / n, np.nan), np.where(n > 50, -tx / n, np.nan)


def _nearest_run(hit: np.ndarray) -> float:
    """Mean offset of the contiguous annotated run nearest to offset 0 (NaN if none)."""
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return float("nan")
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    runs = np.split(idx, cuts)
    centres = [float(_OFFSETS[r].mean()) for r in runs]
    gaps = [
        min(abs(_OFFSETS[r[0]]), abs(_OFFSETS[r[-1]])) if _OFFSETS[r[0]] * _OFFSETS[r[-1]] > 0 else 0.0 for r in runs
    ]
    return centres[int(np.argmin(gaps))]


class GvccsRaw:
    """:class:`~skywind.evaluate.contrails.TruthSource` from the raw GVCCS train split."""

    def __init__(self, settings: Settings, stride: int = 3, max_age_min: float = 40.0) -> None:
        self.settings = settings
        self.stride = stride
        self.max_age_min = max_age_min
        site = settings.site
        root = settings.paths.gvccs_dir
        self.root = root
        self.camera = Camera.from_grid(
            root / "azimuth_zenith_grid.npy",
            site.camera_lat,
            site.camera_lon,
            site.camera_alt_m,
            site.camera_x_offset_px,
        )
        self.matches: dict[str, pd.DataFrame] = {}

    def days(self) -> list[str]:
        have = set(available_days(self.settings.paths))
        return [d for d in train_days(self.root) if d in have]

    # ------------------------------------------------------------------ emission points
    def _emissions(self, g: GvccsDay, sv: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
        ok = matches[matches["matched"]]
        t0, t1 = g.frames["time"].min() - pd.Timedelta(minutes=45), g.frames["time"].max()
        src = sv[sv["flight_id"].isin(set(ok["flight_id"])) & sv["timestamp"].between(t0, t1)].copy()
        src["gap_m"] = (src["geoaltitude"] - src["altitude"]) * FT
        src["gap_m"] = src.groupby("flight_id")["gap_m"].transform(
            lambda x: x.rolling(25, center=True, min_periods=5).median().interpolate(limit_direction="both")
        )
        pts = emission_points(src)
        gap = src.set_index(["flight_id", "timestamp"])["gap_m"]
        key = pd.MultiIndex.from_arrays([pts["flight_id"], pd.to_datetime(pts["emission_time"], utc=True)])
        pts["gap_m"] = gap.reindex(key).to_numpy()
        pts["gap_m"] = pts["gap_m"].fillna(pts["gap_m"].median()).fillna(0.0)
        flight = dict(zip(ok["flight_id"], ok["flight"], strict=True))
        pts["flight"] = pts["flight_id"].map(flight)
        pts["emission_time"] = pd.to_datetime(pts["emission_time"], utc=True).dt.as_unit("ns")
        return pts

    def load(self, day: str) -> ContrailTruth:
        g = load_day(self.root, day)
        sv, _ = state_vectors(self.settings, day)
        matches = match_flights(g.tracks, sv, sorted(g.polygons["flight"].unique()))
        self.matches[day] = matches
        em = self._emissions(g, sv, matches)
        ref = advect(
            em,
            met_dataset(load_era5(self.settings.paths.era5(day))),
            g.frames["time"].max(),
            max_age_min=self.max_age_min + 5,
        )
        centres = [
            self._flight_centres(g, f, em[em["flight_id"] == fid], ref[ref["flight_id"] == fid])
            for f, fid in zip(matches["flight"], matches.get("flight_id", pd.Series(dtype=str)), strict=False)
            if isinstance(fid, str) and (em["flight_id"] == fid).any()
        ]
        frames = [c for c in centres if len(c)]
        cols = [
            "flight",
            "flight_id",
            "waypoint",
            "time",
            "age_min",
            "centre_lat",
            "centre_lon",
            "normal_e",
            "normal_n",
            "offset_m",
        ]
        return ContrailTruth(day, em, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols))

    # ------------------------------------------------------------------ centre lines
    def _flight_centres(self, g: GvccsDay, flight: str, em: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
        lat0 = self.settings.site.lat
        ex, ey = _metres(em["latitude"].to_numpy(), em["longitude"].to_numpy(), lat0)
        s_map = pd.Series(np.r_[0, np.cumsum(np.hypot(np.diff(ex), np.diff(ey)))], index=em["waypoint"].to_numpy())
        gap = pd.Series(em["gap_m"].to_numpy(), index=em["waypoint"].to_numpy())
        polys = g.polygons[g.polygons["flight"] == flight]
        out = []
        for t, frame in polys.groupby("time"):
            pts = ref[ref["time"] == t]
            if len(pts) >= 2:
                out.append(self._frame_centres(pts, s_map, gap, annotation_mask(list(frame["polygon"])), lat0))
        res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        return res.assign(flight=flight, flight_id=str(em["flight_id"].iloc[0])) if len(res) else res

    def _frame_centres(
        self, pts: pd.DataFrame, s_map: pd.Series, gap: pd.Series, mask: np.ndarray, lat0: float
    ) -> pd.DataFrame:
        pts = pts.assign(s=pts["waypoint"].map(s_map)).dropna(subset=["s"]).sort_values("s")
        x, y = _metres(pts["latitude"].to_numpy(), pts["longitude"].to_numpy(), lat0)
        ne, nn = _normals(pts["s"].to_numpy(float), x, y)
        age = (pts["time"] - pts["emission_time"]).dt.total_seconds().to_numpy() / 60
        keep = np.isfinite(ne) & (age >= 0.5) & (age <= self.max_age_min)
        keep &= (np.arange(len(pts)) % self.stride) == 0
        pts, ne, nn, age = pts[keep], ne[keep], nn[keep], age[keep]
        if pts.empty:
            return pd.DataFrame()
        h = isa_altitude_m(pts["level"].to_numpy() * 100) + pts["waypoint"].map(gap).to_numpy()
        off = self._offsets(pts, ne, nn, h, mask)
        ok = np.isfinite(off)
        c_lat, c_lon = offset_latlon(
            pts["latitude"].to_numpy()[ok], pts["longitude"].to_numpy()[ok], off[ok] * ne[ok], off[ok] * nn[ok]
        )
        return pd.DataFrame(
            {
                "waypoint": pts["waypoint"].to_numpy()[ok],
                "time": pts["time"].to_numpy()[ok],
                "age_min": age[ok],
                "centre_lat": c_lat,
                "centre_lon": c_lon,
                "normal_e": ne[ok],
                "normal_n": nn[ok],
                "offset_m": off[ok],
            }
        ).assign(time=lambda d: pd.to_datetime(d["time"], utc=True).dt.as_unit("ns"))

    def _offsets(
        self, pts: pd.DataFrame, ne: np.ndarray, nn: np.ndarray, h: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Annotated centre offset along the normal for each point (NaN where not annotated / not visible)."""
        lat = pts["latitude"].to_numpy()[:, None]
        lon = pts["longitude"].to_numpy()[:, None]
        la, lo = offset_latlon(lat, lon, _OFFSETS[None, :] * ne[:, None], _OFFSETS[None, :] * nn[:, None])
        az, zen = self.camera.az_zen(la, lo, np.broadcast_to(h[:, None], la.shape))
        px, py = self.camera.pixel(az, zen)
        xi, yi = np.rint(px).astype(int), np.rint(py).astype(int)
        inside = (zen <= _MAX_ZENITH) & (xi >= 0) & (xi < mask.shape[1]) & (yi >= 0) & (yi < mask.shape[0])
        hit = inside & mask[np.clip(yi, 0, mask.shape[0] - 1), np.clip(xi, 0, mask.shape[1] - 1)]
        out = np.full(len(pts), np.nan)
        for i in np.flatnonzero(inside.mean(1) >= 0.5):
            out[i] = _nearest_run(hit[i])
        return out
