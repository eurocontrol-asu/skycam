"""Array containers shared by the residual-field estimators."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from skywind.config import Site
from skywind.geo import FT, FloatArray, isa_altitude_m, local_xy_km, unit_vectors

__all__ = ["ALONG", "CROSS", "IntArray", "ObsSet", "Points", "Prediction"]

type IntArray = npt.NDArray[np.int64]
ALONG, CROSS = 0, 1


@dataclass(frozen=True)
class Points:
    """Query points: ``X`` (m, 4) = east km, north km, pressure altitude m, epoch s; ``bg`` (m, 2) ERA5 wind m/s."""

    X: FloatArray
    bg: FloatArray

    @classmethod
    def from_geo(cls, site: Site, t: object, p_hpa: object, lat: object, lon: object,
                 bg: FloatArray | None = None) -> Points:
        x, y = local_xy_km(lat, lon, site.lat, site.lon)
        z = isa_altitude_m(np.asarray(p_hpa, float) * 100)
        tt = np.broadcast_to(np.asarray(t, float), x.shape)
        X = np.stack([x, y, np.broadcast_to(z, x.shape), tt], 1).astype(np.float64)
        return cls(X, np.zeros((len(X), 2)) if bg is None else np.asarray(bg, np.float64))

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class ObsSet:
    """Scalar observations y_i = e_i . r(X_i) + b_{group_i} + noise, with ``kind`` ALONG (heading) or CROSS (right)."""

    X: FloatArray
    e: FloatArray
    y: FloatArray
    kind: IntArray
    aircraft: IntArray
    bg: FloatArray
    day: npt.NDArray[np.str_]
    icao24: npt.NDArray[np.str_]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def group(self) -> IntArray:
        """Random-effect group: one per (aircraft, kind)."""
        return self.aircraft * 2 + self.kind

    def subset(self, mask: npt.NDArray[np.bool_] | IntArray) -> ObsSet:
        return replace(self, X=self.X[mask], e=self.e[mask], y=self.y[mask], kind=self.kind[mask],
                       aircraft=self.aircraft[mask], bg=self.bg[mask], day=self.day[mask], icao24=self.icao24[mask])

    @classmethod
    def from_frame(cls, w: pd.DataFrame, site: Site, use_cross: bool = False) -> ObsSet:
        """From an observation table with residuals (:func:`skywind.observations.add_residuals`). Cross-heading rows are
        added for airframes with a heading offset (``hdg_offset`` finite) when ``use_cross``."""
        x, y = local_xy_km(w["latitude"], w["longitude"], site.lat, site.lon)
        X = np.stack([x, y, w["altitude"].to_numpy(float) * FT, w["t"].to_numpy(float)], 1)
        hx, hy, qx, qy = unit_vectors(w["hdg"])
        ac_codes = pd.factorize(w["day"].astype(str) + ":" + w["icao24"].astype(str))[0].astype(np.int64)
        bg = np.stack([w["u_era"].to_numpy(float), w["v_era"].to_numpy(float)], 1)
        day, icao = w["day"].astype(str).to_numpy(), w["icao24"].astype(str).to_numpy()
        along = cls(X, np.stack([hx, hy], 1), w["d_along"].to_numpy(float), np.zeros(len(w), np.int64), ac_codes, bg,
                    day, icao)
        if not use_cross:
            return along
        ok = np.isfinite(w["hdg_offset"].to_numpy(float)) & np.isfinite(w["d_right"].to_numpy(float))
        cross = cls(X[ok], np.stack([qx, qy], 1)[ok], w["d_right"].to_numpy(float)[ok], np.ones(int(ok.sum()), np.int64),
                    ac_codes[ok], bg[ok], day[ok], icao[ok])
        return along.concat(cross)

    def concat(self, other: ObsSet) -> ObsSet:
        return ObsSet(*(np.concatenate([getattr(self, f), getattr(other, f)]) for f in
                        ("X", "e", "y", "kind", "aircraft", "bg", "day", "icao24")))


@dataclass(frozen=True)
class Prediction:
    """Posterior of the residual wind at query points: mean (du, dv) and covariance terms, m/s and (m/s)^2."""

    du: FloatArray
    dv: FloatArray
    var_u: FloatArray
    var_v: FloatArray
    cov_uv: FloatArray
    prior_var: float

    @property
    def std_u(self) -> FloatArray:
        return np.sqrt(np.maximum(self.var_u, 0.0))

    @property
    def std_v(self) -> FloatArray:
        return np.sqrt(np.maximum(self.var_v, 0.0))

    def var_along(self, ex: FloatArray, ey: FloatArray) -> FloatArray:
        return self.var_u * ex**2 + 2 * self.cov_uv * ex * ey + self.var_v * ey**2

    @property
    def information(self) -> FloatArray:
        """1 - (mean posterior variance / prior variance): 0 = ERA5 only, 1 = fully determined."""
        return 1 - 0.5 * (self.var_u + self.var_v) / self.prior_var
