"""Corrected wind = ERA5 + estimated residual, with uncertainty, at any (time, pressure or altitude, lat, lon)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from skywind.config import Settings
from skywind.era5 import Era5
from skywind.field.data import ObsSet, Points, Prediction
from skywind.field.gp import GPEstimator, GPParams
from skywind.field.ridge import RidgeEstimator
from skywind.geo import isa_pressure_pa

__all__ = ["CorrectedWind", "ResidualEstimator", "build_estimator"]


class ResidualEstimator(Protocol):
    name: str

    @property
    def prior_var(self) -> float: ...

    def predict(self, points: Points, variance: npt.NDArray[np.bool_] | None = None) -> Prediction: ...


def build_estimator(obs: ObsSet, settings: Settings, method: str = "gp") -> ResidualEstimator:
    """``gp`` (default) or ``ridge`` (prototype baseline) from the configuration."""
    if method == "gp":
        return GPEstimator(
            obs,
            GPParams.from_config(settings.gp),
            settings.gp.cutoff,
            settings.gp.tile_minutes,
            settings.gp.max_neighbours,
        )
    if method == "ridge":
        return RidgeEstimator(obs, settings.ridge)
    raise ValueError(f"unknown estimator {method!r} (gp | ridge)")


class CorrectedWind:
    """``at`` returns a table with ERA5 wind, residual, corrected wind and the residual standard deviations.

    Where no observation informs the residual, the residual is 0 (ERA5) and its standard deviation equals the prior
    standard deviation; ``information`` (0..1) says how much the observations constrain the point.
    """

    def __init__(self, era5: Era5, estimator: ResidualEstimator, settings: Settings) -> None:
        self.era5 = era5
        self.estimator = estimator
        self.site = settings.site

    def at(self, t_epoch: object, p_hpa: object, lat: object, lon: object) -> pd.DataFrame:
        t, p, la, lo = np.broadcast_arrays(*(np.asarray(a, float) for a in (t_epoch, p_hpa, lat, lon)))
        u, v = self.era5.wind(t, p, la, lo)
        bg = np.stack([np.nan_to_num(u), np.nan_to_num(v)], 1)
        pred = self.estimator.predict(Points.from_geo(self.site, t.ravel(), p.ravel(), la.ravel(), lo.ravel(), bg))
        return pd.DataFrame(
            {
                "u_era5": u,
                "v_era5": v,
                "du": pred.du,
                "dv": pred.dv,
                "u": u + pred.du,
                "v": v + pred.dv,
                "std_u": pred.std_u,
                "std_v": pred.std_v,
                "cov_uv": pred.cov_uv,
                "information": pred.information,
            }
        )

    def at_altitude(self, t_epoch: object, alt_m: object, lat: object, lon: object) -> pd.DataFrame:
        """Same as :meth:`at` with ISA pressure altitude (m) instead of pressure."""
        return self.at(t_epoch, isa_pressure_pa(alt_m) / 100, lat, lon)
