"""Advection of emission points with pycontrails ``DryAdvection`` (pointwise, u / v / omega, no sedimentation),
on ERA5 or on the corrected dataset exported by :mod:`skywind.export`."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from pycontrails import GeoVectorDataset, MetDataset
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.physics import units

from skywind.export import CF_NAMES

__all__ = ["advect", "emission_points", "met_dataset"]

_MET_VARS = ["air_temperature", "eastward_wind", "northward_wind", "lagrangian_tendency_of_air_pressure"]


def met_dataset(ds: xr.Dataset, winds: Literal["corrected", "era5"] = "corrected") -> MetDataset:
    """pycontrails MetDataset from an exported dataset (or a raw ERA5 dataset with short names)."""
    ds = ds.rename({k: v for k, v in CF_NAMES.items() if k in ds.data_vars})
    if winds == "era5" and "era5_eastward_wind" in ds:
        ds = ds.assign(eastward_wind=ds["era5_eastward_wind"], northward_wind=ds["era5_northward_wind"])
    ds = ds[_MET_VARS].transpose("longitude", "latitude", "level", "time")
    return MetDataset(ds.sortby("longitude").sortby("latitude").sortby("level").sortby("time").load())


def emission_points(sv: pd.DataFrame, min_level: float = 150.0, max_level: float = 400.0) -> pd.DataFrame:
    """State vectors -> emission points (``flight_id``, ``emission_time``, position, ``level`` from barometric altitude)."""
    out = pd.DataFrame({
        "flight_id": sv["flight_id"].astype(str).to_numpy(),
        "emission_time": pd.to_datetime(sv["timestamp"], utc=True).to_numpy(),
        "latitude": sv["latitude"].to_numpy(float),
        "longitude": sv["longitude"].to_numpy(float),
        "level": units.m_to_pl(sv["altitude"].to_numpy(float) * 0.3048),
    })
    out = out[(out["level"] > min_level) & (out["level"] < max_level)].reset_index(drop=True)
    out["waypoint"] = np.arange(len(out))
    return out


def advect(points: pd.DataFrame, met: MetDataset, until: pd.Timestamp, step_s: float = 30.0,
           max_age_min: float = 90.0) -> pd.DataFrame:
    """Positions of every emission point at each ``step_s`` step (aligned on multiples of ``step_s``) until ``until``.

    ``points`` needs ``flight_id``, ``waypoint``, ``emission_time`` (UTC), ``latitude``, ``longitude``, ``level`` (hPa).
    """
    t_em = pd.to_datetime(points["emission_time"], utc=True).dt.as_unit("ns")
    source = GeoVectorDataset({
        "longitude": points["longitude"].to_numpy(float), "latitude": points["latitude"].to_numpy(float),
        "level": points["level"].to_numpy(float), "time": t_em.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"),
        "waypoint": points["waypoint"].to_numpy(), "flight_id": points["flight_id"].astype(str).to_numpy(),
    })
    step = pd.Timedelta(seconds=step_s)
    steps = pd.date_range(t_em.min().floor(step) + step, until.ceil(step), freq=step)
    params = {"timesteps": steps.tz_localize(None).to_numpy(dtype="datetime64[ns]"),
              "max_age": np.timedelta64(int(max_age_min * 60), "s"), "azimuth": None, "width": None, "depth": None,
              "sedimentation_rate": 0.0, "include_source_in_output": False, "interpolation_bounds_error": False,
              "downselect_met": False}
    ev = DryAdvection(met, params=params).eval(source).dataframe
    ev["time"] = pd.to_datetime(ev["time"]).dt.tz_localize("UTC").dt.as_unit("ns")
    ev = ev.dropna(subset=["latitude", "longitude", "level"])
    ev = ev.merge(points[["waypoint", "emission_time"]].assign(emission_time=t_em), on="waypoint")
    return ev[["flight_id", "waypoint", "emission_time", "time", "latitude", "longitude", "level"]]
