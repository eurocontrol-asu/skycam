"""Gridded export of the corrected wind: ERA5 horizontal grid, ERA5 + intermediate pressure levels, finer time step.

Variable names follow the CF / pycontrails conventions so that the NetCDF file opens directly as a pycontrails
``MetDataset`` in place of ERA5 (``eastward_wind`` / ``northward_wind`` are the corrected winds). The ERA5 winds, the
residual and its standard deviation are kept alongside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from skywind.config import ExportConfig, Site
from skywind.field.data import Points
from skywind.windfield import ResidualEstimator

__all__ = ["CF_NAMES", "era5_on_grid", "export_grid", "residual_dataset", "to_met_dataset", "write_netcdf"]

CF_NAMES = {
    "t": "air_temperature",
    "q": "specific_humidity",
    "u": "eastward_wind",
    "v": "northward_wind",
    "w": "lagrangian_tendency_of_air_pressure",
    "z": "geopotential",
}


def export_grid(era5: xr.Dataset, site: Site, cfg: ExportConfig, start: pd.Timestamp, stop: pd.Timestamp
                ) -> dict[str, np.ndarray]:
    """ERA5 latitudes / longitudes within ``half_width_deg`` of the site, ERA5 levels plus a regular ``level_step_hpa``
    grid, and times every ``time_step_min`` aligned on the hour, covering [start - pad_before, stop + pad_after]."""
    lat, lon = era5["latitude"].values, era5["longitude"].values
    lat = lat[np.abs(lat - site.lat) <= cfg.half_width_deg + 1e-9]
    lon = lon[np.abs(lon - site.lon) <= cfg.half_width_deg + 1e-9]
    regular = np.arange(cfg.min_level_hpa, cfg.max_level_hpa + 1e-9, cfg.level_step_hpa)
    native = era5["level"].values.astype(float)
    levels = np.unique(np.concatenate([regular, native[(native >= cfg.min_level_hpa) & (native <= cfg.max_level_hpa)]]))
    step = pd.Timedelta(minutes=cfg.time_step_min)
    t0 = (start - pd.Timedelta(minutes=cfg.pad_before_min)).floor(step)
    t1 = (stop + pd.Timedelta(minutes=cfg.pad_after_min)).ceil(step)
    times = pd.date_range(t0, t1, freq=step).tz_localize(None).values
    return {"latitude": lat, "longitude": lon, "level": levels, "time": times}


def era5_on_grid(era5: xr.Dataset, grid: dict[str, np.ndarray]) -> xr.Dataset:
    """ERA5 linearly interpolated (time, pressure) onto the export grid; horizontal nodes are ERA5 nodes."""
    ds = era5.sel(latitude=grid["latitude"], longitude=grid["longitude"])
    return ds.interp(level=grid["level"], time=grid["time"], method="linear")


def residual_dataset(ds: xr.Dataset, estimator: ResidualEstimator, site: Site) -> xr.Dataset:
    """Residual posterior on every node of ``ds`` (ERA5 on the export grid, short names u, v...)."""
    sub = ds[["u", "v"]].transpose("time", "level", "latitude", "longitude")
    T, P, LA, LO = np.meshgrid(sub["time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9,
                               sub["level"].values, sub["latitude"].values, sub["longitude"].values, indexing="ij")
    bg = np.stack([np.nan_to_num(sub["u"].values.ravel()), np.nan_to_num(sub["v"].values.ravel())], 1)
    pred = estimator.predict(Points.from_geo(site, T.ravel(), P.ravel(), LA.ravel(), LO.ravel(), bg))
    dims = ("time", "level", "latitude", "longitude")
    shape = T.shape
    fields = {
        "residual_eastward_wind": pred.du, "residual_northward_wind": pred.dv,
        "residual_eastward_wind_std": pred.std_u, "residual_northward_wind_std": pred.std_v,
        "residual_wind_covariance": pred.cov_uv, "residual_information": pred.information,
    }
    return xr.Dataset({k: (dims, v.reshape(shape).astype(np.float32)) for k, v in fields.items()}, coords=sub.coords)


def to_met_dataset(ds: xr.Dataset, residual: xr.Dataset, attrs: dict[str, Any]) -> xr.Dataset:
    """Assemble the exported dataset: corrected winds under the CF names, ERA5 winds and residual kept."""
    out = ds.rename({k: v for k, v in CF_NAMES.items() if k in ds})
    out["era5_eastward_wind"] = out["eastward_wind"]
    out["era5_northward_wind"] = out["northward_wind"]
    out = xr.merge([out, residual])
    out["eastward_wind"] = out["era5_eastward_wind"] + out["residual_eastward_wind"]
    out["northward_wind"] = out["era5_northward_wind"] + out["residual_northward_wind"]
    out = out.transpose("longitude", "latitude", "level", "time")
    out["level"].attrs.update(units="hPa", long_name="pressure level")
    out.attrs.update({k: str(v) for k, v in attrs.items()})
    return out


def write_netcdf(ds: xr.Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    return path
