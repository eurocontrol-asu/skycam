"""Gridded export of the corrected wind: ERA5 horizontal grid, ERA5 + intermediate pressure levels, finer time step.

Variable names follow the CF / pycontrails conventions so that the NetCDF file opens directly as a pycontrails
``MetDataset`` in place of ERA5 (``eastward_wind`` / ``northward_wind`` are the corrected winds). The ERA5 winds, the
residual and its standard deviation are kept alongside.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from skywind.config import ExportConfig, Site
from skywind.field.data import Points
from skywind.geo import isa_altitude_m, local_xy_km
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


def export_grid(
    era5: xr.Dataset, site: Site, cfg: ExportConfig, start: pd.Timestamp, stop: pd.Timestamp
) -> dict[str, np.ndarray]:
    """ERA5 latitudes / longitudes within ``half_width_deg`` of the site (each interval split ``horizontal_refine``
    times, so that the residual's horizontal scale is resolved), ERA5 levels plus a regular ``level_step_hpa`` grid,
    and times every ``time_step_min`` aligned on the hour, covering [start - pad_before, stop + pad_after]."""
    lat, lon = era5["latitude"].values, era5["longitude"].values
    lat = _refine(lat[np.abs(lat - site.lat) <= cfg.half_width_deg + 1e-9], cfg.horizontal_refine)
    lon = _refine(lon[np.abs(lon - site.lon) <= cfg.half_width_deg + 1e-9], cfg.horizontal_refine)
    regular = np.arange(cfg.min_level_hpa, cfg.max_level_hpa + 1e-9, cfg.level_step_hpa)
    native = era5["level"].values.astype(float)
    levels = np.unique(np.concatenate([regular, native[(native >= cfg.min_level_hpa) & (native <= cfg.max_level_hpa)]]))
    step = pd.Timedelta(minutes=cfg.time_step_min)
    t0 = (start - pd.Timedelta(minutes=cfg.pad_before_min)).floor(step)
    t1 = (stop + pd.Timedelta(minutes=cfg.pad_after_min)).ceil(step)
    times = pd.date_range(t0, t1, freq=step).tz_localize(None).values
    return {"latitude": lat, "longitude": lon, "level": levels, "time": times}


def _refine(nodes: np.ndarray, factor: int) -> np.ndarray:
    nodes = np.sort(nodes.astype(float))
    if factor <= 1 or len(nodes) < 2:
        return nodes
    fine = [np.linspace(a, b, factor, endpoint=False) for a, b in itertools.pairwise(nodes)]
    return np.round(np.concatenate([*fine, nodes[-1:]]), 6)


def era5_on_grid(era5: xr.Dataset, grid: dict[str, np.ndarray]) -> xr.Dataset:
    """ERA5 linearly interpolated (time, pressure, and bilinearly between ERA5 nodes) onto the export grid; the result
    equals ERA5 at its own nodes and is the same piecewise-linear field elsewhere."""
    return era5.interp(
        latitude=grid["latitude"], longitude=grid["longitude"], level=grid["level"], time=grid["time"], method="linear"
    )


def _stride_mask(n: int, stride: int) -> np.ndarray:
    m = np.zeros(n, bool)
    m[:: max(stride, 1)] = True
    m[-1] = True
    return m


def _posterior_on_grid(
    sub: xr.Dataset, estimator: ResidualEstimator, site: Site, want: np.ndarray
) -> dict[str, np.ndarray]:
    """Posterior arrays (time, level, latitude, longitude); factorised grid solver when the estimator has one."""
    times = sub["time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    lat, lon, lev = sub["latitude"].values, sub["longitude"].values, sub["level"].values
    shape = (len(times), len(lev), len(lat), len(lon))
    grid_solver = getattr(estimator, "predict_grid", None)
    if grid_solver is not None:
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        x, y = local_xy_km(LA.ravel(), LO.ravel(), site.lat, site.lon)
        res = grid_solver(times, isa_altitude_m(lev * 100.0), np.stack([x, y], 1), want[0].reshape(len(lev), -1))
        return {k: v.reshape(shape) for k, v in res.items()}
    T, P, LA4, LO4 = np.meshgrid(times, lev, lat, lon, indexing="ij")
    bg = np.stack([np.nan_to_num(sub["u"].values.ravel()), np.nan_to_num(sub["v"].values.ravel())], 1)
    pred = estimator.predict(
        Points.from_geo(site, T.ravel(), P.ravel(), LA4.ravel(), LO4.ravel(), bg), variance=want.ravel()
    )
    return {
        "du": pred.du.reshape(shape),
        "dv": pred.dv.reshape(shape),
        "var_u": pred.var_u.reshape(shape),
        "var_v": pred.var_v.reshape(shape),
        "cov_uv": pred.cov_uv.reshape(shape),
    }


def residual_dataset(
    ds: xr.Dataset, estimator: ResidualEstimator, site: Site, stride_h: int = 1, stride_level: int = 1
) -> xr.Dataset:
    """Residual posterior on every node of ``ds`` (ERA5 on the export grid, short names u, v...).

    The posterior mean is computed on every node; the posterior variance on every ``stride_h``-th horizontal node and
    ``stride_level``-th level (1 = everywhere), then linearly interpolated to the other nodes."""
    sub = ds[["u", "v"]].transpose("time", "level", "latitude", "longitude")
    dims = ("time", "level", "latitude", "longitude")
    masks = [
        np.ones(len(sub["time"]), bool),
        _stride_mask(len(sub["level"]), stride_level),
        _stride_mask(len(sub["latitude"]), stride_h),
        _stride_mask(len(sub["longitude"]), stride_h),
    ]
    coarse = np.ix_(*masks)
    want = np.zeros([len(m) for m in masks], bool)
    want[coarse] = True
    post = _posterior_on_grid(sub, estimator, site, want)
    out = xr.Dataset(
        {"residual_eastward_wind": (dims, post["du"]), "residual_northward_wind": (dims, post["dv"])},
        coords=sub.coords,
    )
    full = {}
    for name in ("var_u", "var_v", "cov_uv"):
        da = xr.DataArray(
            post[name][coarse], dims=dims, coords={d: sub[d].values[m] for d, m in zip(dims, masks, strict=True)}
        )
        full[name] = da.interp({d: sub[d].values for d in dims[1:]}, method="linear").values
    info = 1 - 0.5 * (full["var_u"] + full["var_v"]) / estimator.prior_var
    out["residual_eastward_wind_std"] = (dims, np.sqrt(np.maximum(full["var_u"], 0)))
    out["residual_northward_wind_std"] = (dims, np.sqrt(np.maximum(full["var_v"], 0)))
    out["residual_wind_covariance"] = (dims, full["cov_uv"])
    out["residual_information"] = (dims, info)
    return out.astype(np.float32)


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
