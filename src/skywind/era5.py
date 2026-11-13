"""ERA5 pressure-level fields: loading and linear interpolation in (time, pressure, latitude, longitude).

Linear in pressure, as the pycontrails advection does, so that the residuals and the advection share one convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from skywind.geo import FloatArray

__all__ = ["Era5", "load_era5"]

_RENAME = {"valid_time": "time", "pressure_level": "level"}


def load_era5(path: Path, variables: tuple[str, ...] = ("t", "q", "u", "v", "w", "z")) -> xr.Dataset:
    """Dataset with dims (time, level, latitude, longitude), all ascending; ``time`` naive UTC."""
    ds = xr.open_dataset(path)
    ds = ds.rename({k: v for k, v in _RENAME.items() if k in ds.dims or k in ds.coords})
    ds = ds[[v for v in variables if v in ds]]
    ds = ds.drop_vars([c for c in ("number", "expver") if c in ds.coords])
    for dim in ("time", "level", "latitude", "longitude"):
        ds = ds.sortby(dim)
    return ds.load()


class Era5:
    """Interpolator of u, v, t (and any other variable present) at arbitrary points."""

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds
        t = ds["time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
        self.axes = (
            t,
            ds["level"].values.astype(float),
            ds["latitude"].values.astype(float),
            ds["longitude"].values.astype(float),
        )
        self._f: dict[str, RegularGridInterpolator] = {}

    @classmethod
    def from_file(cls, path: Path) -> Era5:
        return cls(load_era5(path))

    def _interp(self, name: str) -> RegularGridInterpolator:
        if name not in self._f:
            values = self.ds[name].transpose("time", "level", "latitude", "longitude").values.astype(float)
            self._f[name] = RegularGridInterpolator(self.axes, values, bounds_error=False, fill_value=np.nan)
        return self._f[name]

    def __call__(self, name: str, t_epoch: Any, p_hpa: Any, lat: Any, lon: Any) -> FloatArray:
        x = np.stack([np.asarray(a, dtype=np.float64) for a in (t_epoch, p_hpa, lat, lon)], -1)
        return np.asarray(self._interp(name)(x), dtype=np.float64)

    def wind(self, t_epoch: Any, p_hpa: Any, lat: Any, lon: Any) -> tuple[FloatArray, FloatArray]:
        return self("u", t_epoch, p_hpa, lat, lon), self("v", t_epoch, p_hpa, lat, lon)
