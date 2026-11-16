"""Units, ISA atmosphere, local projection and angle helpers (pure numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "FT",
    "KT",
    "R_EARTH",
    "FloatArray",
    "angle_diff_deg",
    "cas_from_mach",
    "circular_mean_deg",
    "circular_sd_deg",
    "isa_altitude_m",
    "isa_pressure_pa",
    "local_xy_km",
    "offset_latlon",
    "unit_vectors",
]

type FloatArray = npt.NDArray[np.float64]

KT = 0.514444
FT = 0.3048
R_EARTH = 6371e3
_P0, _T_GRAD, _EXP = 101325.0, 2.25577e-5, 5.25588
_H_TROPO, _P_TROPO, _H_SCALE = 11000.0, 22632.06, 6341.62


def _f(x: Any) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def isa_pressure_pa(alt_m: Any) -> FloatArray:
    """ISA static pressure (Pa) at pressure altitude ``alt_m`` (troposphere + isothermal layer up to 20 km)."""
    h = _f(alt_m)
    lo = _P0 * (1 - _T_GRAD * np.minimum(h, _H_TROPO)) ** _EXP
    return np.where(h > _H_TROPO, _P_TROPO * np.exp(-(h - _H_TROPO) / _H_SCALE), lo)


def isa_altitude_m(p_pa: Any) -> FloatArray:
    """Inverse of :func:`isa_pressure_pa`."""
    p = _f(p_pa)
    tropo = (1 - (np.maximum(p, _P_TROPO) / _P0) ** (1 / _EXP)) / _T_GRAD
    return np.where(p < _P_TROPO, _H_TROPO - _H_SCALE * np.log(p / _P_TROPO), tropo)


def cas_from_mach(mach: Any, p_pa: Any) -> FloatArray:
    """Calibrated airspeed (m/s) from Mach number and static pressure (Pa)."""
    p0, a0 = 101325.0, 340.294
    qc = _f(p_pa) * ((1 + 0.2 * _f(mach) ** 2) ** 3.5 - 1)
    return a0 * np.sqrt(5 * ((qc / p0 + 1) ** (2 / 7) - 1))


def local_xy_km(lat: Any, lon: Any, lat0: float, lon0: float) -> tuple[FloatArray, FloatArray]:
    """Equirectangular east / north coordinates (km) around (lat0, lon0)."""
    k = np.pi / 180 * R_EARTH / 1000
    return (_f(lon) - lon0) * np.cos(np.radians(lat0)) * k, (_f(lat) - lat0) * k


def offset_latlon(lat: Any, lon: Any, de_m: Any, dn_m: Any) -> tuple[FloatArray, FloatArray]:
    """Move points by (east, north) metres."""
    la = _f(lat) + np.degrees(_f(dn_m) / R_EARTH)
    lo = _f(lon) + np.degrees(_f(de_m) / (R_EARTH * np.cos(np.radians(_f(lat)))))
    return la, lo


def angle_diff_deg(a: Any, b: Any) -> FloatArray:
    """Signed difference a - b wrapped to [-180, 180)."""
    return (_f(a) - _f(b) + 180.0) % 360.0 - 180.0


def unit_vectors(heading_deg: Any) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """(hx, hy) along the heading and (qx, qy) to its right, east / north components."""
    r = np.radians(_f(heading_deg))
    return np.sin(r), np.cos(r), np.cos(r), -np.sin(r)


def circular_mean_deg(sin_mean: Any, cos_mean: Any) -> FloatArray:
    return np.degrees(np.arctan2(_f(sin_mean), _f(cos_mean))) % 360.0


def circular_sd_deg(sin_mean: Any, cos_mean: Any) -> FloatArray:
    """Circular standard deviation (deg) from the mean sine and cosine."""
    r = np.minimum(np.hypot(_f(sin_mean), _f(cos_mean)), 1.0)
    return _f(np.degrees(np.sqrt(np.maximum(-2 * np.log(np.maximum(r, 1e-300)), 0.0))))
