"""Magnetic declination from the IGRF model (ppigrf), computed at run time at the observation points."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import ppigrf

from skywind.geo import FloatArray

__all__ = ["declination_deg"]


def declination_deg(lat: Any, lon: Any, alt_km: Any, when: datetime) -> FloatArray:
    """Declination (deg, positive east) so that true heading = magnetic heading + declination."""
    la = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lo = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    h = np.broadcast_to(np.asarray(alt_km, dtype=np.float64), la.shape)
    if la.size == 0:
        return np.zeros(0)
    be, bn, _ = ppigrf.igrf(lo, la, h, when)
    return np.asarray(np.degrees(np.arctan2(be, bn)), dtype=np.float64).reshape(la.shape)
