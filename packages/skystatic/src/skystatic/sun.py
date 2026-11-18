"""Where the sun is, to keep what it leaves in an average out of the marks.

Near the sun, an hourly average holds its glare, the dust grains it lights and the reflections
that travel with it. In winter the sun stays low in one part of the sky for hours, so these can
recur at the same pixels in most hourly windows. When the camera's azimuth map and site are known,
a window in which the sun comes within a clearance angle of a pixel does not count for that pixel
(see ``contrast.fixed_marks``).

Solar position: the NOAA approximation (Spencer series), about 0.5 degree, enough for a clearance
of tens of degrees. Frame timestamps must be UTC.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from skystatic.geometry import SkyGeometry

__all__ = ["sun_clearance", "sun_position"]

# Sampling of the sun's path within a window: the sun moves about a degree in 5 minutes.
_STEP = timedelta(minutes=5)


def sun_position(time: datetime, latitude: float, longitude: float) -> tuple[float, float]:
    """(zenith, azimuth) of the sun in degrees at UTC ``time``; azimuth from north through east."""
    day = time.timetuple().tm_yday
    hour = time.hour + time.minute / 60 + time.second / 3600
    g = 2 * np.pi / 365 * (day - 1 + (hour - 12) / 24)
    equation = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    hour_angle = np.radians((hour * 60 + equation + 4 * longitude) / 4 - 180)
    lat = np.radians(latitude)
    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(
        hour_angle
    )
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
    azimuth = np.degrees(
        np.arctan2(
            np.sin(hour_angle),
            np.cos(hour_angle) * np.sin(lat) - np.tan(declination) * np.cos(lat),
        )
    )
    return float(zenith), float((azimuth + 180) % 360)


def _unit(
    zenith_deg: NDArray[np.floating], azimuth_deg: NDArray[np.floating]
) -> NDArray[np.float64]:
    z, a = np.radians(zenith_deg), np.radians(azimuth_deg)
    return np.stack([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)], axis=-1)


def sun_clearance(
    geometry: SkyGeometry, spans: Sequence[tuple[datetime, datetime]]
) -> NDArray[np.float32]:
    """Smallest angle (degrees) between each pixel and the sun during each span, (spans, H, W).

    Requires ``geometry.azimuth_deg`` and ``geometry.site``.
    """
    if geometry.azimuth_deg is None or geometry.site is None:
        raise ValueError("The sun guard needs the azimuth map and the site of the camera")
    finite = np.isfinite(geometry.zenith_deg) & np.isfinite(geometry.azimuth_deg)
    pixels = _unit(
        np.where(finite, geometry.zenith_deg, 180.0), np.where(finite, geometry.azimuth_deg, 0.0)
    )
    out = np.full((len(spans), *geometry.shape), 180.0, dtype=np.float32)
    for index, (start, end) in enumerate(spans):
        time = start
        while time <= end:
            zenith, azimuth = sun_position(time, *geometry.site)
            sun = _unit(np.array(zenith), np.array(azimuth))
            angle = np.degrees(np.arccos(np.clip(pixels @ sun, -1, 1))).astype(np.float32)
            np.minimum(out[index], angle, out=out[index])
            time += _STEP
    return out
