"""Where the sun is: its azimuth and elevation seen from the camera site at a UTC time.

The NOAA solar position algorithm (Meeus' low-precision series, as in the NOAA solar calculator):
about 0.01 degree between 1950 and 2050 when compared with the reference algorithm SPA (see the
tests), far below the 0.5-degree annuli of the sun zone. No atmospheric refraction: the geometric
position (refraction lifts the sun by 0.5 degree at the horizon, less than 0.02 degree above +45).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Site", "SunPosition", "sun_position"]


class Site(BaseModel):
    """Position of the camera, WGS84 degrees (longitude positive east)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=360.0)


class SunPosition(BaseModel):
    """Azimuth (degrees from north through east) and elevation of the sun's centre, degrees."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    azimuth_deg: float
    elevation_deg: float

    @property
    def zenith_deg(self) -> float:
        return 90.0 - self.elevation_deg


def _declination_and_equation(julian_century: float) -> tuple[float, float]:
    """Sun's apparent declination (radians) and the equation of time (minutes)."""
    t = julian_century
    mean_longitude = math.radians((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360)
    anomaly = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    centre = (
        math.sin(anomaly) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * anomaly) * (0.019993 - 0.000101 * t)
        + math.sin(3 * anomaly) * 0.000289
    )
    node = math.radians(125.04 - 1934.136 * t)
    apparent = mean_longitude + math.radians(centre - 0.00569 - 0.00478 * math.sin(node))
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    obliquity = math.radians(23 + (26 + seconds / 60) / 60 + 0.00256 * math.cos(node))
    declination = math.asin(math.sin(obliquity) * math.sin(apparent))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    y = math.tan(obliquity / 2) ** 2
    equation = 4 * math.degrees(
        y * math.sin(2 * mean_longitude)
        - 2 * eccentricity * math.sin(anomaly)
        + 4 * eccentricity * y * math.sin(anomaly) * math.cos(2 * mean_longitude)
        - 0.5 * y * y * math.sin(4 * mean_longitude)
        - 1.25 * eccentricity * eccentricity * math.sin(2 * anomaly)
    )
    return declination, equation


def sun_position(time: datetime, site: Site) -> SunPosition:
    """The sun at ``time`` (timezone-aware; naive times are refused, frames are in UTC)."""
    if time.tzinfo is None:
        raise ValueError("sun_position needs a timezone-aware time (UTC for the frames)")
    utc = time.astimezone(UTC)
    julian_day = utc.timestamp() / 86400 + 2440587.5
    declination, equation = _declination_and_equation((julian_day - 2451545) / 36525)
    minutes = utc.hour * 60 + utc.minute + (utc.second + utc.microsecond / 1e6) / 60
    true_solar = (minutes + equation + 4 * site.longitude_deg) % 1440
    hour_angle = math.radians(true_solar / 4 - 180)
    latitude = math.radians(site.latitude_deg)
    cos_zenith = math.sin(latitude) * math.sin(declination) + math.cos(latitude) * math.cos(
        declination
    ) * math.cos(hour_angle)
    zenith = math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))
    azimuth = math.degrees(
        math.atan2(
            math.sin(hour_angle),
            math.cos(hour_angle) * math.sin(latitude) - math.tan(declination) * math.cos(latitude),
        )
    )
    return SunPosition(azimuth_deg=(azimuth + 180) % 360, elevation_deg=90.0 - zenith)
