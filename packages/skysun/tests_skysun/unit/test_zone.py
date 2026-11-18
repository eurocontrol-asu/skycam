"""The sun zone on a synthetic glow: radial, non-increasing, fading, saturation reads 1."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from skysun import CameraGeometry, ZoneParams, sun_position
from skysun.zone import radial_profile, sun_saturation, zone_map
from tests_skysun.conftest import MORNING, SITE, frame

PARAMS = ZoneParams()


def glow_frame(geometry: CameraGeometry) -> tuple[np.ndarray, np.ndarray]:
    angle = geometry.angle_to(sun_position(MORNING, SITE))
    return frame(geometry, MORNING), angle


def test_profile_reads_the_glow_over_the_headroom(geometry: CameraGeometry) -> None:
    rgb, angle = glow_frame(geometry)
    profile = radial_profile(rgb, angle, geometry.inside, PARAMS)
    assert profile.sky_level == pytest.approx(80, abs=1)
    # glow 170 exp(-a/5) over the headroom 250 - 80 = 170: value exp(-a/5), 0.5 at 3.5 deg
    assert profile.radius_at(0.5) == pytest.approx(5 * np.log(2), abs=0.6)
    assert profile.radius_at(0.05) == pytest.approx(5 * np.log(20), abs=1.0)


def test_zone_is_non_increasing_and_fades_without_an_edge(geometry: CameraGeometry) -> None:
    rgb, angle = glow_frame(geometry)
    lifted = rgb.copy()
    # brighter beyond the reference annuli (a horizon): the prototype cut the zone at 40 deg
    lifted[(angle > 40) & (angle < 45)] = 120
    profile = radial_profile(lifted, angle, geometry.inside, PARAMS)
    assert (np.diff(profile.value) <= 1e-12).all()
    assert (profile.value[profile.centres >= 40] == 0).all()
    assert profile.value[(profile.centres > 28) & (profile.centres < 30)].min() > 0.05
    assert np.abs(np.diff(profile.value)).max() < 0.1  # no step between neighbouring annuli


def test_saturated_sun_and_its_spike_read_1_a_far_blob_does_not(geometry: CameraGeometry) -> None:
    rgb, angle = glow_frame(geometry)
    sx, sy = (int(v) for v in geometry.sun_pixel(angle))
    rgb[sy, sx:] = 255  # a star spike from the sun to the rim, beyond the zone's reach
    rows = np.indices(angle.shape)[0]
    blob = (angle > 60) & (angle < 63) & (np.abs(rows - sy) > 5)
    rgb[blob] = 255  # saturated, but not connected to the sun
    saturated = sun_saturation(rgb, angle, geometry.inside, PARAMS)
    profile = radial_profile(rgb, angle, geometry.inside, PARAMS)
    zone = zone_map(profile, angle, saturated, geometry.inside)
    spike = np.zeros(angle.shape, bool)
    spike[sy, sx:] = True
    assert (zone[spike & geometry.inside] == 1).all()
    assert (zone[blob & ~spike] == 0).all()
    assert (zone[angle < 1.9] == 1).all()


def test_no_sky_measured_gives_zeros_without_warnings(geometry: CameraGeometry) -> None:
    rgb, angle = glow_frame(geometry)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = radial_profile(rgb, angle, np.zeros(angle.shape, bool), PARAMS)
        zone = zone_map(profile, angle, np.zeros(angle.shape, bool), geometry.inside)
    assert np.isnan(profile.sky_level)
    assert (zone == 0).all()


def test_parameters_refuse_a_fade_beyond_the_reach() -> None:
    with pytest.raises(ValueError, match="reach"):
        ZoneParams(fade_deg=(30.0, 50.0))
