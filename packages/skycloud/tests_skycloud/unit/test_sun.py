from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from skycloud.sun import (
    AureoleInput,
    artefact_confidence,
    aureole_confidence,
    bilinear_nan_border,
    detect_core,
    sun_position,
)
from tests_skycloud.unit.test_geometry import fisheye


def test_sun_position_at_the_june_solstice_noon() -> None:
    # Brétigny, 2024-06-20 11:51 UTC is close to solar noon: zenith = latitude - declination (23.44)
    az, zen = sun_position(datetime(2024, 6, 20, 11, 51, tzinfo=UTC), 48.600518, 2.346795)
    assert abs(zen - (48.6005 - 23.44)) < 0.3
    assert abs(az - 180.0) < 3.0
    _, zen_night = sun_position(datetime(2024, 6, 20, 23, 51, tzinfo=UTC), 48.600518, 2.346795)
    assert zen_night > 90.0


def test_bilinear_sampling_with_nan_border() -> None:
    y = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = bilinear_nan_border(
        y, np.array([1.5, 0.0, 3.5], np.float32), np.array([0.5, 0.0, 1.0], np.float32)
    )
    assert np.isclose(out[0], 0.5 * (1.5 + 5.5)) and out[1] == 0.0
    assert np.isnan(out[2])  # a neighbour outside the image


def test_detect_core_finds_the_saturated_sun_and_its_angle() -> None:
    geo = fisheye(81)
    img = np.full((81, 81, 3), 100, np.uint8)
    sx, sy = detect_core(img, 90.0, 30.0, geo).sun_px  # the ephemeris pixel
    iy, ix = sy, sx
    img[iy - 3 : iy + 4, ix - 3 : ix + 4] = 255
    sun = detect_core(img, 90.0, 30.0, geo)
    assert sun.visible and sun.core[iy, ix] and sun.core.sum() > 49
    assert sun.gam[iy, ix] < 3.0
    hidden = detect_core(np.full((81, 81, 3), 100, np.uint8), 90.0, 30.0, geo)
    assert not hidden.visible and not hidden.core.any()
    conf = artefact_confidence(img, hidden, geo.inside, geo.inside)
    assert np.all(conf == 1.0)  # no core: no artefact


def test_aureole_rule_narrows_the_cut_on_a_clear_homogeneous_ring() -> None:
    geo = fisheye(161)
    g = np.asarray(
        np.degrees(np.arccos(np.clip(geo.vec @ np.array([0.0, 0.0, 1.0], np.float32), -1, 1)))
    )
    kconf = np.clip((g - 30.0) / 2.0, 0.0, 1.0).astype(np.float32)  # a kit cut at about 31 deg
    a = AureoleInput(
        kconf, np.zeros(geo.shape, np.float32), geo.inside, np.zeros(geo.shape, bool), 0.0, 0.0
    )
    out, info = aureole_confidence(a, geo, alpha_clear=0.2, aur_min=8.0, aur_veil=18.0)
    assert info is not None and info["new_cut"] == 8.0
    assert out[(g > 12) & (g < 25) & geo.inside].min() == 1.0
