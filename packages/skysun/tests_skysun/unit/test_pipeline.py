"""The two layers of a frame: always a value, zero at night, the sun read even behind a mast or
the tree line, reflections on their own scale, a temporal median that never smooths the map."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from skysun import CameraGeometry, SunCamera, measure_frame, sun_map, sun_position, temporal_profile
from tests_skysun.conftest import MORNING, NIGHT, NOON, REFERENCE, SITE, frame, optics, table


def camera(geometry: CameraGeometry) -> SunCamera:
    return SunCamera(geometry, SITE, optics(), table(0.04))


@pytest.mark.parametrize("time", [MORNING, NOON, NIGHT])
@pytest.mark.parametrize("visible", [True, False])
def test_every_pixel_has_a_value_in_the_unit_interval(
    geometry: CameraGeometry, time: datetime, visible: bool
) -> None:
    rgb = frame(geometry, time, sun_visible=visible)
    result = sun_map(rgb, time, camera(geometry), geometry.inside)
    for layer in (result.zone, result.reflections):
        assert layer.dtype == np.float32 and layer.shape == geometry.shape
        assert np.isfinite(layer).all()
        assert layer.min() >= 0 and layer.max() <= 1


def test_night_is_zero_everywhere(geometry: CameraGeometry) -> None:
    rgb = np.full((*geometry.shape, 3), 200, dtype=np.uint8)  # a bright night frame
    result = sun_map(rgb, NIGHT, camera(geometry), geometry.inside)
    assert (result.zone == 0).all() and (result.reflections == 0).all()
    assert result.info.sun_elevation_deg < 0 and result.info.visibility_source == "night"


def test_reflections_on_their_own_scale_only_when_the_sun_shines(geometry: CameraGeometry) -> None:
    cam = camera(geometry)
    shining = sun_map(frame(geometry, MORNING), MORNING, cam, geometry.inside)
    hidden = sun_map(frame(geometry, MORNING, sun_visible=False), MORNING, cam, geometry.inside)
    assert shining.info.visibility == 1.0 and shining.info.visibility_source == "core"
    # contrast 0.04 over the reference 0.08
    assert shining.reflections.max() == pytest.approx(0.04 / REFERENCE, abs=1e-3)
    assert shining.info.reflections_reference_contrast == REFERENCE
    assert hidden.info.visibility == 0.0
    assert (hidden.reflections == 0).all()
    assert hidden.zone.max() < 0.05  # no glow, no saturated core
    sx, sy = shining.info.sun_pixel_xy or (0.0, 0.0)
    y, x = np.nonzero(shining.reflections > 0.25)
    centre = (geometry.shape[1] - 1) / 2
    facing = (x - centre) * (sx - centre) + (y - centre) * (sy - centre)
    assert (facing > 0).mean() > 0.9  # the lit half of the ring faces the sun


def test_a_mast_on_the_sun_does_not_switch_it_off(geometry: CameraGeometry) -> None:
    """The disc shines past a thin masked object: the core is read on every field pixel."""
    rgb = frame(geometry, MORNING)
    angle = geometry.angle_to(sun_position(MORNING, SITE))
    sx, _ = (int(v) for v in geometry.sun_pixel(angle))
    sky = geometry.inside.copy()
    sky[:, sx - 1 : sx + 1] = False  # a mast through the sun (0.9 deg wide), in the static mask
    result = sun_map(rgb, MORNING, camera(geometry), sky)
    assert result.info.core_saturation == 1.0 and result.info.visibility_source == "core"
    assert result.info.visibility == 1.0
    assert result.zone[(angle > 5) & (angle < 6) & sky].min() > 0.25


def test_the_sun_behind_the_tree_line_keeps_its_glow(geometry: CameraGeometry) -> None:
    """The disc and the core lie on a masked object; the glow over the sky beside decides."""
    rgb = frame(geometry, MORNING)
    angle = geometry.angle_to(sun_position(MORNING, SITE))
    behind = angle < 3.5
    rgb[behind] = 40  # the tree line in front of the sun: dark, nothing saturated
    sky = geometry.inside & ~behind
    result = sun_map(rgb, MORNING, camera(geometry), sky)
    assert result.info.core_saturation == 0.0 and result.info.core_sky == 0.0
    assert result.info.visibility_source == "glow"
    assert result.info.visibility > 0.5
    assert result.zone[(angle > 6) & (angle < 7)].min() > 0.15
    # the same object in front of an overcast sky: no glow, nothing
    dull = frame(geometry, MORNING, sun_visible=False)
    dull[behind] = 40
    hidden = sun_map(dull, MORNING, camera(geometry), sky)
    assert hidden.info.visibility_source == "glow" and hidden.info.visibility == 0.0
    assert (hidden.zone == 0).all()


def test_temporal_median_of_profiles_only_while_the_sun_shines(geometry: CameraGeometry) -> None:
    cam = camera(geometry)
    base = measure_frame(frame(geometry, MORNING), MORNING, cam, geometry.inside)
    assert base.profile is not None
    measures = [
        replace(
            base,
            time=MORNING + timedelta(seconds=30 * k),
            profile=base.profile.with_value(base.profile.value * scale),
        )
        for k, scale in enumerate([1.0, 0.5, 1.0, 1.0, 1.0])
    ]
    value, used = temporal_profile(measures, 1, cam)
    assert used == 4  # +-60 s around frame 1: frames 0..3
    assert value is not None and np.allclose(value, base.profile.value)  # the dip is removed
    dimmed = [*measures[:2], replace(measures[2], visibility=0.3), *measures[3:]]
    value, used = temporal_profile(dimmed, 2, cam)
    assert value is None and used == 1  # the frame's own sun is not shining: its own profile
    value, used = temporal_profile(dimmed, 1, cam)
    assert used == 3  # the dimmed frame does not enter the median


def test_wrong_shapes_are_refused(geometry: CameraGeometry) -> None:
    with pytest.raises(ValueError, match="geometry"):
        sun_map(np.zeros((10, 10, 3), np.uint8), MORNING, camera(geometry), geometry.inside)


def test_a_table_for_other_circles_is_refused(geometry: CameraGeometry) -> None:
    other = optics().scaled((200, 200))
    small = CameraGeometry(*[m[::2, ::2] for m in (geometry.azimuth_deg, geometry.zenith_deg)])
    with pytest.raises(ValueError, match="differ"):
        SunCamera(small, SITE, other, table())
