"""Ring sectors: measured against their flanks, painted with a taper."""

from __future__ import annotations

import numpy as np

from skysun import CameraGeometry, ReflectionParams
from skysun.reflections import RingLayout
from tests_skysun.conftest import CENTRE, RING_RADIUS, optics


def layout(geometry: CameraGeometry) -> RingLayout:
    return RingLayout(optics(), geometry.shape, geometry.inside, ReflectionParams())


def test_a_lit_arc_is_measured_in_its_sectors_only(geometry: CameraGeometry) -> None:
    rings = layout(geometry)
    y, x = np.indices(geometry.shape)
    lum = np.full(geometry.shape, 100.0, dtype=np.float32)
    angle = np.degrees(np.arctan2(y - CENTRE, x - CENTRE)) % 360
    arc = (np.abs(np.hypot(x - CENTRE, y - CENTRE) - RING_RADIUS) <= 1.5) & (angle < 90)
    lum[arc] = 110.0
    measure = rings.measure(lum, geometry.inside)
    value = measure.value(250)[0]
    assert measure.usable[0].all()
    assert (value[:9] > 0.03).all()  # sectors 0-90 deg: 10 % of a band, partly lit
    assert np.abs(value[10:]).max() < 1e-6
    assert measure.contrast()[0, 4] > 0.03


def test_unusable_pixels_leave_a_sector_unmeasured(geometry: CameraGeometry) -> None:
    rings = layout(geometry)
    y, x = np.indices(geometry.shape)
    angle = np.degrees(np.arctan2(y - CENTRE, x - CENTRE)) % 360
    usable = geometry.inside & ~(angle < 10)
    measure = rings.measure(np.full(geometry.shape, 100.0, dtype=np.float32), usable)
    assert not measure.usable[0, 0]
    assert np.isnan(measure.value(250)[0, 0])
    assert measure.usable[0, 1:].all()


def test_paint_keeps_the_value_on_the_line_and_tapers_to_zero(geometry: CameraGeometry) -> None:
    rings = layout(geometry)
    st = rings.structures[0]
    painted = rings.paint([np.full(st.paint.size, 0.5, dtype=np.float32)])
    y, x = np.indices(geometry.shape)
    r = np.hypot(x - CENTRE, y - CENTRE)
    assert painted[np.abs(r - RING_RADIUS) <= 1.9].min() == 0.5
    assert painted[np.abs(r - RING_RADIUS) >= 4.1].max() == 0.0
    edge = painted[(np.abs(r - RING_RADIUS) > 2.3) & (np.abs(r - RING_RADIUS) < 3.7)]
    assert 0 < edge.min() and edge.max() < 0.5
