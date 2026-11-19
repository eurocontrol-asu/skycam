"""The sun handling: fill gives a value everywhere near the sun, legacy leaves the hole."""

from __future__ import annotations

import numpy as np
import pytest

from skycloud.engine import FrameInputs, FrameResult, render_frame
from skycloud.geometry import CameraGeometry, angle_to_sun
from skycloud.params import CloudMapParams
from skycloud.sun import SunArea
from skycloud.tables import ReferenceTables
from tests_skycloud.unit.test_geometry import fisheye

SIZE, SAZ, SZEN = 241, 180.0, 40.0


def _frame() -> tuple[CameraGeometry, np.ndarray, SunArea]:
    geo = fisheye(SIZE)
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[...] = (60, 110, 200)  # blue sky
    for cy, cx in ((0, 0), (0, -6), (-6, 0), (-6, -6)):
        img[cy : cy + 6 or None, cx : cx + 6 or None] = (6, 3, 5)  # dark corners: the pedestal
    angle = angle_to_sun(geo.azimuth_deg, geo.zenith_deg, SAZ, SZEN).astype(np.float32)
    angle[~geo.inside] = 180.0
    core = geo.inside & (angle < 4.0)
    img[core] = 255  # the saturated sun
    return geo, img, SunArea(np.asarray(angle), core)


def _render(mode: str, tiny_tables: ReferenceTables) -> tuple[FrameResult, SunArea, CameraGeometry]:
    geo, img, area = _frame()
    x = FrameInputs(
        saz=SAZ, szen=SZEN, sky_mask=np.ones(geo.shape, bool), level={}, offset=None,
        offset_note="", neighbours={-30: lambda: None, 30: lambda: None}, sun_area=area,
    )  # fmt: skip
    p = CloudMapParams(unseen_sky=mode)
    return render_frame(img, "20240101120000", x, geo, tiny_tables, p), area, geo


def test_no_hole_near_the_sun_and_the_core_takes_its_ring(tiny_tables: ReferenceTables) -> None:
    new, area, geo = _render("fill", tiny_tables)
    old, _, _ = _render("legacy", tiny_tables)
    near = geo.inside & (area.angle < 30)
    assert np.isnan(old.alpha[area.core]).all()  # legacy: a hole on the sun
    assert np.isfinite(new.alpha[near]).all()  # fill: none
    ring = geo.inside & (area.angle >= 6) & (area.angle < 8)
    core_value = float(np.median(new.alpha[area.core]))
    assert abs(core_value - float(np.median(new.alpha[ring]))) < 0.05
    assert core_value < 0.5  # the white of the sensor is not read as thick cloud
    assert new.info["fill"]["unfilled_px"] == 0
    assert new.info["fill"]["core_px"] == int(area.core.sum())


def test_nothing_changes_beyond_the_sun_area(tiny_tables: ReferenceTables) -> None:
    new, area, geo = _render("fill", tiny_tables)
    old, _, _ = _render("legacy", tiny_tables)
    far = area.angle >= 30
    assert np.array_equal(new.alpha[far], old.alpha[far], equal_nan=True)
    assert np.array_equal(new.conf[far], old.conf[far])
    measured = geo.inside & (area.angle < 30) & np.isfinite(old.alpha)
    assert np.array_equal(new.alpha[measured], old.alpha[measured])  # the aureole is as measured
    assert "fill" not in old.info


def test_fill_needs_the_sun_area(tiny_tables: ReferenceTables) -> None:
    geo, img, _ = _frame()
    x = FrameInputs(
        saz=SAZ, szen=SZEN, sky_mask=np.ones(geo.shape, bool), level={}, offset=None,
        offset_note="", neighbours={-30: lambda: None, 30: lambda: None},
    )  # fmt: skip
    with pytest.raises(ValueError, match="sun_area"):
        render_frame(img, "20240101120000", x, geo, tiny_tables, CloudMapParams())


def test_variable_static_is_filled_from_the_sky_around_it(tiny_tables: ReferenceTables) -> None:
    geo, img, area = _frame()
    spoilt = np.zeros(geo.shape, bool)
    spoilt[30:36, 110:116] = True  # a mark far from the sun
    img[spoilt] = (20, 20, 20)  # dark dirt: it would read as anything but the sky
    sky = ~spoilt
    out = {}
    for mode in ("fill", "legacy"):
        x = FrameInputs(
            saz=SAZ, szen=SZEN, sky_mask=sky, level={}, offset=None, offset_note="",
            neighbours={-30: lambda: None, 30: lambda: None}, sun_area=area, spoilt=spoilt,
        )  # fmt: skip
        p = CloudMapParams(unseen_sky=mode)
        out[mode] = render_frame(img, "20240101120000", x, geo, tiny_tables, p)
    assert np.isnan(out["legacy"].alpha[spoilt]).all()
    ring = np.zeros(geo.shape, bool)
    ring[26:40, 106:120] = True
    ring &= ~spoilt
    assert np.allclose(out["fill"].alpha[spoilt], np.median(out["fill"].alpha[ring]), atol=0.05)
    assert np.array_equal(out["fill"].alpha[ring], out["legacy"].alpha[ring])  # never read
