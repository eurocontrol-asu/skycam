from __future__ import annotations

import numpy as np

from skycloud.engine import FrameInputs, render_frame
from skycloud.params import CloudMapParams
from skycloud.tables import ReferenceTables
from tests_skycloud.unit.test_geometry import fisheye

P = CloudMapParams(unseen_sky="legacy", unknown_state="mask")


def _inputs(geo_shape: tuple[int, int], szen: float) -> FrameInputs:
    return FrameInputs(
        saz=180.0, szen=szen, sky_mask=np.ones(geo_shape, bool), level={}, offset=None,
        offset_note="no gain ledger", neighbours={-30: lambda: None, 30: lambda: None},
    )  # fmt: skip


def _sky(size: int) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    img[...] = (60, 110, 200)  # blue sky
    img[:4, :4] = img[:4, -4:] = img[-4:, :4] = img[-4:, -4:] = (6, 3, 5)  # dark corners: pedestal
    c = size // 2
    img[c - 12 : c + 12, c + 20 : c + 44] = (215, 220, 225)  # a white cloud
    return img


def test_render_frame_values_the_sky_and_not_the_outside(tiny_tables: ReferenceTables) -> None:
    geo = fisheye(161)
    res = render_frame(_sky(161), "20240101120000", _inputs(geo.shape, 50.0), geo, tiny_tables, P)
    inside = geo.inside
    assert np.isnan(res.alpha[~inside]).all()
    assert np.isfinite(res.alpha[inside & (geo.zenith_deg < 80)]).mean() > 0.9
    assert (res.conf >= 0).all() and (res.conf <= 1).all()
    assert (
        res.alpha[80, 80 + 32] > res.alpha[40, 80]
    )  # the white cloud reads cloudier than the blue sky
    assert res.known is None and res.cert is None  # daytime: no knowability, no certification
    assert res.lens is not None and not res.lens.any()


def test_no_value_below_the_minimum_sun_elevation(tiny_tables: ReferenceTables) -> None:
    geo = fisheye(81)
    res = render_frame(_sky(81), "20240101000000", _inputs(geo.shape, 95.0), geo, tiny_tables, P)
    assert np.isnan(res.alpha).all() and (res.conf == 0).all() and "no_value" in res.info
    assert res.lens is None
