from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from skycloud.params import CloudMapParams
from skycloud.unknown import (
    UnknownContext,
    _clean,
    _clean_valued,
    _deck_guard,
    _drop_unresolved,
    _fill_islands,
    apply_unknown,
)

P = CloudMapParams()


def test_clean_absorbs_small_components_of_both_states() -> None:
    m = np.zeros((50, 50), bool)
    m[10:12, 10:12] = True  # small unknown speck
    m[30:50, :] = True
    m[40:42, 5:7] = False  # small known hole
    out = _clean(m, 10)
    assert not out[10, 10] and out[40, 5] and out[45, 30]


def test_clean_valued_never_creates_a_value() -> None:
    v = np.zeros((30, 30), bool)
    v[5:7, 5:7] = True
    v[10:30, 10:30] = True
    out = _clean_valued(v, 10)
    assert not out[5, 5] and out[20, 20] and not out[0, 0]


def test_deck_guard_hides_mid_readings_of_a_gap_free_deck() -> None:
    v = np.array([[30, 60, 200]], np.uint8)
    has = np.ones((1, 3), bool)
    unk = np.zeros((1, 3), bool)
    assert _deck_guard(unk, has, v, 0.6, P).tolist() == [[False, True, False]]
    assert not _deck_guard(unk, has, v, 0.4, P).any()


def test_unresolved_specks_go_unknown_but_lines_stay() -> None:
    unk = np.ones((120, 120), bool)
    has = np.ones((120, 120), bool)
    unk[10:14, 10:14] = False  # speck
    unk[60, 10:110] = False  # a 100 px line
    out = _drop_unresolved(unk, has, P)
    assert out[11, 11] and not out[60, 50]


def test_enclosed_cloud_hole_takes_its_body() -> None:
    has = np.ones((80, 80), bool)
    unk = np.zeros((80, 80), bool)
    unk[38:42, 38:42] = True
    v = np.full((80, 80), 200, np.uint8)
    c = np.full((80, 80), 0.6, np.float32)
    c[38:42, 38:42] = 0.1
    unk2, c2 = _fill_islands(unk, has, v, c, P)
    assert not unk2.any() and np.isclose(c2[40, 40], 0.6)


def _one_frame_folder(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A rendered folder with one daytime frame: a valued sky whose left half has a confidence
    under the contract."""
    value = np.full((256, 256), 100, np.uint8)
    value[:, :8] = 255  # not sky
    conf = np.full((256, 256), 230, np.uint8)
    conf[:, :128] = 20
    Image.fromarray(value).save(tmp_path / "20240101120000.png")
    Image.fromarray(conf).save(tmp_path / "20240101120000_conf.png")
    return tmp_path, value


def _context(state: Literal["layer", "mask"]) -> UnknownContext:
    return UnknownContext(
        p=CloudMapParams(unknown_state=state),
        same=lambda _a, _b, shape: np.zeros(shape, bool),
        elev=lambda _t: 30.0,
        zen99=np.full((256, 256), 30.0, np.float32),
    )


def test_unknown_layer_keeps_the_value_and_writes_the_verdict(tmp_path: Path) -> None:
    out, value = _one_frame_folder(tmp_path)
    share = apply_unknown(out, ["20240101120000"], {}, _context("layer"))["20240101120000"]
    delivered = np.asarray(Image.open(out / "20240101120000.png"))
    verdict = np.asarray(Image.open(out / "_unknown" / "20240101120000.png"))
    assert np.array_equal(delivered, value)  # nothing removed by a low confidence
    assert share is not None and share > 0.3
    assert verdict[:, 16:100].all() and not verdict[:, 160:].any()
    assert not verdict[:, :8].any()  # the verdict is about sky only


def test_unknown_mask_still_blanks_the_value(tmp_path: Path) -> None:
    out, value = _one_frame_folder(tmp_path)
    apply_unknown(out, ["20240101120000"], {}, _context("mask"))
    delivered = np.asarray(Image.open(out / "20240101120000.png"))
    assert (delivered[:, 16:100] == 255).all() and (delivered[:, 160:] == 100).all()
    assert not (out / "_unknown").exists()
    assert np.array_equal(np.asarray(Image.open(out / "_unmasked" / "20240101120000.png")), value)
