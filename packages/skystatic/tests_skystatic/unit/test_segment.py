from __future__ import annotations

import numpy as np
import pytest

from skystatic.segment import edge_strength, sky_region, static_mask, static_mask_all_parts
from tests_skystatic.conftest import DIRT, SIZE, STRUCTURE, scene, zenith_map


def averaged(dirty: bool = False, frames: int = 32) -> np.ndarray:
    return np.mean([scene(i, dirty=dirty) for i in range(frames)], axis=0).astype(np.float32)


@pytest.fixture
def domain() -> np.ndarray:
    return zenith_map() < 90


@pytest.fixture
def core() -> np.ndarray:
    return zenith_map() < 40


def test_edge_strength_does_not_depend_on_exposure(core: np.ndarray) -> None:
    image = averaged()
    assert np.allclose(edge_strength(image, core), edge_strength(image * 0.1, core), atol=1e-4)


def test_a_steady_gradient_is_not_an_edge_but_a_step_is(core: np.ndarray) -> None:
    x = np.indices((SIZE, SIZE))[1]
    ramp = np.repeat((40 * np.exp(x / 30))[..., None], 3, axis=2).astype(np.float32)
    raw = edge_strength(ramp, core, background=0)
    local = edge_strength(ramp, core, background=36)
    assert raw[48, 20:76].min() > 0.05  # steep enough to pass for an edge everywhere
    assert local[48, 20:76].max() < 0.01
    step = ramp.copy()
    step[:, 60:] *= 0.5
    assert edge_strength(step, core, background=36)[48, 58:62].max() > 0.3


def test_edge_strength_rejects_a_black_sky(core: np.ndarray) -> None:
    with pytest.raises(ValueError, match="black"):
        edge_strength(np.zeros((SIZE, SIZE, 3), np.float32), core)


def test_sky_region_is_the_component_holding_the_core() -> None:
    smooth = np.zeros((6, 6), bool)
    smooth[0:2, 0:2] = True  # small isolated patch
    smooth[3:6, 2:6] = True  # holds the core
    core = np.zeros((6, 6), bool)
    core[4, 4] = True
    region = sky_region(smooth, core)
    assert region[4, 4] and not region[0, 0]
    assert not sky_region(np.zeros((6, 6), bool), core).any()


def test_fixed_structure_is_static_and_the_averaged_sky_is_not(
    domain: np.ndarray, core: np.ndarray
) -> None:
    mask = static_mask(averaged(), domain, core, edge=0.05)
    assert mask[STRUCTURE].all()
    assert mask.sum() < 300  # the mast (80 px) and its outline, out of ~6000 sky pixels
    assert not mask[30, 20:60].any()  # the path of the moving cloud is sky


def test_dirt_on_the_optics_is_static_on_that_day(domain: np.ndarray, core: np.ndarray) -> None:
    clean = static_mask(averaged(), domain, core, edge=0.05)
    dirty = static_mask(averaged(dirty=True), domain, core, edge=0.05)
    assert dirty[DIRT].all()
    assert not clean[DIRT].any()


def test_a_cloud_parked_during_one_part_is_not_static(domain: np.ndarray, core: np.ndarray) -> None:
    moving = averaged()
    parked = moving.copy()
    parked[20:40, 20:40] += 60  # sharp-edged bank that did not move during that part
    one_part = static_mask(parked, domain, core, edge=0.05)
    all_parts = static_mask_all_parts(np.stack([parked, moving, moving]), domain, core, edge=0.05)
    assert one_part[20, 20:40].any()
    assert not all_parts[18:42, 18:42].any()
    assert all_parts[STRUCTURE].all()
