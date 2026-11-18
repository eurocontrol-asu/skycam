from __future__ import annotations

import numpy as np
import pytest

from skystatic.masks import class_map, hysteresis, installation_mask, variable_mask


def day(*pixels: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((5, 5), bool)
    for pixel in pixels:
        mask[pixel] = True
    return mask


def test_installation_needs_most_reference_days() -> None:
    days = [day((0, 0), (1, 1)), day((0, 0)), day((0, 0), (2, 2)), day((0, 0)), day((0, 0))]
    installation = installation_mask(days, min_fraction=0.8)
    assert installation[0, 0]
    assert not installation[1, 1] and not installation[2, 2]


def test_installation_tolerates_one_missed_day_out_of_five() -> None:
    days = [day((0, 0))] * 4 + [day()]
    assert installation_mask(days, min_fraction=0.8)[0, 0]
    assert not installation_mask(days, min_fraction=1.0)[0, 0]


def test_installation_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="reference day"):
        installation_mask([], 0.8)
    with pytest.raises(ValueError, match="min_fraction"):
        installation_mask([day()], 0.0)


def test_variable_is_what_the_day_adds_beyond_the_slack() -> None:
    installation = day((2, 2))
    today = day((2, 2), (2, 3), (0, 0))
    assert variable_mask(today, installation, tolerance=0).tolist() == day((2, 3), (0, 0)).tolist()
    assert variable_mask(today, installation, tolerance=1).tolist() == day((0, 0)).tolist()


def test_class_map_labels() -> None:
    domain = np.ones((5, 5), bool)
    domain[4] = False
    classes = class_map(domain, day((0, 0), (4, 4)), day((1, 1)))
    assert classes[0, 0] == 1 and classes[1, 1] == 2 and classes[2, 2] == 0
    assert classes[4, 4] == 3  # outside wins over installation


def test_a_mark_touching_the_installation_is_kept_whole_and_edge_jitter_is_dropped() -> None:
    installation = np.zeros((40, 40), bool)
    installation[:, :10] = True
    day = installation.copy()
    day[5:15, 10:12] = True  # jitter: a 2-px strip along the edge
    day[20:30, 10:20] = True  # a mark touching the installation
    variable = variable_mask(day, installation, tolerance=2)
    assert not variable[5:15, 10:12].any()
    assert variable[20:30, 10:20].all()  # no ring of sky inside the mark


def test_sky_enclosed_by_an_object_takes_its_class_but_not_across_a_ring_line() -> None:
    from skystatic.masks import enclosed_sky

    classes = np.zeros((40, 40), np.uint8)
    classes[30:38, 5:15] = 1  # a building
    classes[32:36, 8:12] = 0  # its white face, read as sky
    classes[:, 20] = 1  # a thin ring line of the optics crossing the sky
    core = np.zeros((40, 40), bool)
    core[:10, :10] = True
    ring = np.zeros((40, 40), bool)
    ring[:, 20] = True
    out = enclosed_sky(classes, core, ring)
    assert (out[32:36, 8:12] == 1).all()
    assert (out[:, 25:] == 0).all()  # sky beyond the ring line stays sky


def test_hysteresis_keeps_weak_pixels_connected_to_a_seed_only() -> None:
    seeds = np.zeros((10, 10), bool)
    seeds[0, 0] = True
    weak = np.zeros((10, 10), bool)
    weak[1, 1:6] = True  # attached diagonally to the seed
    weak[8, 8] = True  # alone
    kept = hysteresis(seeds, weak)
    assert kept[0, 0] and kept[1, 1:6].all()
    assert not kept[8, 8]
