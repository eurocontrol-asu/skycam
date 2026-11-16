from __future__ import annotations

import numpy as np
import pytest

from skystatic.masks import class_map, installation_mask, variable_mask


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
