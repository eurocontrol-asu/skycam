from __future__ import annotations

import math

import numpy as np
import pytest

from skystatic.evaluate import agreement


def box(rows: slice, cols: slice) -> np.ndarray:
    mask = np.zeros((20, 20), bool)
    mask[rows, cols] = True
    return mask


def test_identical_masks_agree_fully() -> None:
    mask = box(slice(5, 10), slice(5, 10))
    result = agreement(mask, mask, np.ones((20, 20), bool))
    assert result.iou == result.precision == result.recall == 1.0
    assert result.computed_fraction == pytest.approx(25 / 400)


def test_a_shifted_border_is_forgiven_only_by_the_tolerant_scores() -> None:
    computed = box(slice(5, 10), slice(5, 10))
    reference = box(slice(5, 10), slice(7, 12))  # shifted by two columns
    result = agreement(computed, reference, np.ones((20, 20), bool), tolerance=2)
    assert result.precision == pytest.approx(0.6)
    assert result.recall == pytest.approx(0.6)
    assert result.iou == pytest.approx(15 / 35)
    assert result.precision_within_tolerance == 1.0
    assert result.recall_within_tolerance == 1.0


def test_pixels_outside_the_domain_are_ignored() -> None:
    domain = box(slice(0, 20), slice(0, 10))
    computed = box(slice(5, 10), slice(5, 15))
    reference = box(slice(5, 10), slice(5, 10))
    assert agreement(computed, reference, domain).iou == 1.0


def test_empty_masks_give_nan_ratios() -> None:
    empty = np.zeros((20, 20), bool)
    result = agreement(empty, empty, np.ones((20, 20), bool))
    assert math.isnan(result.iou) and math.isnan(result.precision)
    assert result.computed_fraction == 0.0
