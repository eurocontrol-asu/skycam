from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from skystatic.aggregate import aggregate_parts, central_luminance, select_daytime
from skystatic.frames import Frame
from tests_skystatic.conftest import encode


class MemorySource:
    """Frames held in memory."""

    label = "memory"

    def __init__(self, images: list[np.ndarray | None]) -> None:
        self._data = {
            f"image_202401011200{i:02d}.png": b"" if im is None else encode(im)
            for i, im in enumerate(images)
        }

    def frames(self) -> list[Frame]:
        return [Frame(datetime(2024, 1, 1, 12, 0, i), n) for i, n in enumerate(self._data)]

    def read(self, frame: Frame) -> bytes:
        return self._data[frame.name]


def flat(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, np.uint8)


def test_central_luminance_is_a_fraction_of_full_scale() -> None:
    assert central_luminance(np.full((9, 9, 3), 255, np.float32)) == pytest.approx(1.0)
    assert central_luminance(np.zeros((9, 9, 3), np.float32)) == 0.0


def test_select_daytime_is_relative_to_the_bright_frames() -> None:
    assert select_daytime([0.01, 0.5, 0.6, None], relative=0.25, floor=0.0) == [
        False,
        True,
        True,
        False,
    ]


def test_select_daytime_adapts_to_a_dark_exposure() -> None:
    # Same scene at a short exposure: the absolute levels are low but the day is still found.
    assert select_daytime([0.0005, 0.04, 0.05], relative=0.25, floor=0.0) == [False, True, True]


def test_select_daytime_floor_rejects_a_dark_batch() -> None:
    assert select_daytime([0.01, 0.02], relative=0.25, floor=0.1) == [False, False]
    assert select_daytime([None], relative=0.25, floor=0.0) == [False]


def test_means_of_constant_frames_are_that_constant() -> None:
    source = MemorySource([flat(100)] * 4)
    result = aggregate_parts(source, source.frames(), (8, 8), parts=2, workers=2)
    assert result.counts == (2, 2)
    assert np.allclose(result.mean, 100)
    assert np.allclose(result.geometric, 100, atol=1e-3)


def test_geometric_mean_is_pulled_down_by_dark_moments() -> None:
    source = MemorySource([flat(200), flat(200), flat(200), flat(10)])
    result = aggregate_parts(source, source.frames(), (8, 8), parts=1, workers=2)
    assert result.mean[0, 0, 0, 0] == pytest.approx(152.5)
    # (201^3 * 11)^(1/4) - 1
    assert result.geometric[0, 0, 0, 0] == pytest.approx((201**3 * 11) ** 0.25 - 1, rel=1e-4)
    assert result.geometric[0, 0, 0, 0] < result.mean[0, 0, 0, 0]


def test_undecodable_frames_are_skipped_and_not_counted() -> None:
    source = MemorySource([flat(50), None, flat(150)])
    result = aggregate_parts(source, source.frames(), (8, 8), parts=1, workers=2)
    assert result.counts == (2,)
    assert np.allclose(result.mean, 100)


def test_parts_are_consecutive_groups() -> None:
    source = MemorySource([flat(10), flat(10), flat(90), flat(90)])
    result = aggregate_parts(source, source.frames(), (8, 8), parts=2, workers=1)
    assert np.allclose(result.mean[0], 10)
    assert np.allclose(result.mean[1], 90)


def test_too_few_frames_or_an_empty_part_is_an_error() -> None:
    source = MemorySource([flat(10)])
    with pytest.raises(ValueError, match="cannot be cut"):
        aggregate_parts(source, source.frames(), (8, 8), parts=2)
    broken = MemorySource([None, flat(10)])
    with pytest.raises(ValueError, match="no decodable frame"):
        aggregate_parts(broken, broken.frames(), (8, 8), parts=2)
