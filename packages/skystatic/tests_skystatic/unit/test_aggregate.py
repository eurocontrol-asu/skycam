from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pytest

from skystatic.aggregate import (
    aggregate_parts,
    bounded_map,
    central_luminance,
    circle_contrast,
    select_daytime,
)
from skystatic.frames import Frame
from tests_skystatic.conftest import encode, night_frame, scene, zenith_map


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


def test_circle_contrast_is_one_for_a_flat_frame_and_large_for_a_fisheye_frame() -> None:
    zenith = zenith_map()
    sky, beyond = zenith < 60, zenith >= 90
    assert circle_contrast(night_frame().astype(np.float32), sky, beyond) == pytest.approx(1.0)
    assert circle_contrast(scene(0).astype(np.float32), sky, beyond) > 2
    assert circle_contrast(scene(0).astype(np.float32), sky, np.zeros_like(beyond)) is None


def test_select_daytime_rejects_flat_frames_brighter_than_the_day() -> None:
    # Night frames of the site camera: flat (contrast 1) and brighter than the daytime centre.
    levels = [0.25, 0.25, 0.2, 0.22, 0.25]
    contrasts = [1.0, 1.0, 3.0, 2.7, 1.0]
    assert select_daytime(levels, 0.25, 0.0, contrasts, 2.0) == [False, False, True, True, False]


def test_select_daytime_without_contrast_keeps_the_brightness_rule() -> None:
    assert select_daytime([0.01, 0.5], 0.25, 0.0, [None, None], 2.0) == [False, True]


def test_a_dim_overcast_frame_with_a_visible_circle_is_daytime() -> None:
    # Brightness varies by a factor of four or more under clouds; the circle stays visible.
    levels = [0.9, 0.05, 0.4]
    contrasts = [3.0, 2.5, 1.0]
    assert select_daytime(levels, 0.25, 0.0, contrasts, 2.0) == [True, True, False]


def test_the_floor_applies_to_every_frame() -> None:
    assert select_daytime([0.05, 0.5], 0.25, 0.1, [3.0, 3.0], 2.0) == [False, True]


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


def test_bounded_map_keeps_order_and_few_results_in_memory() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def work(i: int) -> int:
        with lock:
            started.append(i)
        return i * i

    with ThreadPoolExecutor(4) as pool:
        consumed = []
        for result in bounded_map(pool, work, range(50), ahead=3):
            # Never more than `ahead` tasks submitted beyond what has been consumed.
            assert len(started) <= len(consumed) + 3
            consumed.append(result)
    assert consumed == [i * i for i in range(50)]


class TimedSource(MemorySource):
    """Frames at given clock times."""

    def __init__(self, images: list[np.ndarray], times: list[datetime]) -> None:
        super().__init__(images)
        self._times = times

    def frames(self) -> list[Frame]:
        return [Frame(t, n) for t, n in zip(self._times, self._data, strict=True)]


def test_hourly_windows_average_luminance_and_drop_stubs() -> None:
    times = [datetime(2024, 1, 1, 9, 59)] + [
        datetime(2024, 1, 1, 10 + i // 4, (i % 4) * 15) for i in range(8)
    ]
    images = [flat(200)] + [flat(40)] * 4 + [flat(160)] * 4
    source = TimedSource(images, times)
    result = aggregate_parts(source, source.frames(), (8, 8), parts=1, workers=2)
    # 09:59 holds one frame against four in each full hour: a stub, dropped.
    assert result.window_counts == (4, 4)
    assert [t.hour for t in result.window_starts] == [10, 11]
    luminance = float(np.array([0.2126, 0.7152, 0.0722]).sum())
    levels = np.exp(result.windows[:, 0]).mean(axis=(1, 2))
    assert np.allclose(levels, [40 * luminance, 160 * luminance], rtol=1e-4)


def test_a_bright_magenta_surround_does_not_hide_the_daylight_disk() -> None:
    zenith = zenith_map()
    sky, beyond = zenith < 60, zenith >= 90
    frame = np.full((96, 96, 3), 110, np.float32)  # grey daylight sky
    frame[beyond] = (96, 51, 84)  # the camera's magenta fill
    luminance_ratio = 110 / float(np.array([96, 51, 84]) @ np.array([0.2126, 0.7152, 0.0722]))
    assert luminance_ratio < 2
    assert circle_contrast(frame, sky, beyond) >= 2  # green: 110 / 51
    assert circle_contrast(night_frame().astype(np.float32), sky, beyond) == pytest.approx(1.0)
