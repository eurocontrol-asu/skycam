from __future__ import annotations

from pathlib import Path

import numpy as np

from skycloud.images import frame_path, pedestal, shift_stamp, stamp_minutes, when_of


def test_stamps() -> None:
    assert shift_stamp("20240101235945", 30) == "20240102000015"
    assert when_of("20240101120000").tzinfo is not None
    assert stamp_minutes("20240101120100") - stamp_minutes("20240101120000") == 1.0
    assert frame_path(Path("/r"), "20240102030405") == Path(
        "/r/2024/01/02/image_20240102030405.jp2"
    )


def test_pedestal_is_the_mean_of_the_four_corners() -> None:
    img = np.zeros((200, 200, 3), np.uint8)
    img[:60, :60] = 8
    img[-60:, -60:] = 16
    assert np.allclose(pedestal(img), [6.0, 6.0, 6.0])
