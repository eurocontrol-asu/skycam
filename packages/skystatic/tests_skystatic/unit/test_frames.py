from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from PIL import Image

from skystatic.frames import decode_rgb, parse_timestamp


def test_parse_timestamp_reads_fourteen_digits() -> None:
    assert parse_timestamp("image_20231006123000.jp2") == datetime(2023, 10, 6, 12, 30)
    assert parse_timestamp("CAM/2022/04/20/20220420152200_160.jpg") == datetime(2022, 4, 20, 15, 22)


def test_parse_timestamp_rejects_missing_or_invalid_dates() -> None:
    assert parse_timestamp("notes.txt") is None
    assert parse_timestamp("image_20231345999999.jpg") is None


def test_decode_rgb_resizes_to_requested_size() -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.full((40, 60, 3), 200, np.uint8)).save(buffer, format="PNG")
    rgb = decode_rgb(buffer.getvalue(), (30, 20))
    assert rgb is not None
    assert rgb.shape == (20, 30, 3)
    assert rgb.dtype == np.float32
    assert np.allclose(rgb, 200)


def test_decode_rgb_returns_none_on_empty_or_truncated_data() -> None:
    assert decode_rgb(b"", (10, 10)) is None
    assert decode_rgb(b"\xff\xd8\xff not a jpeg", (10, 10)) is None
