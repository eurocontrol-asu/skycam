"""The 8-bit encoding of the products."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from skysun.frames import FrameFile
from skysun.products import decode, encode, frame_paths


def test_code_is_round_255_times_value_and_decodes_back() -> None:
    value = np.array([0.0, 0.002, 0.5, 0.999, 1.0, 1.7, -0.3], dtype=np.float32)
    code = encode(value)
    assert code.tolist() == [0, 1, 128, 255, 255, 255, 0]
    assert np.abs(decode(code)[:5] - value[:5]).max() <= 1 / 510 + 1e-7


def test_products_are_grouped_by_utc_date() -> None:
    f = FrameFile(Path("image_20240115235930.jp2"), datetime(2024, 1, 15, 23, 59, 30, tzinfo=UTC))
    paths = frame_paths(Path("out"), f)
    assert set(paths) == {"zone", "reflections", "json"}  # two layers, no combined map
    assert paths["zone"] == Path("out/2024-01-15/zone_20240115235930.png")
    assert paths["json"] == Path("out/2024-01-15/sun_20240115235930.json")
    assert paths["reflections"].name == "reflections_20240115235930.png"
