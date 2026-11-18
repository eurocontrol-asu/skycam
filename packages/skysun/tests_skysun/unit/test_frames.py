"""Frame times come from the file names, in UTC."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from skysun.frames import frame_time


def test_time_is_the_last_stamp_of_the_name() -> None:
    assert frame_time(Path("2024/image_20240115110030.jp2")) == datetime(
        2024, 1, 15, 11, 0, 30, tzinfo=UTC
    )


def test_a_name_without_a_stamp_is_refused() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        frame_time(Path("image.jp2"))
