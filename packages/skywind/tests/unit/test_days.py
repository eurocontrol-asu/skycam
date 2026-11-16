from __future__ import annotations

import pandas as pd
import pytest

from skywind.days import DayInfo, parse_day, parse_windows


def test_parse_day() -> None:
    assert parse_day("2024-03-18") == parse_day("20240318") == "20240318"
    with pytest.raises(ValueError, match="not a day"):
        parse_day("18/03/2024")


def test_parse_windows() -> None:
    w = parse_windows("20240318", "15:30-17:30; 18:00:30-18:45")
    assert w[0] == (pd.Timestamp("2024-03-18T15:30", tz="UTC"), pd.Timestamp("2024-03-18T17:30", tz="UTC"))
    assert DayInfo("20240318", tuple(w)).stop == pd.Timestamp("2024-03-18T18:45", tz="UTC")
    with pytest.raises(ValueError, match="no HH:MM"):
        parse_windows("20240318", "afternoon")
