from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from skywind.field.heading import TABLE_VERSION, combine_offsets, update_table

pytestmark = pytest.mark.integration


def test_table_accumulates_and_replaces_days(tmp_path: Path) -> None:
    path = tmp_path / "offsets.parquet"
    terms = pd.DataFrame({"icao24": ["a"], "n": [30], "sum_sin": [-0.3], "sum_cos": [29.9], "typecode": ["A320"]})
    update_table(path, "d1", terms, "era5")
    update_table(path, "d2", terms, "era5")
    table = update_table(path, "d1", terms.assign(n=40, sum_sin=-0.4, sum_cos=39.9), "field")
    assert sorted(table["day"]) == ["d1", "d2"] and set(table["version"]) == {TABLE_VERSION}
    off = combine_offsets(pd.read_parquet(path))
    assert off.loc["a", "n_obs"] == 70 and off.loc["a", "n_days"] == 2
