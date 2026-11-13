from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skywind.field.heading import combine_offsets, offset_terms


def _obs(offset_deg: float, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    hdg_true = rng.uniform(0, 360, n)
    tas, u, v = 230.0, 20.0, -5.0
    ax, ay = tas * np.sin(np.radians(hdg_true)), tas * np.cos(np.radians(hdg_true))
    gx, gy = ax + u, ay + v
    return pd.DataFrame({"icao24": "abc", "typecode": "A320", "trk": np.degrees(np.arctan2(gx, gy)) % 360,
                         "gs": np.hypot(gx, gy) / 0.514444, "hdg_mag": (hdg_true - offset_deg - 1.0) % 360,
                         "decl": 1.0, "u": u, "v": v})


def test_offset_recovered_with_exact_reference_wind() -> None:
    w = _obs(-1.3)
    terms = offset_terms(w, w["u"].to_numpy(), w["v"].to_numpy()).assign(day="d1")
    off = combine_offsets(terms)
    assert off.loc["abc", "offset_deg"] == pytest.approx(-1.3, abs=1e-6)
    assert off.loc["abc", "n_obs"] == 40


def test_leave_day_out_and_minimum_count() -> None:
    a = offset_terms(_obs(-1.0), np.full(40, 20.0), np.full(40, -5.0)).assign(day="d1")
    b = offset_terms(_obs(-2.0), np.full(40, 20.0), np.full(40, -5.0)).assign(day="d2")
    table = pd.concat([a, b])
    assert combine_offsets(table, exclude_days=("d2",)).loc["abc", "offset_deg"] == pytest.approx(-1.0, abs=1e-6)
    assert combine_offsets(table).loc["abc", "n_days"] == 2
    assert combine_offsets(table, min_obs=100).empty
