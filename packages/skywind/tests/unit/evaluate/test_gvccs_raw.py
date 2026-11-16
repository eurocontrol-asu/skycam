from __future__ import annotations

import numpy as np
import pandas as pd

from skywind.evaluate.gvccs_raw import _OFFSETS, _nearest_run, annotation_mask, match_flights


def test_annotation_mask_fills_polygon() -> None:
    m = annotation_mask([[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]], size=32)
    assert m[15, 15] and not m[5, 5]


def test_nearest_run_picks_run_closest_to_zero() -> None:
    hit = np.zeros(len(_OFFSETS), bool)
    centre = len(_OFFSETS) // 2
    hit[centre + 8 : centre + 13] = True  # +200..+300 m
    hit[centre - 60 : centre - 50] = True  # -1500..-1275 m
    assert _nearest_run(hit) == 250.0
    assert np.isnan(_nearest_run(np.zeros(len(_OFFSETS), bool)))


def test_flight_matching_by_position() -> None:
    t = pd.date_range("2024-01-01T10:00", periods=20, freq="5s", tz="UTC")
    tracks = pd.DataFrame(
        {
            "FLIGHT_ID": "g1",
            "CALL_SIGN": "ABC1",
            "ICAO_TYPE": "A320",
            "timestamp": t,
            "LATITUDE": 48.0 + np.arange(20) * 1e-3,
            "LONGITUDE": 2.0,
        }
    )
    sv = pd.DataFrame(
        {
            "timestamp": np.tile(t, 2),
            "flight_id": ["near"] * 20 + ["far"] * 20,
            "callsign": "X",
            "latitude": np.r_[48.0 + np.arange(20) * 1e-3 + 1e-4, 48.5 + np.arange(20) * 1e-3],
            "longitude": 2.0,
        }
    )
    m = match_flights(tracks, sv, ["g1"]).iloc[0]
    assert m["flight_id"] == "near" and bool(m["matched"])
