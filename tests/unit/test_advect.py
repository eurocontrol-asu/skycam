from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from skywind.advect import advect, emission_points, met_dataset


def test_uniform_wind_displacement(era5_ds: xr.Dataset) -> None:
    met = met_dataset(era5_ds)
    pts = pd.DataFrame(
        {
            "flight_id": ["f"],
            "waypoint": [0],
            "latitude": [48.6],
            "longitude": [2.35],
            "level": [250.0],
            "emission_time": [pd.Timestamp("2024-03-18T12:10", tz="UTC")],
        }
    )
    out = advect(pts, met, pd.Timestamp("2024-03-18T12:40", tz="UTC"), max_age_min=60)
    last = out.sort_values("time").iloc[-1]
    assert last["time"] == pd.Timestamp("2024-03-18T12:40", tz="UTC")
    dt = 30 * 60
    assert (last["latitude"] - 48.6) * np.pi / 180 * 6371e3 == pytest.approx(-5.0 * dt, rel=1e-2)
    assert (last["longitude"] - 2.35) * np.pi / 180 * 6371e3 * np.cos(np.radians(48.6)) == pytest.approx(
        10 * dt, rel=1e-2
    )


def test_emission_points_from_state_vectors() -> None:
    sv = pd.DataFrame(
        {
            "flight_id": ["a", "a"],
            "timestamp": pd.to_datetime(["2024-01-01T00:00", "2024-01-01T00:01"], utc=True),
            "latitude": [48.0, 48.1],
            "longitude": [2.0, 2.1],
            "altitude": [36000.0, 10000.0],
        }
    )
    pts = emission_points(sv)
    assert len(pts) == 1 and 220 < pts["level"].iloc[0] < 235
