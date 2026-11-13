from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skywind.config import ObservationConfig
from skywind.observations import apply_heading_offsets, bin_observations, wind_from_vectors


def test_wind_triangle() -> None:
    # heading north at 450 kt, ground track east of north: wind from the west
    u, v = wind_from_vectors(450.0, 45.0, 450.0, 0.0)
    assert u == pytest.approx(450 * 0.514444 * np.sin(np.radians(45)))
    assert v == pytest.approx(450 * 0.514444 * (np.cos(np.radians(45)) - 1))


def _messages(roll: float = 0.0, hdg_jitter: float = 0.0) -> pd.DataFrame:
    rows = []
    for i in range(4):
        common = {"t": 1_700_000_010.0 + 5 * i, "flight_id": "f", "icao24": "abc", "callsign": "X", "typecode": "A320",
                  "latitude": 48.6, "longitude": 2.35, "altitude": 36000.0, "geoaltitude": 37000.0,
                  "groundspeed": 450.0, "trk_sv": 90.0}
        rows.append({**common, "bds": "50", "gs50": 450.0, "tas50": 440.0, "roll": roll, "trk50": 90.0,
                     "mach60": np.nan, "ias60": np.nan, "hdg60": np.nan, "vrb60": np.nan})
        rows.append({**common, "bds": "60", "gs50": np.nan, "tas50": np.nan, "roll": np.nan, "trk50": np.nan,
                     "mach60": 0.78, "ias60": 262.0, "hdg60": 88.0 + hdg_jitter * (-1) ** i, "vrb60": 0.0})
    return pd.DataFrame(rows)


def test_bin_observation_from_steady_messages() -> None:
    w = bin_observations(_messages(), "20240318", ObservationConfig(), 25000.0)
    assert len(w) == 1
    assert w["hdg"].iloc[0] == pytest.approx((88.0 + w["decl"].iloc[0]) % 360)
    assert np.isfinite(w["u_obs"].iloc[0])


@pytest.mark.parametrize(("roll", "jitter"), [(10.0, 0.0), (0.0, 3.0)])
def test_turning_or_unsteady_bins_rejected(roll: float, jitter: float) -> None:
    assert bin_observations(_messages(roll, jitter), "20240318", ObservationConfig(), 25000.0).empty


def test_heading_offset_changes_cross_component() -> None:
    w = bin_observations(_messages(), "20240318", ObservationConfig(), 25000.0)
    w = w.assign(u_era=0.0, v_era=0.0)
    a = apply_heading_offsets(w, pd.Series({"zzz": 1.0}))
    b = apply_heading_offsets(w, pd.Series({"abc": 1.0}))
    assert np.isnan(a["hdg_offset"].iloc[0]) and b["hdg_offset"].iloc[0] == 1.0
    tas = w["tas"].iloc[0] * 0.514444
    change = np.hypot(a["u_obs"] - b["u_obs"], a["v_obs"] - b["v_obs"]).iloc[0]
    assert change == pytest.approx(tas * np.radians(1.0), rel=1e-3)
