from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skywind.statevectors import add_position_track, sanitise


def _flight(n: int = 40, jump_every: int = 0) -> pd.DataFrame:
    t = np.arange(n) * 5.0
    lat = 48.0 + t * 230 / 111190.0  # 230 m/s due north
    if jump_every:
        lat[::jump_every] += 0.05
    return pd.DataFrame({"flight_id": "f1", "t": t, "latitude": lat, "longitude": 2.0, "altitude": 36000.0,
                         "geoaltitude": 37000.0, "groundspeed": 230 / 0.514444})


def test_position_track_due_north() -> None:
    sv = add_position_track(_flight())
    assert np.nanmedian(sv["trk_sv"]) % 360 == pytest.approx(0.0, abs=1e-6)
    assert np.nanmedian(sv["gs_sv_pos"]) * 0.514444 == pytest.approx(230.0, abs=0.5)


def test_sanitise_leaves_clean_data_untouched() -> None:
    df, rep = sanitise(_flight())
    assert rep["repositioned"] == 0
    np.testing.assert_allclose(df["latitude"], _flight()["latitude"])


def test_sanitise_repairs_jumping_positions() -> None:
    df, rep = sanitise(_flight(80, jump_every=7))
    assert rep["bad_step_share"] > 0.1 and rep["repositioned"] > 0
    assert rep["bad_step_share_after"] < rep["bad_step_share"]
