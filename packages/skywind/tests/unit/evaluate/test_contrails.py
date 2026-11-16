from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skywind.config import Site
from skywind.evaluate.contrails import ContrailTruth, age_table, cross_track_errors, drift_table, flight_fits
from skywind.geo import offset_latlon


def _truth_and_advected(offset_m: float, drift: float) -> tuple[ContrailTruth, pd.DataFrame]:
    ages = np.arange(1.0, 30.0, 0.5)
    t0 = pd.Timestamp("2024-03-18T12:00", tz="UTC")
    times = [t0 + pd.Timedelta(minutes=float(a)) for a in ages]
    lat, lon = np.full(len(ages), 48.6), np.full(len(ages), 2.35)
    c_lat, c_lon = offset_latlon(
        lat, lon, offset_m + drift * ages, 0.0
    )  # contrail to the east = right of a northbound track
    centres = pd.DataFrame(
        {
            "flight": "g1",
            "flight_id": "f1",
            "waypoint": 0,
            "time": times,
            "age_min": ages,
            "centre_lat": c_lat,
            "centre_lon": c_lon,
            "normal_e": 1.0,
            "normal_n": 0.0,
        }
    )
    adv = pd.DataFrame({"flight_id": "f1", "waypoint": 0, "time": times, "latitude": lat, "longitude": lon})
    return ContrailTruth("20240318", pd.DataFrame(), centres), adv


def test_cross_track_error_offset_and_drift() -> None:
    truth, adv = _truth_and_advected(300.0, 20.0)
    e = cross_track_errors(truth, adv, Site())
    f = flight_fits(e)
    assert f["offset_m"].iloc[0] == pytest.approx(300.0, abs=2)
    assert f["drift_m_per_min"].iloc[0] == pytest.approx(20.0, abs=0.1)


def test_paired_tables() -> None:
    truth, adv = _truth_and_advected(300.0, 20.0)
    e1 = cross_track_errors(truth, adv, Site())
    e2 = e1.assign(error_m=e1["error_m"] - 15.0 * e1["age_min"])
    fits = {"era5": flight_fits(e1), "gp": flight_fits(e2)}
    _, summary = drift_table(fits)
    assert summary.set_index("day").loc["pooled", "improved_gp"] == 1.0
    ages = age_table({"era5": e1, "gp": e2}, fits)
    pooled = ages[(ages["day"] == "pooled") & (ages["method"] == "gp")]
    assert (pooled["drift_only_med"] < 200).all()


def test_paired_tests_one_sided() -> None:
    from skywind.evaluate.contrails import paired_tests

    paired = pd.DataFrame(
        {
            "day": "d",
            "flight": [str(i) for i in range(12)],
            "drift_era5": np.linspace(20, 80, 12),
            "drift_gp": np.linspace(5, 30, 12),
        }
    )
    t = paired_tests(paired).set_index("day")
    assert t.loc["pooled", "n_improved"] == 12
    assert t.loc["pooled", "wilcoxon_p"] < 0.01 and t.loc["pooled", "sign_test_p"] < 0.01
