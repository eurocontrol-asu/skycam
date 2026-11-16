from __future__ import annotations

import pytest

from skywind.advect import advect, met_dataset
from skywind.config import Settings
from skywind.era5 import load_era5
from skywind.evaluate.contrails import cross_track_errors, flight_fits
from skywind.evaluate.gvccs_raw import GvccsRaw
from tests.integration.conftest import DAY

pytestmark = pytest.mark.integration


def test_raw_truth_drift_matches_the_prototype_artefacts(real_settings: Settings) -> None:
    if not (real_settings.paths.gvccs_dir / "train" / "annotations.json").is_file():
        pytest.skip("GVCCS not reachable")
    truth = GvccsRaw(real_settings).load(DAY)
    adv = advect(
        truth.emissions, met_dataset(load_era5(real_settings.paths.era5(DAY))), truth.last_time, max_age_min=45
    )
    fits = flight_fits(cross_track_errors(truth, adv, real_settings.site)).set_index("flight")
    # prototype calibration artefacts (anchor_flights.csv rho_gt) for two flights of the day: -20.9 and 113.8 m/min
    assert fits.loc["33e32a8b", "drift_m_per_min"] == pytest.approx(-20.9, abs=3.0)
    assert fits.loc["33e3def3", "drift_m_per_min"] == pytest.approx(113.8, abs=6.0)
