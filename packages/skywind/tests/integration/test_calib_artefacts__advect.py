from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skywind.advect import advect, met_dataset
from skywind.config import Settings
from skywind.era5 import load_era5
from skywind.evaluate.contrails import cross_track_errors, flight_fits
from skywind.evaluate.gvccs_calib import CalibArtefacts
from tests.integration.conftest import DAY

pytestmark = pytest.mark.integration


def test_era5_advection_and_drift_match_the_prototype(real_settings: Settings) -> None:
    root = real_settings.paths.prototype_dir / "calib"
    if not root.is_dir():
        pytest.skip("prototype artefacts not reachable")
    truth = CalibArtefacts(real_settings).load(DAY)
    adv = advect(
        truth.emissions, met_dataset(load_era5(real_settings.paths.era5(DAY))), truth.last_time, max_age_min=45
    )
    ref = pd.read_parquet(root / "flights" / f"advected_{DAY}.parquet")
    ref["flight_id"] = ref["flight_id"].astype(str)
    ref["time"] = ref["time"].dt.as_unit("ns")
    m = adv.merge(ref, on=["flight_id", "waypoint", "time"], suffixes=("", "_ref"))
    assert len(m) > 100_000
    assert np.abs(m["latitude"] - m["latitude_ref"]).max() < 1e-6
    fits = flight_fits(cross_track_errors(truth, adv, real_settings.site)).set_index("flight")
    anchor = pd.read_csv(real_settings.paths.prototype_dir / "flights" / "anchor_flights.csv", dtype=str)
    rho = anchor.set_index("flight")["rho_gt"].astype(float)
    common = fits.index.intersection(rho.index)
    assert len(common) >= 3
    np.testing.assert_allclose(fits.loc[common, "drift_m_per_min"], rho[common], atol=5.0)
