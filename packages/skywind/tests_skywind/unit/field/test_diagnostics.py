from __future__ import annotations

import numpy as np

from skywind.field.diagnostics import fit_correlation_length, hollingsworth_lonnberg
from tests_skywind.unit.field.test_gp import tracks


def test_hl_covariance_ignores_same_aircraft_pairs_and_decays() -> None:
    obs = tracks([0.0, 90.0, 45.0, 135.0], n_tracks=10)
    table = hollingsworth_lonnberg([obs])
    h = table[(table["dimension"] == "horizontal_km") & (table["pairs"] > 50)]
    assert h["covariance"].iloc[0] > h["covariance"].iloc[-1]
    fit = fit_correlation_length(table, "horizontal_km", min_pairs=50)
    assert 10 < fit["length"] < 200


def test_fit_returns_nan_without_data() -> None:
    import pandas as pd

    empty = pd.DataFrame(columns=["dimension", "separation", "covariance", "pairs"])
    assert np.isnan(fit_correlation_length(empty, "vertical_m")["length"])
