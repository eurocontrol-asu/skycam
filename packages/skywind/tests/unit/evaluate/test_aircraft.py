from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skywind.evaluate.aircraft import loao_frame, score_predictions
from skywind.field.gp import GPEstimator
from tests.unit.field.test_gp import PARAMS, tracks


def test_scores_skill_and_calibration() -> None:
    pred = pd.DataFrame(
        {
            "day": ["a"] * 4,
            "icao24": ["x", "x", "y", "y"],
            "y": [1.0, -1.0, 2.0, -2.0],
            "pred": [0.5, -0.5, 1.0, -1.0],
            "var_field": 1.0,
            "var_total": 1.0,
            "z": [0.5, -0.5, 1.0, -1.0],
        }
    )
    s = score_predictions(pred).set_index("day")
    assert s.loc["pooled", "skill"] == pytest.approx(0.75)
    assert s.loc["a", "cover68"] == 1.0


def test_gp_loao_has_skill_on_synthetic_traffic() -> None:
    obs = tracks([0.0, 90.0, 45.0])
    s = score_predictions(loao_frame(GPEstimator(obs, PARAMS, tile_minutes=60), obs, 0.0)).set_index("day")
    assert s.loc["pooled", "skill"] > 0.5
    assert 0.7 < s.loc["pooled", "z_sd"] < 1.3
    assert np.isfinite(s.loc["pooled", "rmse_field"])
