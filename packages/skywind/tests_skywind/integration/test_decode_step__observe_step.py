from __future__ import annotations

import pandas as pd
import pytest

from skywind.config import Settings
from skywind.pipeline import decode_step, observe_step
from tests_skywind.integration.conftest import DAY

pytestmark = pytest.mark.integration


@pytest.mark.requires_store
def test_decode_and_observe_reproduce_the_prototype(tmp_settings: Settings) -> None:
    dec = decode_step(tmp_settings, DAY)
    assert dec["bds50"] == 193977 and dec["bds60"] == 378051  # prototype m2_decode.log
    obs = observe_step(tmp_settings, DAY)
    assert obs["observations"] == 6946 and obs["aircraft"] == 220  # prototype wind_vs_era5.log
    assert obs["along_bias"] == pytest.approx(-0.28, abs=0.01)
    assert obs["right_bias"] == pytest.approx(-5.15, abs=0.01)
    w = pd.read_parquet(tmp_settings.paths.observations(DAY))
    assert w["decl"].between(0.0, 4.0).all()
