from __future__ import annotations

import pytest
from pydantic import ValidationError

from skycloud.reference_build.params import BuildParams


def test_defaults_are_the_c16_build_and_documented() -> None:
    p = BuildParams()
    assert p.labels.min_gap == 0.03
    assert (p.labels.clear_q90_max, p.labels.clear_q50_max, p.labels.overcast_q10_min) == (
        0.35, 0.15, 0.65,
    )  # fmt: skip
    assert p.labels.episode_below == 6.0 and p.smooth.table_min_frames == 8
    assert len(p.bins.elevation_nodes) == 27 and p.bins.n_gamma * p.bins.n_zenith == 5460
    assert p.duplicates == "error"
    for model in (p.bins, p.labels, p.smooth, p.thresholds, p.lowcount):
        assert all(f.description for f in type(model).model_fields.values())
    with pytest.raises(ValidationError):
        BuildParams.model_validate({"labels": {"not_a_parameter": 1}})


def test_params_round_trip_through_json() -> None:
    p = BuildParams.model_validate({"labels": {"min_gap": 0.05}})
    assert BuildParams.model_validate_json(p.model_dump_json()) == p
    assert p.labels.min_gap == 0.05 and p.labels.q_clear == 85.0
