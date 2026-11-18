from __future__ import annotations

import pytest
from pydantic import ValidationError

from skycloud.params import CloudMapParams


def test_defaults_are_version_c16_and_frozen() -> None:
    p = CloudMapParams()
    assert p.unknown_conf == 0.30 and p.known_thr == 0.45 and p.contract_band == 0.037
    assert all(f.description for f in CloudMapParams.model_fields.values())
    with pytest.raises(ValidationError):
        p.unknown_conf = 0.5  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CloudMapParams(not_a_parameter=1)  # type: ignore[call-arg]
