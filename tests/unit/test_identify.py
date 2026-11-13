from __future__ import annotations

import pandas as pd

from skywind.config import DecodeConfig
from skywind.identify import distrusted_receivers, identify_registers


def _row(**kw: object) -> dict[str, object]:
    base = {"t": 0.0, "ok50": False, "ok60": False, "gs50": 0.0, "trk50": 0.0, "groundspeed": 450.0, "trk_sv": 90.0,
            "mach60": 0.78, "ias60": 0.0, "hdg60": 0.0, "altitude": 36000.0, "serials": "1"}
    return {**base, **kw}


def test_registers_identified_against_state_vectors() -> None:
    m = pd.DataFrame([
        _row(ok50=True, gs50=452.0, trk50=91.0),                   # BDS 5,0 consistent
        _row(ok50=True, gs50=300.0, trk50=91.0),                   # ground speed inconsistent
        _row(ok60=True, ias60=262.0, hdg60=95.0),                  # BDS 6,0 (CAS(M .78, FL360) ~ 262 kt)
        _row(ok60=True, ias60=320.0, hdg60=95.0),                  # IAS inconsistent with Mach
    ])
    out = identify_registers(m, DecodeConfig())
    assert list(out["bds"]) == ["50", "60"]


def test_receivers_scored_only_with_enough_replies() -> None:
    scores = pd.DataFrame({"n": [500, 500, 10], "gs2": [0.99, 0.5, 0.1], "trk05": [0.95, 0.9, 0.1]},
                          index=["good", "bad", "few"])
    out = distrusted_receivers(scores, DecodeConfig())
    assert dict(out["trusted"]) == {"good": True, "bad": False, "few": True}
