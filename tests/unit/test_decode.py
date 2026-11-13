from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skywind.decode import bits, decode_bds50, decode_bds60, decode_payloads, payloads

# Reference values decoded with pyModeS 2 (commb.roll50 / trk50 / gs50 / rtrk50 / tas50, hdg60 / ias60 / mach60 / vr60*)
BDS50 = "A000139381951536E024D4CCF6B5"
BDS60 = "A00004128F39F91A7E27C46ADC21"


def test_bits_extracts_one_based_fields() -> None:
    mb = np.array([1 << 55 | 0b101], dtype=np.uint64)
    assert bits(mb, 1, 1)[0] == 1
    assert bits(mb, 54, 3)[0] == 0b101


def test_bds50_known_message() -> None:
    d = decode_bds50(payloads([BDS50]))
    assert d["roll"][0] == pytest.approx(2.109375)
    assert d["trk50"][0] == pytest.approx(114.2578125)
    assert d["gs50"][0] == 438
    assert d["trk_rate50"][0] == pytest.approx(0.125)
    assert d["tas50"][0] == 424
    assert d["ok50"][0]


def test_bds60_known_message() -> None:
    d = decode_bds60(payloads([BDS60]))
    assert d["hdg60"][0] == pytest.approx(42.71484375)
    assert d["ias60"][0] == 252
    assert d["mach60"][0] == pytest.approx(0.42)
    assert d["vrb60"][0] == -1920
    assert d["vri60"][0] == -1920
    assert d["ok60"][0]


def test_empty_payload_is_invalid() -> None:
    mb = np.zeros(1, dtype=np.uint64)
    assert not decode_bds50(mb)["ok50"][0]
    assert not decode_bds60(mb)["ok60"][0]


def test_decode_payloads_keeps_valid_rows_and_drops_short_messages() -> None:
    raw = pd.DataFrame({"mintime": [1.0, 2.0, 3.0], "rawmsg": [BDS50, BDS60, "A0001393"],
                        "icao24": ["abc", "abc", "abc"], "serials": ["1", "1 2", "3"]})
    out = decode_payloads(raw)
    assert list(out["t"]) == [1.0, 2.0]
    assert bool(out["ok50"].iloc[0]) and bool(out["ok60"].iloc[1])
