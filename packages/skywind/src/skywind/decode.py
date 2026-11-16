"""Vectorised decoding of Comm-B (DF20/21) payloads as BDS 5,0 (track and turn report) and BDS 6,0 (heading and speed).

Comm-B replies do not carry their register number: these functions decode every payload under both interpretations and
return the format-level validity (status bits, reserved values, physical ranges). The register is then identified
against the ADS-B state vectors in :mod:`skywind.identify`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from skywind.geo import FloatArray

__all__ = ["bits", "decode_bds50", "decode_bds60", "decode_payloads", "payloads"]

type IntArray = npt.NDArray[np.int64]
type U64Array = npt.NDArray[np.uint64]
type BoolArray = npt.NDArray[np.bool_]


def payloads(rawmsg: Sequence[str]) -> U64Array:
    """56-bit MB field of 112-bit (28 hex digit) messages, as uint64."""
    return np.array([int(s[8:22], 16) for s in rawmsg], dtype=np.uint64)


def bits(mb: U64Array, start: int, n: int) -> IntArray:
    """Field of ``n`` bits starting at 1-based bit ``start`` of the 56-bit MB payload."""
    shift = np.uint64(56 - (start + n - 1))
    return ((mb >> shift) & np.uint64((1 << n) - 1)).astype(np.int64)


def _signed(sign: IntArray, value: IntArray, nbits: int) -> IntArray:
    return value - (1 << nbits) * sign


def decode_bds50(mb: U64Array) -> dict[str, FloatArray | BoolArray]:
    """Roll (deg), true track (deg), ground speed (kt), track rate (deg/s), true airspeed (kt) and format validity."""
    s_roll, roll_raw = bits(mb, 1, 1), _signed(bits(mb, 2, 1), bits(mb, 3, 9), 9)
    s_trk, trk_raw = bits(mb, 12, 1), _signed(bits(mb, 13, 1), bits(mb, 14, 10), 10)
    s_gs, v_gs = bits(mb, 24, 1), bits(mb, 25, 10)
    s_rate, rate_raw = bits(mb, 35, 1), _signed(bits(mb, 36, 1), bits(mb, 37, 9), 9)
    s_tas, v_tas = bits(mb, 46, 1), bits(mb, 47, 10)
    roll = roll_raw * 45.0 / 256.0
    track = (trk_raw * 90.0 / 512.0) % 360.0
    gs, tas = v_gs * 2.0, v_tas * 2.0
    status = (s_roll == 1) & (s_trk == 1) & (s_gs == 1) & (s_tas == 1) & ((s_rate == 1) | (rate_raw == 0))
    physical = (
        (np.abs(roll) <= 40) & (gs >= 150) & (gs <= 700) & (tas >= 150) & (tas <= 600) & (np.abs(tas - gs) <= 220)
    )
    return {
        "roll": roll,
        "trk50": track,
        "gs50": gs,
        "trk_rate50": rate_raw * 8.0 / 256.0,
        "tas50": tas,
        "ok50": (mb != 0) & status & physical,
    }


def decode_bds60(mb: U64Array) -> dict[str, FloatArray | BoolArray]:
    """Magnetic heading (deg), IAS (kt), Mach, barometric and inertial vertical rates (ft/min) and format validity."""
    s_hdg, hdg_raw = bits(mb, 1, 1), _signed(bits(mb, 2, 1), bits(mb, 3, 10), 10)
    s_ias, v_ias = bits(mb, 13, 1), bits(mb, 14, 10)
    s_mach, v_mach = bits(mb, 24, 1), bits(mb, 25, 10)
    s_vb, sg_vb, v_vb = bits(mb, 35, 1), bits(mb, 36, 1), bits(mb, 37, 9)
    s_vi, sg_vi, v_vi = bits(mb, 46, 1), bits(mb, 47, 1), bits(mb, 48, 9)
    heading = (hdg_raw * 90.0 / 512.0) % 360.0
    ias, mach = v_ias * 1.0, v_mach * 2.048 / 512.0
    vr_baro, vr_ins = _signed(sg_vb, v_vb, 9) * 32.0, _signed(sg_vi, v_vi, 9) * 32.0
    status = (
        (s_hdg == 1)
        & (s_ias == 1)
        & (s_mach == 1)
        & ((s_vb == 1) | ((sg_vb == 0) & (v_vb == 0)))
        & ((s_vi == 1) | ((sg_vi == 0) & (v_vi == 0)))
    )
    physical = (
        (ias >= 120)
        & (ias <= 450)
        & (mach >= 0.3)
        & (mach <= 0.95)
        & (np.abs(vr_baro) <= 6000)
        & (np.abs(vr_ins) <= 6000)
    )
    return {
        "hdg60": heading,
        "ias60": ias,
        "mach60": mach,
        "vrb60": vr_baro,
        "vri60": vr_ins,
        "ok60": (mb != 0) & status & physical,
    }


def decode_payloads(raw: pd.DataFrame) -> pd.DataFrame:
    """Decode a Comm-B table (columns ``mintime``, ``rawmsg``, ``icao24``, ``serials``) under both registers.

    Returns the rows valid for at least one register, with column ``t`` (epoch seconds) and the decoded fields.
    """
    raw = raw[raw["rawmsg"].str.len() == 28].drop_duplicates(["icao24", "mintime", "rawmsg"])
    mb = payloads(raw["rawmsg"].astype(str).tolist())
    fields = {**decode_bds50(mb), **decode_bds60(mb)}
    keep = np.asarray(fields["ok50"]) | np.asarray(fields["ok60"])
    out = pd.DataFrame(
        {
            "t": raw["mintime"].astype(float).to_numpy()[keep],
            "icao24": raw["icao24"].astype(str).to_numpy()[keep],
            "serials": raw["serials"].astype(str).to_numpy()[keep],
        }
    )
    for name, values in fields.items():
        out[name] = np.asarray(values)[keep]
    return out.sort_values("t").reset_index(drop=True)
