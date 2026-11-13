"""Register identification of decoded Comm-B replies against ADS-B state vectors, and the receiver clock check.

A reply is accepted as
  BDS 5,0 if its format is valid and its ground speed and true track agree with the state vectors of the same aircraft
          at the same time (|dGS| <= 12 kt, |dtrack| <= 6 deg);
  BDS 6,0 if its format is valid, IAS and Mach are mutually consistent at the barometric altitude (|CAS(Mach, p) - IAS|
          <= 12 kt) and the heading lies within 30 deg of the track.
Replies passing both tests are ambiguous and dropped.

Receiver clock check: some OpenSky receivers stamp replies minutes away from the truth (2024-05-10), which mixes
headings of different flight legs. Each receiver is scored on the cruise replies only it received that look like BDS 5,0;
a receiver with enough such replies whose share of |dGS| <= 2 kt or |dtrack| <= 0.5 deg is too low is distrusted and
every reply it contributed to is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skywind.config import DecodeConfig
from skywind.geo import FT, KT, angle_diff_deg, cas_from_mach, isa_pressure_pa

__all__ = ["attach_state_vectors", "decode_day", "distrusted_receivers", "identify_registers", "receiver_scores"]

_SV_KEEP = ["t", "icao24", "flight_id", "callsign", "typecode", "latitude", "longitude", "altitude", "geoaltitude",
            "groundspeed", "trk_sv"]


def attach_state_vectors(msgs: pd.DataFrame, sv: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
    """Nearest state vector of the same aircraft (within ``max_gap_s``), position dead-reckoned to the message time."""
    right = sv[_SV_KEEP].rename(columns={"t": "t_sv"}).sort_values("t_sv")
    right = right.assign(tk=right["t_sv"], icao24=right["icao24"].astype(str).astype(object))
    left = msgs.sort_values("t").assign(icao24=msgs["icao24"].astype(str).astype(object))
    m = pd.merge_asof(left, right, left_on="t", right_on="tk",
                      by="icao24", direction="nearest", tolerance=max_gap_s)
    m = m.dropna(subset=["latitude", "trk_sv", "groundspeed"]).drop(columns=["tk"]).reset_index(drop=True)
    dist = m["groundspeed"] * KT * (m["t"] - m["t_sv"])
    trk = np.radians(m["trk_sv"])
    m["latitude"] = m["latitude"] + np.degrees(dist * np.cos(trk) / 6371e3)
    m["longitude"] = m["longitude"] + np.degrees(dist * np.sin(trk) / 6371e3 / np.cos(np.radians(m["latitude"])))
    return m


def receiver_scores(m: pd.DataFrame, min_alt_ft: float) -> pd.DataFrame:
    """Per single-receiver serial: number of BDS 5,0-like cruise replies and shares with |dGS| <= 2 kt, |dtrk| <= 0.5 deg."""
    s = m[m["ok50"] & ~m["serials"].str.contains(" ", regex=False) & (m["altitude"] >= min_alt_ft)]
    dtrk = angle_diff_deg(s["trk50"], s["trk_sv"])
    dgs = (s["gs50"] - s["groundspeed"]).to_numpy()
    like = (np.abs(dgs) <= 40) & (np.abs(dtrk) <= 45)
    s = s[like].assign(gs2=np.abs(dgs[like]) <= 2, trk05=np.abs(dtrk[like]) <= 0.5)
    return s.groupby("serials").agg(n=("t", "size"), gs2=("gs2", "mean"), trk05=("trk05", "mean"))


def distrusted_receivers(scores: pd.DataFrame, cfg: DecodeConfig) -> pd.DataFrame:
    """Add ``scored`` and ``trusted`` columns. Receivers with too few replies cannot be scored and are kept."""
    out = scores.copy()
    out["scored"] = out["n"] >= cfg.receiver_min_replies
    good = (out["gs2"] >= cfg.receiver_min_share_gs2) & (out["trk05"] >= cfg.receiver_min_share_trk05)
    out["trusted"] = ~out["scored"] | good
    return out.sort_values("n", ascending=False)


def _drop_receivers(m: pd.DataFrame, bad: set[str]) -> pd.DataFrame:
    if not bad:
        return m
    hit = m["serials"].str.split(" ").map(lambda v: any(x in bad for x in v))
    return m[~hit.astype(bool)]


def identify_registers(m: pd.DataFrame, cfg: DecodeConfig) -> pd.DataFrame:
    """Column ``bds`` in {"50", "60"}; ambiguous and unidentified replies dropped."""
    is50 = (
        m["ok50"]
        & ((m["gs50"] - m["groundspeed"]).abs() <= cfg.tol_gs_kt)
        & (np.abs(angle_diff_deg(m["trk50"], m["trk_sv"])) <= cfg.tol_track_deg)
    )
    p = isa_pressure_pa(m["altitude"].to_numpy() * FT)
    cas_kt = cas_from_mach(m["mach60"].to_numpy(), p) / KT
    is60 = (
        m["ok60"]
        & (np.abs(cas_kt - m["ias60"]) <= cfg.tol_ias_kt)
        & (np.abs(angle_diff_deg(m["hdg60"], m["trk_sv"])) <= cfg.tol_heading_track_deg)
    )
    bds = np.where(is50 & is60, "both", np.where(is50, "50", np.where(is60, "60", "")))
    out = m.assign(bds=bds)
    out = out[out["bds"].isin(["50", "60"])].drop(columns=["ok50", "ok60", "serials"])
    out["pressure_hpa"] = isa_pressure_pa(out["altitude"].to_numpy() * FT) / 100
    return out.reset_index(drop=True)


def decode_day(
    decoded: pd.DataFrame, sv: pd.DataFrame, cfg: DecodeConfig, min_alt_ft: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """From decoded candidates (:func:`skywind.decode.decode_payloads`) and state vectors with ``trk_sv``:
    identified messages, receiver table and counts."""
    m = attach_state_vectors(decoded, sv, cfg.max_sv_gap_s)
    receivers = distrusted_receivers(receiver_scores(m, min_alt_ft), cfg)
    bad = set(receivers.index[~receivers["trusted"]].astype(str))
    kept = _drop_receivers(m, bad)
    out = identify_registers(kept, cfg)
    stats = {
        "candidates": len(decoded),
        "with_state_vector": len(m),
        "receivers_distrusted": len(bad),
        "dropped_by_receiver_check": len(m) - len(kept),
        "bds50": int((out["bds"] == "50").sum()),
        "bds60": int((out["bds"] == "60").sum()),
    }
    return out, receivers, stats
