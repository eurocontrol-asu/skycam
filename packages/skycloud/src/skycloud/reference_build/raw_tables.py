"""Step 4, raw reference tables (prototype ``b3_tables.py``): per elevation node and bin, the
median and robust sd (half the 16th-84th percentile range) of the bin-mean colour index N and
log brightness L over the clear frames and over the overcast frames, the median relative
texture, the frame counts and the mean sun elevation. A statistic needs ``table_min_frames``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from skycloud.reference_build.binning import BinMeans
from skycloud.reference_build.labels import CLEAR, OVERCAST, node_of
from skycloud.reference_build.params import BinParams

__all__ = ["RAW_FIELDS", "column_percentiles", "raw_tables", "robust_median_sd"]

RAW_FIELDS = (
    "Nc", "Nc_sd", "No", "No_sd", "Lc", "Lc_sd", "Lo", "Lo_sd", "Tc", "To", "elev_c", "elev_o",
)  # fmt: skip


def column_percentiles(m: Any, qs: list[float]) -> Any:
    """Per column percentiles (NaN ignored) AT THE INPUT'S PRECISION: NumPy 2.5.3, which built
    the c16 tables, returned float32 percentiles of float32 data; 2.5.5 returns float64, which
    changes the last bit of a spread computed from them."""
    return np.nanpercentile(m, qs, axis=0).astype(m.dtype)


def robust_median_sd(m: Any) -> tuple[Any, Any]:
    """Per column: the median and half the 16th-84th percentile range (NaN ignored)."""
    q = column_percentiles(m, [16, 50, 84])
    return q[1], (q[2] - q[0]) / 2


def raw_tables(bm: BinMeans, lab: Any, p: BinParams, min_frames: int) -> dict[str, Any]:
    """The raw tables, each (nodes, bins); ``lab``: the frames' labels (1 clear, 2 overcast)."""
    el = bm.elev
    cls = node_of(el, p.elevation_nodes)
    ne, nb = len(p.elevation_nodes), p.n_gamma * p.n_zenith
    out: dict[str, Any] = {k: np.full((ne, nb), np.nan, np.float32) for k in RAW_FIELDS}
    out["n_c"] = np.zeros((ne, nb), np.int32)
    out["n_o"] = np.zeros((ne, nb), np.int32)
    with np.errstate(all="ignore"):
        for c in range(ne):
            for tag, label in (("c", CLEAR), ("o", OVERCAST)):
                idx = np.flatnonzero((cls == c) & (lab == label))
                if len(idx) < min_frames:
                    continue
                n = np.zeros(nb, int)
                for cue, field in (("N", bm.MN), ("L", bm.ML)):
                    a = field[idx].astype(np.float32)
                    n = np.isfinite(a).sum(0)
                    med, sd = robust_median_sd(a)
                    out[f"{cue}{tag}"][c] = np.where(n >= min_frames, med, np.nan)
                    out[f"{cue}{tag}_sd"][c] = np.where(n >= min_frames, sd, np.nan)
                a = bm.MT[idx].astype(np.float32)
                keep = np.isfinite(a).sum(0) >= min_frames
                out[f"T{tag}"][c] = np.where(keep, np.nanmedian(a, 0), np.nan)
                out[f"n_{tag}"][c] = n  # the count of the last cue read (L), as b3_tables.py
                out[f"elev_{tag}"][c] = np.mean(el[idx])
    return out
