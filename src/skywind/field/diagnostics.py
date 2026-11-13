"""Consistency diagnostics of the background (ERA5) and observation errors.

Hollingsworth-Lonnberg: the covariance of the along-heading innovations of pairs of DIFFERENT aircraft, binned by
separation, is the background-error covariance projected on the two headings (observation errors and per-aircraft
biases are uncorrelated between aircraft). For a pair, E[y_i y_j] = c(d) e_i . e_j, so each bin is estimated by least
squares c = sum(y_i y_j e_i.e_j) / sum((e_i.e_j)^2). Extrapolated to zero separation, c(0) is the background
variance and var(y) - c(0) the observation (bias + noise) variance; the decay gives the correlation lengths directly.

Desroziers et al. (2005): with d_ob = y - H x_b, d_oa = y - H x_a, d_ab = H x_a - H x_b,
E[d_oa d_ob] = R and E[d_ab d_ob] = H B H^T; the ratios to the assumed values should be close to 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from skywind.field.data import ALONG, ObsSet, Points
from skywind.field.gp import GPEstimator, centred
from skywind.geo import FloatArray

__all__ = ["DIMENSIONS", "desroziers", "fit_correlation_length", "hollingsworth_lonnberg"]

# separation dimension -> (bin edges, limits on the two other separations: horizontal km, vertical m, time s)
DIMENSIONS: dict[str, tuple[FloatArray, dict[str, float]]] = {
    "horizontal_km": (np.arange(0.0, 205.0, 10.0), {"vertical_m": 150.0, "time_s": 300.0}),
    "vertical_m": (np.arange(0.0, 2050.0, 100.0), {"horizontal_km": 20.0, "time_s": 300.0}),
    "time_s": (np.arange(0.0, 5460.0, 300.0), {"horizontal_km": 20.0, "vertical_m": 150.0}),
}


def _separations(X: FloatArray, i: slice) -> dict[str, FloatArray]:
    A = X[i]
    return {
        "horizontal_km": np.hypot(A[:, None, 0] - X[None, :, 0], A[:, None, 1] - X[None, :, 1]),
        "vertical_m": np.abs(A[:, None, 2] - X[None, :, 2]),
        "time_s": np.abs(A[:, None, 3] - X[None, :, 3]),
    }


def _accumulate(
    obs: ObsSet, y: FloatArray, dim: str, edges: FloatArray, limits: dict[str, float], acc: FloatArray
) -> None:
    for s in range(0, len(y), 500):
        i = slice(s, min(s + 500, len(y)))
        sep = _separations(obs.X, i)
        ok = (obs.aircraft[i][:, None] != obs.aircraft[None, :]) & (np.arange(s, i.stop)[:, None] < np.arange(len(y)))
        for other, lim in limits.items():
            ok &= sep[other] <= lim
        ok &= sep[dim] < edges[-1]
        ee = (obs.e[i] @ obs.e.T)[ok]
        yy = (y[i][:, None] * y[None, :])[ok]
        b = np.digitize(sep[dim][ok], edges) - 1
        np.add.at(acc[0], b, yy * ee)
        np.add.at(acc[1], b, ee**2)
        np.add.at(acc[2], b, 1.0)


def hollingsworth_lonnberg(obs_sets: Sequence[ObsSet], mean_bias: float = 0.0) -> pd.DataFrame:
    """Binned cross-aircraft innovation covariance for each separation dimension, pooled over ``obs_sets``."""
    rows = []
    for dim, (edges, limits) in DIMENSIONS.items():
        acc = np.zeros((3, len(edges) - 1))
        for obs in obs_sets:
            o = obs.subset(obs.kind == ALONG)
            _accumulate(o, o.y - mean_bias, dim, edges, limits, acc)
        centre = 0.5 * (edges[:-1] + edges[1:])
        cov = np.where(acc[1] > 0, acc[0] / np.maximum(acc[1], 1e-12), np.nan)
        rows.append(pd.DataFrame({"dimension": dim, "separation": centre, "covariance": cov, "pairs": acc[2]}))
    return pd.concat(rows, ignore_index=True)


def fit_correlation_length(table: pd.DataFrame, dimension: str, min_pairs: int = 200) -> dict[str, float]:
    """Gaussian fit c(d) = c0 exp(-d^2 / (2 L^2)) to one dimension of :func:`hollingsworth_lonnberg`."""
    t = table[(table["dimension"] == dimension) & (table["pairs"] >= min_pairs)].dropna()
    if len(t) < 3:
        return {"c0": float("nan"), "length": float("nan")}
    x, c = t["separation"].to_numpy(float), t["covariance"].to_numpy(float)
    p0 = (max(float(c[0]), 0.1), float(x[len(x) // 3]) + 1.0)
    try:
        (c0, length), _ = curve_fit(
            lambda d, a, s: a * np.exp(-(d**2) / (2 * s**2)),
            x,
            c,
            p0=p0,
            sigma=1 / np.sqrt(t["pairs"].to_numpy(float)),
            maxfev=5000,
        )
    except RuntimeError:
        return {"c0": float("nan"), "length": float("nan")}
    return {"c0": float(c0), "length": float(abs(length))}


def desroziers(est: GPEstimator) -> dict[str, float]:
    """Desroziers ratios for the along-heading observations of an estimator (analysis with all observations)."""
    obs = est.obs.subset(est.obs.kind == ALONG)
    pred = est.predict(Points(obs.X, obs.bg))
    hxa = pred.du * obs.e[:, 0] + pred.dv * obs.e[:, 1]
    d_ob = centred(obs, est.p)
    d_oa, d_ab = d_ob - hxa, hxa
    p = est.p.effective()
    r_assumed = p.sigma_bias**2 + p.sigma_noise**2
    return {
        "n": float(len(d_ob)),
        "R_estimated": float(np.mean(d_oa * d_ob)),
        "R_assumed": r_assumed,
        "R_ratio": float(np.mean(d_oa * d_ob) / r_assumed),
        "HBH_estimated": float(np.mean(d_ab * d_ob)),
        "HBH_assumed": p.sigma_field**2,
        "HBH_ratio": float(np.mean(d_ab * d_ob) / p.sigma_field**2),
    }
