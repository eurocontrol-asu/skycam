"""Gaussian-process (optimal interpolation) estimator of the ERA5 residual wind, solved locally.

Model. The residual r(x, y, z, t) = (du, dv) has independent zero-mean GP components with variance ``sigma_field**2`` and
a separable correlation (horizontal distance, pressure altitude, time). An observation is a scalar
    y_i = e_i . r(X_i) + b_{a(i)} + eps_i,
with e_i the heading (ALONG) or its right normal (CROSS, optional), b_a a per-aircraft random effect
(along: airspeed bias ~ N(mean_bias, sigma_bias**2); cross: residual heading offset ~ N(0, sigma_cross_bias**2)) and
white noise. The posterior mean and covariance of r at query points are exact conditional Gaussians, computed tile by
tile on the observations within ``cutoff`` length scales of the tile (local GP), so that a whole day scales. Far from any
observation the posterior mean tends to 0 (ERA5) and the variance to the prior variance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from skywind.config import GPConfig
from skywind.field.data import ALONG, CROSS, IntArray, ObsSet, Points, Prediction
from skywind.field.kernels import Kernel, Lengths, correlation, correlation_and_grads, flow_coordinates
from skywind.geo import FloatArray

__all__ = ["GPEstimator", "GPParams", "centred", "covariance", "covariance_and_grads", "cross_covariance"]

_JITTER = 1e-8


@dataclass(frozen=True)
class GPParams:
    length_h_km: float = 40.0
    length_z_m: float = 300.0
    length_t_min: float = 15.0
    sigma_field: float = 2.0
    sigma_bias: float = 1.0
    mean_bias: float = -0.25
    sigma_noise: float = 1.2
    sigma_cross_noise: float = 2.0
    sigma_cross_bias: float = 1.0
    kernel: Kernel = "gaussian"
    use_bias: bool = True
    flow_following: bool = False

    @property
    def lengths(self) -> Lengths:
        return (self.length_h_km, self.length_z_m, self.length_t_min)

    @classmethod
    def from_config(cls, cfg: GPConfig) -> GPParams:
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in cfg.model_dump().items() if k in names})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective(self) -> GPParams:
        """Bias terms switched off when ``use_bias`` is false."""
        return self if self.use_bias else replace(self, sigma_bias=0.0, mean_bias=0.0)


def _coords(X: FloatArray, bg: FloatArray, p: GPParams, t0: float) -> FloatArray:
    return flow_coordinates(X, bg, t0) if p.flow_following else X


def centred(obs: ObsSet, p: GPParams) -> FloatArray:
    """Observations minus the prior mean of the per-aircraft bias (along-heading rows only)."""
    return obs.y - p.effective().mean_bias * (obs.kind == ALONG)


def _bias_and_noise(obs: ObsSet, p: GPParams) -> tuple[FloatArray, FloatArray, FloatArray]:
    q = p.effective()
    sb = np.where(obs.kind == CROSS, q.sigma_cross_bias, q.sigma_bias)
    sn = np.where(obs.kind == CROSS, q.sigma_cross_noise, q.sigma_noise)
    same = (obs.group[:, None] == obs.group[None, :]).astype(float)
    return same, sb, sn


def covariance(obs: ObsSet, p: GPParams, t0: float) -> FloatArray:
    X = _coords(obs.X, obs.bg, p, t0)
    k = correlation(X, X, p.lengths, p.kernel)
    same, sb, sn = _bias_and_noise(obs, p)
    C = p.sigma_field**2 * (obs.e @ obs.e.T) * k + same * sb[:, None] * sb[None, :]
    C[np.diag_indices_from(C)] += sn**2
    return C


def covariance_and_grads(obs: ObsSet, p: GPParams, t0: float) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Covariance and its derivatives with respect to the log of each positive parameter."""
    X = _coords(obs.X, obs.bg, p, t0)
    k, dk = correlation_and_grads(X, X, p.lengths, p.kernel)
    ee = obs.e @ obs.e.T
    sf2 = p.sigma_field**2
    same, sb, sn = _bias_and_noise(obs, p)
    cross = obs.kind == CROSS
    grads = {
        "log_length_h_km": sf2 * ee * dk[0],
        "log_length_z_m": sf2 * ee * dk[1],
        "log_length_t_min": sf2 * ee * dk[2],
        "log_sigma_field": 2 * sf2 * ee * k,
        "log_sigma_bias": 2 * same * np.outer(sb * ~cross, sb * ~cross),
        "log_sigma_cross_bias": 2 * same * np.outer(sb * cross, sb * cross),
        "log_sigma_noise": np.diag(2 * sn**2 * ~cross),
        "log_sigma_cross_noise": np.diag(2 * sn**2 * cross),
    }
    C = sf2 * ee * k + same * sb[:, None] * sb[None, :]
    C[np.diag_indices_from(C)] += sn**2
    return C, grads


def cross_covariance(points: Points, obs: ObsSet, p: GPParams, t0: float) -> tuple[FloatArray, FloatArray]:
    """Cov(du(x*), y_i) and Cov(dv(x*), y_i), each (m, n)."""
    k = correlation(_coords(points.X, points.bg, p, t0), _coords(obs.X, obs.bg, p, t0), p.lengths, p.kernel)
    k *= p.sigma_field**2
    return k * obs.e[None, :, 0], k * obs.e[None, :, 1]


def _excess2(X: FloatArray, lo: FloatArray, hi: FloatArray, scales: FloatArray) -> FloatArray:
    d = np.maximum(np.maximum(lo[None, :] - X, X - hi[None, :]), 0.0) / scales[None, :]
    return np.asarray((d**2).sum(1), dtype=np.float64)


class GPEstimator:
    """Residual-wind posterior from scalar observations; see the module docstring."""

    name = "gp"

    def __init__(self, obs: ObsSet, params: GPParams, cutoff: float = 3.0, tile_minutes: float = 15.0,
                 max_neighbours: int = 3000) -> None:
        self.obs = obs
        self.p = params
        self.cutoff = cutoff
        self.tile_s = tile_minutes * 60.0
        self.max_neighbours = max_neighbours
        self.yc = centred(obs, params)

    @classmethod
    def from_config(cls, obs: ObsSet, cfg: GPConfig) -> GPEstimator:
        return cls(obs, GPParams.from_config(cfg), cfg.cutoff, cfg.tile_minutes, cfg.max_neighbours)

    @property
    def prior_var(self) -> float:
        return self.p.sigma_field**2

    # ---------------------------------------------------------------- neighbourhoods
    def _scales(self) -> FloatArray:
        lh, lz, lt = self.p.lengths
        return np.array([lh, lh, lz, lt * 60.0])

    def _neighbours(self, Xq: FloatArray, t0: float, must: IntArray | None = None) -> IntArray:
        """Observations within ``cutoff`` scaled distance of the bounding box of the query coordinates ``Xq``."""
        Xo = _coords(self.obs.X, self.obs.bg, self.p, t0)
        d2 = _excess2(Xo, Xq.min(0), Xq.max(0), self._scales())
        if must is not None:
            d2[must] = -1.0
        idx = np.flatnonzero(d2 <= self.cutoff**2)
        if len(idx) > self.max_neighbours:
            idx = idx[np.argsort(d2[idx], kind="stable")[: self.max_neighbours]]
        return np.sort(idx).astype(np.int64)

    def _tiles(self, X: FloatArray) -> list[IntArray]:
        """Query indices grouped by time tile and altitude band."""
        tb = np.floor((X[:, 3] - X[:, 3].min()) / self.tile_s).astype(np.int64)
        zb = np.floor(X[:, 2] / (2 * self.cutoff * self.p.length_z_m)).astype(np.int64)
        keys = pd.factorize(pd.Series(tb.astype(str)) + ":" + pd.Series(zb.astype(str)))[0]
        order = np.argsort(keys, kind="stable")
        return [g.astype(np.int64) for g in np.split(order, np.flatnonzero(np.diff(keys[order])) + 1) if len(g)]

    # ---------------------------------------------------------------- prediction
    def predict(self, points: Points) -> Prediction:
        m = len(points)
        out = {k: np.zeros(m) for k in ("du", "dv", "cov_uv")}
        out["var_u"], out["var_v"] = np.full(m, self.prior_var), np.full(m, self.prior_var)
        if m == 0 or len(self.obs) == 0:
            return Prediction(**out, prior_var=self.prior_var)
        for q in self._tiles(points.X):
            self._predict_tile(points, q, out)
        return Prediction(**out, prior_var=self.prior_var)

    def _predict_tile(self, points: Points, q: IntArray, out: dict[str, FloatArray]) -> None:
        sub = Points(points.X[q], points.bg[q])
        t0 = float(sub.X[:, 3].mean())
        sel = self._neighbours(_coords(sub.X, sub.bg, self.p, t0), t0)
        if len(sel) == 0:
            return
        obs = self.obs.subset(sel)
        C = covariance(obs, self.p, t0)
        C[np.diag_indices_from(C)] += _JITTER
        L = cho_factor(C, lower=True)
        alpha = cho_solve(L, self.yc[sel])
        Ku, Kv = cross_covariance(sub, obs, self.p, t0)
        Vu = solve_triangular(L[0], Ku.T, lower=True)
        Vv = solve_triangular(L[0], Kv.T, lower=True)
        out["du"][q], out["dv"][q] = Ku @ alpha, Kv @ alpha
        out["var_u"][q] = self.prior_var - (Vu**2).sum(0)
        out["var_v"][q] = self.prior_var - (Vv**2).sum(0)
        out["cov_uv"][q] = -(Vu * Vv).sum(0)

    # ---------------------------------------------------------------- leave-one-aircraft-out
    def loao(self, vectors: bool = False) -> pd.DataFrame:
        """Prediction of every ALONG observation from the other aircraft only.

        Columns: ``obs_index``, ``pred`` (e . r from the others), ``var_field`` (its posterior variance), ``var_total``
        (+ bias and noise: predictive variance of the observation) and, with ``vectors``, ``du`` / ``dv``.
        """
        along = np.flatnonzero(self.obs.kind == ALONG)
        if len(along) == 0:
            return pd.DataFrame(columns=["obs_index", "pred", "var_field", "var_total"])
        rows: list[pd.DataFrame] = []
        tb = np.floor((self.obs.X[along, 3] - self.obs.X[along, 3].min()) / self.tile_s).astype(np.int64)
        for b in np.unique(tb):
            rows.append(self._loao_tile(along[tb == b], vectors))
        return pd.concat(rows, ignore_index=True).sort_values("obs_index").reset_index(drop=True)

    def _loao_tile(self, targets: IntArray, vectors: bool) -> pd.DataFrame:
        t0 = float(self.obs.X[targets, 3].mean())
        Xq = _coords(self.obs.X[targets], self.obs.bg[targets], self.p, t0)
        sel = self._neighbours(Xq, t0, must=targets)
        obs = self.obs.subset(sel)
        C = covariance(obs, self.p, t0)
        C[np.diag_indices_from(C)] += _JITTER
        P = cho_solve(cho_factor(C, lower=True), np.eye(len(sel)))
        y = self.yc[sel]
        alpha = P @ y
        is_target = np.isin(sel, targets)
        frames = [self._loao_aircraft(obs, sel, P, y, alpha, np.flatnonzero(obs.aircraft == a), is_target, t0, vectors)
                  for a in np.unique(self.obs.aircraft[targets])]
        return pd.concat(frames, ignore_index=True)

    def _loao_aircraft(self, obs: ObsSet, sel: IntArray, P: FloatArray, y: FloatArray, alpha: FloatArray,
                       G: IntArray, is_target: Any, t0: float, vectors: bool) -> pd.DataFrame:
        Pgg_inv = np.linalg.inv(P[np.ix_(G, G)])
        mean = y[G] - Pgg_inv @ alpha[G]
        var_total = np.diag(Pgg_inv).copy()
        sb, sn = self.p.effective().sigma_bias, self.p.sigma_noise
        keep = is_target[G] & (obs.kind[G] == ALONG)
        out = pd.DataFrame({"obs_index": sel[G][keep], "pred": mean[keep], "var_total": var_total[keep],
                            "var_field": np.maximum(var_total[keep] - sb**2 - sn**2, 0.0)})
        if vectors:
            du, dv = self._loo_vectors(obs, P, alpha, G, t0)
            out["du"], out["dv"] = du[keep], dv[keep]
        return out

    def _loo_vectors(self, obs: ObsSet, P: FloatArray, alpha: FloatArray, G: IntArray, t0: float
                     ) -> tuple[FloatArray, FloatArray]:
        """Vector field at the points of group G from the other observations only:
        C_RR^-1 y_R = (P y)_R - P_RG P_GG^-1 (P y)_G."""
        R = np.setdiff1d(np.arange(len(alpha)), G)
        a_r = alpha[R] - P[np.ix_(R, G)] @ np.linalg.solve(P[np.ix_(G, G)], alpha[G])
        Ku, Kv = cross_covariance(Points(obs.X[G], obs.bg[G]), obs.subset(R), self.p, t0)
        return Ku @ a_r, Kv @ a_r
