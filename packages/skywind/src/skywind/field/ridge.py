"""Prototype estimator (baseline): kernel-weighted ridge least squares of the residual vector at each query point.

At a query point, r = (du, dv) minimises  sum_i w_i (y_i - e_i . r)^2 / s_i^2 + |r|^2 / sigma_prior^2, with Gaussian
weights w_i in horizontal distance, altitude and time. ``var_*`` are the inverse normal-matrix terms (a heuristic
uncertainty, not a calibrated posterior).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from skywind.config import RidgeConfig
from skywind.field.data import ALONG, CROSS, ObsSet, Points, Prediction
from skywind.field.kernels import correlation, factor_correlations
from skywind.geo import FloatArray

__all__ = ["RidgeEstimator"]

_CHUNK = 2000


class RidgeEstimator:
    name = "ridge"

    def __init__(self, obs: ObsSet, cfg: RidgeConfig) -> None:
        self.cfg = cfg
        self.obs = obs if cfg.use_cross else obs.subset(obs.kind == ALONG)
        s = np.where(self.obs.kind == CROSS, cfg.sigma_cross, cfg.sigma_obs)
        self.inv_var = 1.0 / s**2

    @property
    def prior_var(self) -> float:
        return self.cfg.sigma_prior**2

    def _weights(self, X: FloatArray) -> FloatArray:
        lengths = (self.cfg.length_h_km, self.cfg.length_z_m, self.cfg.length_t_min)
        return correlation(X, self.obs.X, lengths, "gaussian")

    def _solve(self, W: FloatArray) -> tuple[FloatArray, ...]:
        ex, ey, a = self.obs.e[:, 0], self.obs.e[:, 1], self.obs.y * self.inv_var
        iv = self.inv_var
        A11 = W @ (ex * ex * iv) + 1 / self.prior_var
        A12 = W @ (ex * ey * iv)
        A22 = W @ (ey * ey * iv) + 1 / self.prior_var
        b1, b2 = W @ (a * ex), W @ (a * ey)
        det = A11 * A22 - A12**2
        return (A22 * b1 - A12 * b2) / det, (A11 * b2 - A12 * b1) / det, A22 / det, A11 / det, -A12 / det

    def predict_grid(
        self, times: FloatArray, z_m: FloatArray, xy_km: FloatArray, variance: npt.NDArray[np.bool_] | None = None
    ) -> dict[str, FloatArray]:
        """Same solution as :meth:`predict` on a grid (times x altitudes x horizontal nodes): the Gaussian weights are
        separable, so every weighted sum is a matrix product per time. Arrays of shape (T, L, H)."""
        lengths = (self.cfg.length_h_km, self.cfg.length_z_m, self.cfg.length_t_min)
        ex, ey, a, iv = self.obs.e[:, 0], self.obs.e[:, 1], self.obs.y * self.inv_var, self.inv_var
        vectors = (ex * ex * iv, ex * ey * iv, ey * ey * iv, a * ex, a * ey)
        shape = (len(times), len(z_m), len(xy_km))
        sums = np.zeros((5, *shape))
        for i, t in enumerate(np.asarray(times, float)):
            kh, kz, kt = factor_correlations(
                np.asarray(xy_km, float), np.asarray(z_m, float), float(t), self.obs.X, lengths, "gaussian"
            )
            use = kt > 1e-12
            for j, v in enumerate(vectors):
                sums[j, i] = (kz[:, use] * (kt[use] * v[use])) @ kh[:, use].T
        A11, A12, A22, b1, b2 = sums
        A11 = A11 + 1 / self.prior_var
        A22 = A22 + 1 / self.prior_var
        det = A11 * A22 - A12**2
        return {
            "du": (A22 * b1 - A12 * b2) / det,
            "dv": (A11 * b2 - A12 * b1) / det,
            "var_u": A22 / det,
            "var_v": A11 / det,
            "cov_uv": -A12 / det,
        }

    def predict(self, points: Points, variance: npt.NDArray[np.bool_] | None = None) -> Prediction:
        """Ridge solution at every point (``variance`` is accepted for interface compatibility; it is cheap here)."""
        parts = [self._solve(self._weights(points.X[s : s + _CHUNK])) for s in range(0, len(points), _CHUNK)]
        cols = [np.concatenate([p[i] for p in parts]) if parts else np.zeros(0) for i in range(5)]
        return Prediction(cols[0], cols[1], cols[2], cols[3], cols[4], prior_var=self.prior_var)

    def loao(self, vectors: bool = False) -> pd.DataFrame:
        """Along-heading prediction of each along observation with the weights of its own aircraft set to zero."""
        idx = np.flatnonzero(self.obs.kind == ALONG)
        rows = []
        for s in range(0, len(idx), _CHUNK):
            q = idx[s : s + _CHUNK]
            W = self._weights(self.obs.X[q])
            W[self.obs.aircraft[q][:, None] == self.obs.aircraft[None, :]] = 0.0
            du, dv, vu, vv, cuv = self._solve(W)
            ex, ey = self.obs.e[q, 0], self.obs.e[q, 1]
            var_field = vu * ex**2 + 2 * cuv * ex * ey + vv * ey**2
            frame = pd.DataFrame(
                {
                    "obs_index": q,
                    "pred": du * ex + dv * ey,
                    "var_field": var_field,
                    "var_total": var_field + self.cfg.sigma_obs**2,
                }
            )
            if vectors:
                frame["du"], frame["dv"] = du, dv
            rows.append(frame)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["obs_index", "pred"])
