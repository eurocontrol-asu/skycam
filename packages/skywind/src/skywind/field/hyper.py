"""Hyperparameter estimation from the aircraft data only (never from contrail annotations).

GP: maximum composite marginal likelihood. Each day is cut into time blocks (default 30 min) treated as independent;
inside a block the exact Gaussian log-likelihood and its analytic gradient are computed. Optimised with L-BFGS-B over
log length scales, log standard deviations and the mean airspeed bias.

Ridge baseline: grid search of the leave-one-aircraft-out mean squared error of the along-heading prediction.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from skywind.config import RidgeConfig
from skywind.field.data import ALONG, CROSS, ObsSet
from skywind.field.gp import GPParams, centred, covariance_and_grads
from skywind.field.ridge import RidgeEstimator
from skywind.geo import FloatArray

log = logging.getLogger(__name__)

__all__ = [
    "BOUNDS",
    "FitResult",
    "block_loglik",
    "fit_gp",
    "fit_ridge",
    "free_parameters",
    "make_blocks",
    "params_from_vector",
]

BOUNDS: dict[str, tuple[float, float]] = {
    "log_length_h_km": (math.log(3.0), math.log(400.0)),
    "log_length_z_m": (math.log(30.0), math.log(3000.0)),
    "log_length_t_min": (math.log(1.0), math.log(240.0)),
    "log_sigma_field": (math.log(0.1), math.log(10.0)),
    "log_sigma_bias": (math.log(0.02), math.log(5.0)),
    "mean_bias": (-3.0, 3.0),
    "log_sigma_noise": (math.log(0.1), math.log(5.0)),
    "log_sigma_cross_noise": (math.log(0.1), math.log(10.0)),
    "log_sigma_cross_bias": (math.log(0.02), math.log(10.0)),
    "log_sigma_field2": (math.log(0.05), math.log(10.0)),
    "log_length_h2_km": (math.log(30.0), math.log(2000.0)),
    "log_length_t2_min": (math.log(10.0), math.log(2880.0)),
}


def free_parameters(p: GPParams, has_cross: bool) -> list[str]:
    names = ["log_length_h_km", "log_length_z_m", "log_length_t_min", "log_sigma_field", "log_sigma_noise"]
    if p.use_bias:
        names += ["log_sigma_bias", "mean_bias"]
    if has_cross and p.fit_cross_noise:
        names += ["log_sigma_cross_noise", "log_sigma_cross_bias"]
    if p.sigma_field2 > 0:
        names += ["log_sigma_field2", "log_length_h2_km", "log_length_t2_min"]
    return names


def params_from_vector(base: GPParams, names: Sequence[str], x: Sequence[float] | FloatArray) -> GPParams:
    values: dict[str, Any] = {
        n.removeprefix("log_"): (float(v) if n == "mean_bias" else math.exp(float(v)))
        for n, v in zip(names, x, strict=False)
    }
    return replace(base, **values)


def vector_from_params(p: GPParams, names: Sequence[str]) -> FloatArray:
    d = p.as_dict()
    return np.array([d[n] if n == "mean_bias" else math.log(max(d[n.removeprefix("log_")], 1e-6)) for n in names])


def make_blocks(obs: ObsSet, block_minutes: float = 30.0, max_size: int = 2500, seed: int = 0) -> list[ObsSet]:
    """Disjoint time blocks; blocks larger than ``max_size`` are thinned by whole 30 s observations (random, seeded)."""
    if len(obs) == 0:
        return []
    rng = np.random.default_rng(seed)
    b = np.floor((obs.X[:, 3] - obs.X[:, 3].min()) / (block_minutes * 60)).astype(np.int64)
    out = []
    for k in np.unique(b):
        idx = np.flatnonzero(b == k)
        if len(idx) > max_size:
            idx = np.sort(rng.choice(idx, max_size, replace=False))
        if len(idx) >= 10:
            out.append(obs.subset(idx))
    return out


def block_loglik(obs: ObsSet, p: GPParams, names: Sequence[str]) -> tuple[float, FloatArray]:
    """Exact log-likelihood of one block and its gradient with respect to ``names``."""
    t0 = float(obs.X[:, 3].mean())
    C, dC = covariance_and_grads(obs, p, t0)
    C[np.diag_indices_from(C)] += 1e-8
    L = cho_factor(C, lower=True)
    y = centred(obs, p)
    a = cho_solve(L, y)
    ll = -0.5 * float(y @ a) - float(np.log(np.diag(L[0])).sum()) - 0.5 * len(y) * math.log(2 * math.pi)
    W = np.outer(a, a) - cho_solve(L, np.eye(len(y)))
    grad = []
    for n in names:
        if n == "mean_bias":
            grad.append(float(a @ (obs.kind == ALONG)))
        else:
            grad.append(0.5 * float((W * dC[n]).sum()))
    return ll, np.array(grad)


_BLOCKS: list[ObsSet] = []


def _init(blocks: list[ObsSet]) -> None:
    threadpool_limits(1)  # one BLAS thread per worker process
    global _BLOCKS
    _BLOCKS = blocks


def _eval_block(args: tuple[int, GPParams, list[str]]) -> tuple[float, FloatArray]:
    i, p, names = args
    return block_loglik(_BLOCKS[i], p, names)


@dataclass(frozen=True)
class FitResult:
    params: GPParams
    loglik: float
    n_obs: int
    n_blocks: int
    n_eval: int
    converged: bool
    message: str

    def summary(self) -> dict[str, Any]:
        return {
            **self.params.as_dict(),
            "loglik": self.loglik,
            "n_obs": self.n_obs,
            "n_blocks": self.n_blocks,
            "n_eval": self.n_eval,
            "converged": self.converged,
            "message": self.message,
        }


def fit_gp(
    obs_sets: Sequence[ObsSet],
    base: GPParams,
    block_minutes: float = 30.0,
    max_block: int = 2500,
    workers: int = 8,
    maxiter: int = 60,
    only: Sequence[str] | None = None,
) -> FitResult:
    """Maximum composite marginal likelihood over the blocks of all ``obs_sets`` (e.g. one per training day).

    ``only`` restricts the optimisation to a subset of the free parameters (the others stay at ``base``)."""
    blocks = [b for o in obs_sets for b in make_blocks(o, block_minutes, max_block)]
    has_cross = any(bool((b.kind == CROSS).any()) for b in blocks)
    names = [n for n in free_parameters(base, has_cross) if only is None or n in only]
    bounds = [BOUNDS[n] for n in names]
    counter = {"n": 0}
    order = np.argsort([-len(b) for b in blocks])
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(blocks))), initializer=_init, initargs=(blocks,)
    ) as pool:

        def objective(x: FloatArray) -> tuple[float, FloatArray]:
            counter["n"] += 1
            p = params_from_vector(base, names, x)
            res = list(pool.map(_eval_block, [(int(i), p, names) for i in order]))
            return -sum(r[0] for r in res), -np.sum([r[1] for r in res], axis=0)

        x0 = np.clip(vector_from_params(base, names), [b[0] for b in bounds], [b[1] for b in bounds])
        opt = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    log.info(
        "fit: %d blocks, %d evaluations, loglik %.1f, %s",
        len(blocks),
        counter["n"],
        -float(opt.fun),
        params_from_vector(base, names, opt.x),
    )
    return FitResult(
        params_from_vector(base, names, opt.x),
        -float(opt.fun),
        sum(len(b) for b in blocks),
        len(blocks),
        counter["n"],
        bool(opt.success),
        str(opt.message),
    )


def ridge_loao_mse(obs: ObsSet, cfg: RidgeConfig) -> tuple[float, int]:
    t = RidgeEstimator(obs, cfg).loao()
    err = obs.y[t["obs_index"].to_numpy(int)] - t["pred"].to_numpy(float)
    return float(np.sum(err**2)), len(err)


def fit_ridge(
    obs_sets: Sequence[ObsSet], base: RidgeConfig, grid: dict[str, Sequence[float]] | None = None
) -> pd.DataFrame:
    """LOAO mean squared error of every grid combination pooled over ``obs_sets``; sorted, best first."""
    grid = grid or {
        "length_h_km": [5, 7, 10, 15, 20, 30],
        "length_z_m": [50, 75, 100, 150, 250],
        "length_t_min": [10, 20, 40, 80],
        "sigma_prior": [1.0, 2.0, 4.0],
    }
    rows = []
    for combo in itertools.product(*grid.values()):
        cfg = base.model_copy(update=dict(zip(grid, combo, strict=False)))
        sse, n = map(sum, zip(*(ridge_loao_mse(o, cfg) for o in obs_sets), strict=False))
        rows.append({**dict(zip(grid, combo, strict=False)), "mse": sse / max(n, 1), "n": n})
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
