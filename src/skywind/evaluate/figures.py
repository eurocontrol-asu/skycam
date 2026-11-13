"""Figures regenerated from the CSV / parquet outputs of the evaluation commands (no recomputation of the results).

Each function reads files written by ``skywind evaluate ...`` and writes one PNG.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

__all__ = ["distance_vs_age", "drift_scatter", "hl_covariance", "overlay", "residual_map", "skill_per_day"]

_COLORS = {
    "era5": "#7f7f7f",
    "gp_matern_flow": "#1f77b4",
    "gp_frozen": "#1f77b4",
    "gp_frozen_dayvar": "#17becf",
    "ridge": "#ff7f0e",
    "ridge_dev": "#ff7f0e",
    "gp_matern": "#2ca02c",
}


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def skill_per_day(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    """Bar chart of the LOAO skill per day; ``tables`` maps a set name (dev / holdout) to a scores table with columns
    day, variant, skill."""
    fig, ax = plt.subplots(figsize=(11, 4))
    x0 = 0
    ticks, labels = [], []
    for set_name, t in tables.items():
        t = t[t["day"] != "pooled"]
        variants = list(dict.fromkeys(t["variant"]))
        days = sorted(t["day"].unique())
        width = 0.8 / len(variants)
        for j, v in enumerate(variants):
            s = t[t["variant"] == v].set_index("day").reindex(days)["skill"]
            ax.bar(
                x0 + np.arange(len(days)) + j * width,
                s.to_numpy(),
                width,
                color=_COLORS.get(v),
                label=f"{v} ({set_name})",
            )
        ticks += list(x0 + np.arange(len(days)) + 0.4 - width / 2)
        labels += [f"{d}\n{set_name}" for d in days]
        x0 += len(days) + 1
    ax.set_xticks(ticks, labels, rotation=90, fontsize=7)
    ax.set_ylabel("skill = 1 - MSE / MSE(ERA5)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, out)


def drift_scatter(paired: dict[str, pd.DataFrame], method: str, out: Path) -> Path:
    """Per-flight |drift| with ERA5 vs with ``method`` (paired table of evaluation B), one colour per set."""
    fig, ax = plt.subplots(figsize=(5, 5))
    vmax = 0.0
    for (name, t), c in zip(paired.items(), ("#1f77b4", "#d62728"), strict=False):
        a, b = t["drift_era5"].abs(), t[f"drift_{method}"].abs()
        ax.scatter(a, b, s=14, alpha=0.7, color=c, label=f"{name} ({len(t)} flights)")
        vmax = max(vmax, float(a.max()), float(b.max()))
    ax.plot([0, vmax], [0, vmax], "k--", lw=0.8)
    ax.set_xlabel("|drift error| ERA5 (m/min)")
    ax.set_ylabel(f"|drift error| {method} (m/min)")
    ax.legend(fontsize=8)
    return _save(fig, out)


def distance_vs_age(ages: dict[str, pd.DataFrame], out: Path, methods: Sequence[str]) -> Path:
    """Median |cross-track distance| by age bin, raw and offset-removed, per set."""
    fig, axs = plt.subplots(1, len(ages), figsize=(5 * len(ages), 4), squeeze=False)
    for ax, (name, t) in zip(axs[0], ages.items(), strict=True):
        p = t[t["day"] == "pooled"].copy()
        p["mid"] = p["age_bin"].map(lambda b: np.mean([float(v) for v in str(b).strip("(]").split(",")]))
        for m in methods:
            q = p[p["method"] == m].sort_values("mid")
            ax.plot(q["mid"], q["abs_med"], "-o", color=_COLORS.get(m), label=f"{m}")
            ax.plot(q["mid"], q["drift_only_med"], "--", color=_COLORS.get(m), alpha=0.6)
        ax.set_title(f"{name} (solid: raw, dashed: flight offset removed)", fontsize=9)
        ax.set_xlabel("age (min)")
        ax.set_ylabel("median |cross-track distance| (m)")
        ax.legend(fontsize=8)
    return _save(fig, out)


def _matern(d: np.ndarray, length: float) -> np.ndarray:
    r = np.sqrt(3.0) * np.abs(d) / length
    return np.asarray((1 + r) * np.exp(-r))


def hl_covariance(hl: pd.DataFrame, params: dict[str, float], out: Path, min_pairs: int = 200) -> Path:
    """Binned cross-aircraft innovation covariance vs separation, with the fitted Matern kernel sigma^2 k(d)."""
    scale = {"horizontal_km": ("length_h_km", 1.0), "vertical_m": ("length_z_m", 1.0), "time_s": ("length_t_min", 60.0)}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (dim, (key, mult)) in zip(axs, scale.items(), strict=True):
        t = hl[(hl["dimension"] == dim) & (hl["pairs"] >= min_pairs)]
        ax.plot(t["separation"], t["covariance"], "o", ms=4, label="cross-aircraft pairs")
        d = np.linspace(0, float(hl[hl["dimension"] == dim]["separation"].max()), 200)
        ax.plot(d, params["sigma_field"] ** 2 * _matern(d, params[key] * mult), "-", label="fitted Matern")
        ax.set_xlabel(dim.replace("_", " "))
        ax.set_ylabel("covariance (m$^2$/s$^2$)")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=8)
    return _save(fig, out)


def residual_map(
    field_nc: Path,
    obs: pd.DataFrame,
    level_hpa: float,
    time: pd.Timestamp,
    out: Path,
    dz_m: float = 300.0,
    dt_min: float = 15.0,
) -> Path:
    """Residual wind (arrows) and its posterior sd (colour) at one level and time, with the observations used nearby
    (within ``dz_m`` and ``dt_min``; colour = along-heading residual)."""
    import xarray as xr

    from skywind.geo import FT, isa_altitude_m

    ds = (
        xr.open_dataset(field_nc)
        .sel(level=level_hpa, method="nearest")
        .sel(time=time.tz_localize(None) if time.tzinfo else time, method="nearest")
    )
    sd = (0.5 * (ds["residual_eastward_wind_std"] ** 2 + ds["residual_northward_wind_std"] ** 2)) ** 0.5
    fig, ax = plt.subplots(figsize=(7.5, 6))
    m = ax.pcolormesh(
        ds["longitude"], ds["latitude"], sd.transpose("latitude", "longitude"), cmap="viridis", shading="auto"
    )
    fig.colorbar(m, ax=ax, label="posterior sd of the residual (m/s)")
    q = ds.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
    ax.quiver(
        q["longitude"],
        q["latitude"],
        q["residual_eastward_wind"].transpose("latitude", "longitude"),
        q["residual_northward_wind"].transpose("latitude", "longitude"),
        color="w",
        scale=60,
    )
    z0 = float(isa_altitude_m(float(ds["level"]) * 100))
    t0 = pd.Timestamp(ds["time"].values.item()).tz_localize("UTC").value / 1e9
    near = obs[(np.abs(obs["altitude"] * FT - z0) <= dz_m) & (np.abs(obs["t"] - t0) <= dt_min * 60)]
    sc = ax.scatter(
        near["longitude"],
        near["latitude"],
        c=near["d_along"],
        cmap="RdBu_r",
        vmin=-4,
        vmax=4,
        s=8,
        edgecolors="k",
        linewidths=0.2,
    )
    fig.colorbar(sc, ax=ax, label="observed along-heading residual (m/s)")
    ax.set_title(
        f"residual at {float(ds['level']):.0f} hPa, {pd.Timestamp(ds['time'].values.item()):%Y-%m-%d %H:%M} UTC "
        f"({len(near)} obs within {dz_m:.0f} m / {dt_min:.0f} min)",
        fontsize=9,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return _save(fig, out)


def overlay(
    image: Path, polygons: list[list[float]], lines: dict[str, tuple[np.ndarray, np.ndarray]], title: str, out: Path
) -> Path:
    """GVCCS camera frame with the annotated contrail polygon(s) and advected lines (pixel coordinates)."""
    from PIL import Image

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.asarray(Image.open(image)))
    for poly in polygons:
        xy = np.asarray(poly).reshape(-1, 2)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="none", edgecolor="#ff3b30", lw=1.2)
    for (name, (x, y)), c in zip(lines.items(), ("#ffd60a", "#30d5ff", "#34c759"), strict=False):
        ax.plot(x, y, "-", color=c, lw=1.3, label=name)
    ax.plot([], [], color="#ff3b30", label="GVCCS annotation")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    return _save(fig, out)
