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

from skywind.config import Settings

__all__ = [
    "distance_vs_age",
    "drift_scatter",
    "hl_covariance",
    "make_figures",
    "overlay",
    "overlay_flight",
    "residual_map",
    "skill_per_day",
]

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
    h, w = np.asarray(Image.open(image)).shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    return _save(fig, out)


def _scores(run: Path, keep: Sequence[str], set_name: str) -> pd.DataFrame:
    t = pd.read_csv(run / "summary.csv", dtype={"day": str})
    return t[t["variant"].isin(keep)].assign(set=set_name)


def _best_flights(errors: pd.DataFrame, n: int) -> list[tuple[str, str]]:
    """Flights with the most annotated points (clear, long contrails)."""
    c = errors.groupby(["day", "flight"]).size().sort_values(ascending=False)
    return [(str(d), str(f)) for d, f in c.index[:n]]


def overlay_flight(settings: Settings, contrail_run: Path, day: str, flight: str, method: str, out: Path) -> Path:
    """Camera frame (frame with the most annotated points of the flight) with the annotation, the ERA5-advected and
    the ``method``-advected emission points of that flight, projected with the camera geometry."""
    from skywind.evaluate.gvccs_raw import GvccsRaw
    from skywind.geo import FT, isa_altitude_m
    from skywind.gvccs.dataset import load_day
    from skywind.pipeline import state_vectors

    src = GvccsRaw(settings)
    g = load_day(src.root, day)
    errs = pd.read_parquet(contrail_run / f"errors_{method}.parquet")
    e = errs[(errs["day"].astype(str) == day) & (errs["flight"].astype(str) == flight)]
    t = pd.Timestamp(str(e.groupby("time").size().idxmax()))
    fid = str(e["flight_id"].iloc[0])
    sv, _ = state_vectors(settings, day)
    f = sv[sv["flight_id"] == fid]
    gap = float(((f["geoaltitude"] - f["altitude"]) * FT).median())  # GNSS - barometric, as for the truth lines
    lines = {}
    for name in ("era5", method):
        adv = pd.read_parquet(contrail_run / "advected" / f"{name}_{day}.parquet")
        a = adv[(adv["flight_id"] == fid) & (adv["time"] == t)].sort_values("waypoint")
        h = isa_altitude_m(a["level"].to_numpy() * 100) + (gap if np.isfinite(gap) else 0.0)
        az, zen = src.camera.az_zen(a["latitude"].to_numpy(), a["longitude"].to_numpy(), h)
        ok = zen < np.radians(85)
        x, y = src.camera.pixel(az[ok], zen[ok])
        lines[f"advected with {'ERA5' if name == 'era5' else name}"] = (x, y)
    frame = g.frames[g.frames["time"] == t].iloc[0]
    polys = list(g.polygons[(g.polygons["time"] == t) & (g.polygons["flight"] == flight)]["polygon"])
    image = src.root / "train" / "images" / str(frame["file_name"])
    return overlay(image, polys, lines, f"{day} flight {flight} at {pd.Timestamp(t):%H:%M:%S} UTC", out)


def make_figures(settings: Settings, runs: dict[str, Path], out: Path, n_overlays: int = 4) -> list[Path]:
    """All figures from evaluation runs: ``runs`` keys dev_aircraft, holdout_aircraft, dev_contrails,
    holdout_contrails, diagnostics (any subset)."""
    made: list[Path] = []
    tables = {}
    if "dev_aircraft" in runs:
        tables["dev"] = _scores(runs["dev_aircraft"], ["gp_matern_flow", "ridge"], "dev")
    if "holdout_aircraft" in runs:
        tables["holdout"] = _scores(runs["holdout_aircraft"], ["gp_frozen", "gp_frozen_dayvar", "ridge_dev"], "holdout")
    if tables:
        made.append(skill_per_day(tables, out / "skill_per_day.png"))
    made += _contrail_figures(runs, out)
    if "diagnostics" in runs:
        hl = pd.read_csv(runs["diagnostics"] / "hollingsworth_lonnberg.csv")
        made.append(hl_covariance(hl, settings.gp.model_dump(), out / "hl_covariance.png"))
    if "holdout_contrails" in runs:
        made += _map_and_overlays(settings, runs["holdout_contrails"], out, n_overlays)
    return made


def _contrail_figures(runs: dict[str, Path], out: Path) -> list[Path]:
    sets = {"dev": ("dev_contrails", "gp_matern_flow"), "holdout": ("holdout_contrails", "gp_frozen")}
    paired, ages, made = {}, {}, []
    for name, (key, method) in sets.items():
        if key in runs:
            p = pd.read_csv(runs[key] / "flights_paired.csv", dtype={"day": str})
            paired[name] = p.rename(columns={f"drift_{method}": "drift_gp"})
            ages[name] = pd.read_csv(runs[key] / "age_summary.csv", dtype={"day": str}).replace(
                {"method": {method: "gp"}}
            )
    if paired:
        made.append(drift_scatter(paired, "gp", out / "drift_scatter.png"))
        _COLORS["gp"] = "#1f77b4"
        made.append(distance_vs_age(ages, out / "distance_vs_age.png", ["era5", "gp"]))
    return made


def _map_and_overlays(settings: Settings, run: Path, out: Path, n: int) -> list[Path]:
    made = []
    errs = pd.read_parquet(run / "errors_gp_frozen.parquet")
    flights = _best_flights(errs, n)
    if flights:
        day = flights[0][0]
        obs = pd.read_parquet(settings.paths.observations(day))
        t = pd.Timestamp(errs[errs["day"].astype(str) == day]["time"].median())
        made.append(
            residual_map(run / "fields" / f"gp_frozen_{day}.nc", obs, 250.0, t, out / f"residual_map_{day}.png")
        )
    for day, flight in flights:
        made.append(overlay_flight(settings, run, day, flight, "gp_frozen", out / f"overlay_{day}_{flight}.png"))
    return made
