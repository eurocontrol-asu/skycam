"""Markdown tables of evaluation runs (A: aircraft, B: contrails), as reported in the README."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = ["aircraft_report", "contrail_report", "markdown_table", "table_a", "table_b"]

_A_COLS = ["day", "n_obs", "rmse_era5", "rmse_field", "skill", "z_sd", "cover68", "cover95"]


def markdown_table(df: pd.DataFrame, digits: int = 2) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _wide(scores: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    base = scores[scores["variant"] == variants[0]][["day", "n_obs", "rmse_era5"]].copy()
    for v in variants:
        s = scores[scores["variant"] == v].set_index("day")
        base[f"rmse_{v}"] = base["day"].map(s["rmse_field"])
        base[f"skill_{v}"] = base["day"].map(s["skill"])
    base["n_obs"] = base["n_obs"].astype(int)
    return base


def aircraft_report(run: Path) -> str:
    """Per-day and pooled scores of every variant, fitted hyperparameters, calibration and diagnostics."""
    scores = pd.read_csv(run / "summary.csv", dtype={"day": str})
    variants = list(dict.fromkeys(scores["variant"]))
    parts = [
        "### A. Leave-one-aircraft-out along-heading residual, leave-one-day-out hyperparameters",
        "",
        markdown_table(_wide(scores, [v for v in variants if v in ("gp", "ridge")])),
        "",
        "All variants, pooled:",
        "",
        markdown_table(scores[scores["day"] == "pooled"][["variant", *_A_COLS[1:]]]),
        "",
    ]
    fits = pd.read_csv(run / "fits_gp.csv", dtype={"held_out": str})
    cols = [
        "held_out",
        "length_h_km",
        "length_z_m",
        "length_t_min",
        "sigma_field",
        "sigma_bias",
        "mean_bias",
        "sigma_noise",
    ]
    parts += ["GP hyperparameters per fold (fitted without the held-out day):", "", markdown_table(fits[cols]), ""]
    if (run / "transfer.csv").is_file():
        t = pd.read_csv(run / "transfer.csv", dtype={"day": str})
        lodo = scores[(scores["variant"] == "gp") & (scores["day"] != "pooled")].set_index("day")["skill"]
        piv = t.pivot(index="day", columns="params", values="skill").reset_index()
        piv["lodo"] = piv["day"].map(lodo)
        parts += [
            "Transfer (LOAO skill of each day): parameters from the other days (lodo), all days, own day:",
            "",
            markdown_table(piv),
            "",
        ]
    if (run / "diagnostics.json").is_file():
        d = json.loads((run / "diagnostics.json").read_text())
        hl = pd.DataFrame(d["hl_gaussian_fits"]).T.reset_index(names="dimension")
        parts += [
            "Hollingsworth-Lonnberg Gaussian fits (c0 = background variance, length):",
            "",
            markdown_table(hl),
            "",
            "Desroziers ratios (1 = consistent):",
            "",
            markdown_table(pd.DataFrame(d["desroziers"])[["day", "R_ratio", "HBH_ratio"]]),
            "",
        ]
    return "\n".join(parts)


def contrail_report(run: Path) -> str:
    """Drift and offset per day, distance by age, share of flights improved, uncertainty calibration."""
    drifts = pd.read_csv(run / "drift_summary.csv", dtype={"day": str})
    variants = [c.removeprefix("improved_") for c in drifts.columns if c.startswith("improved_")]
    cols = ["day", "n_flights", "drift_med_era5", "drift_p90_era5"]
    for v in variants:
        cols += [f"drift_med_{v}", f"drift_p90_{v}", f"improved_{v}"]
    off = ["day", "offset_med_era5", "offset_abs_med_era5", *[f"offset_abs_med_{v}" for v in variants]]
    ages = pd.read_csv(run / "age_summary.csv", dtype={"day": str})
    pooled = ages[ages["day"] == "pooled"].pivot(
        index="age_bin", columns="method", values=["abs_med", "abs_p90", "drift_only_med"]
    )
    pooled.columns = ["_".join(map(str, c)) for c in pooled.columns]
    pooled = pooled.loc[sorted(pooled.index, key=lambda b: float(str(b).strip("(]").split(",")[0]))]
    parts = [
        "### B. Contrail advection vs GVCCS labels (m/min for drift, m for distances)",
        "",
        "Drift error |slope of cross-track error vs age| per flight, median / p90, share of flights improved:",
        "",
        markdown_table(drifts[cols], 1),
        "",
        "Offset at emission (not a wind effect):",
        "",
        markdown_table(drifts[off], 0),
        "",
        "Pooled |cross-track distance| by age (raw; drift_only = flight offset removed):",
        "",
        markdown_table(pooled.reset_index(), 0),
        "",
    ]
    cal = run / "uncertainty_calibration.csv"
    if cal.is_file():
        parts += ["Per-flight drift / (60 x posterior sd along the normal):", "", markdown_table(pd.read_csv(cal)), ""]
    return "\n".join(parts)


_A_KEEP = {
    "gp_matern_flow": "GP (frozen config, LODO fits)",
    "ridge": "ridge (LODO grid)",
    "gp_frozen": "GP frozen",
    "gp_frozen_dayvar": "GP frozen + day variances",
    "ridge_dev": "ridge (dev kernel)",
}


def table_a(dev_run: Path | None, holdout_run: Path | None) -> pd.DataFrame:
    """Per day and pooled scores of evaluation A, dev (leave-one-day-out) and holdout (frozen) side by side."""
    parts = []
    for run, set_name in ((dev_run, "dev"), (holdout_run, "holdout")):
        if run is None:
            continue
        t = pd.read_csv(run / "summary.csv", dtype={"day": str})
        t = t[t["variant"].isin(_A_KEEP)].assign(set=set_name)
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t["label"] = t["variant"].map(_A_KEEP)
    cols = [
        "set",
        "day",
        "variant",
        "label",
        "n_obs",
        "n_aircraft",
        "rmse_era5",
        "rmse_field",
        "skill",
        "z_mean",
        "z_sd",
        "cover68",
        "cover95",
    ]
    return t[cols]


def table_b(run: Path) -> dict[str, pd.DataFrame]:
    """Evaluation B tables of a run: drift / offset per day, distance by age, paired tests."""
    return {
        "drift": pd.read_csv(run / "drift_summary.csv", dtype={"day": str}),
        "age": pd.read_csv(run / "age_summary.csv", dtype={"day": str}),
        "tests": pd.read_csv(run / "paired_tests.csv", dtype={"day": str}),
    }
