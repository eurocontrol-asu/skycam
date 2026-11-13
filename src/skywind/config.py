"""Configuration: pydantic models, loaded from defaults, an optional TOML file and environment variables.

No absolute data path lives in the code. Data roots default to the two gitignored symlinks of the repository
(``data`` -> working data, ``store`` -> durable data), resolved against the current directory, and can be redirected
with ``SKYWIND_DATA`` / ``SKYWIND_STORE`` or a ``skywind.toml`` / ``skywind.local.toml`` file.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "DecodeConfig",
    "ExportConfig",
    "GPConfig",
    "ObservationConfig",
    "Paths",
    "RidgeConfig",
    "Settings",
    "Site",
    "SourcesConfig",
    "load_settings",
]

ENV_PATHS = {
    "SKYWIND_DATA": "data_root",
    "SKYWIND_STORE": "store_root",
    "SKYWIND_CAMPAIGN": "campaign",
    "SKYWIND_PROTOTYPE": "prototype_dir",
    "SKYWIND_GVCCS": "gvccs_dir",
}


class Paths(BaseModel):
    """Where data lives. ``inputs`` (durable) / ``cache`` and ``runs`` (working) are derived from the roots."""

    data_root: Path = Path("data")
    store_root: Path = Path("store")
    campaign: str = "20261113-skywind"
    prototype_dir: Path = Path("prototype")
    gvccs_dir: Path = Path("gvccs")

    @property
    def inputs(self) -> Path:
        return self.store_root / "inputs" / self.campaign

    @property
    def cache(self) -> Path:
        return self.data_root / "cache" / self.campaign

    @property
    def runs(self) -> Path:
        return self.data_root / "runs" / self.campaign

    def commb(self, day: str) -> Path:
        return self.inputs / "commb" / f"commb_{day}.parquet"

    def commb_log(self, day: str) -> Path:
        return self.inputs / "commb" / f"commb_{day}.json"

    def opensky(self, day: str) -> Path:
        return self.inputs / "opensky" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"

    def era5(self, day: str) -> Path:
        return self.inputs / "era5" / f"{day[:4]}_{day[4:6]}_{day[6:]}.pl.nc"

    def days_csv(self) -> Path:
        return self.inputs / "days.csv"

    def decoded(self, day: str) -> Path:
        return self.cache / "decoded" / f"messages_{day}.parquet"

    def receivers(self, day: str) -> Path:
        return self.cache / "decoded" / f"receivers_{day}.csv"

    def observations(self, day: str) -> Path:
        return self.cache / "observations" / f"obs_{day}.parquet"

    def heading_table(self) -> Path:
        return self.cache / "calibration" / "heading_offsets_by_day.parquet"

    def fields(self, day: str) -> Path:
        return self.cache / "fields" / f"wind_{day}.nc"

    def advected(self, day: str, tag: str) -> Path:
        return self.cache / "advected" / f"advected_{day}_{tag}.parquet"


class Site(BaseModel):
    """Ground camera and analysis domain."""

    lat: float = 48.6005
    lon: float = 2.3468
    box_deg: float = 1.5
    min_alt_ft: float = 25000.0
    max_alt_ft: float = 41000.0


class DecodeConfig(BaseModel):
    """Register identification and receiver-clock check (see :mod:`skywind.identify`)."""

    max_sv_gap_s: float = 10.0
    tol_gs_kt: float = 12.0
    tol_track_deg: float = 6.0
    tol_ias_kt: float = 12.0
    tol_heading_track_deg: float = 30.0
    receiver_min_replies: int = 200
    receiver_min_share_gs2: float = 0.92
    receiver_min_share_trk05: float = 0.85


class ObservationConfig(BaseModel):
    """One wind observation per (flight, 30 s bin) and its quality filters (see :mod:`skywind.observations`)."""

    bin_s: float = 30.0
    min_msgs_per_register: int = 2
    max_roll_deg: float = 3.0
    max_vrate_fpm: float = 1000.0
    max_dir_sd_deg: float = 0.5
    max_speed_sd_kt: float = 4.0


class GPConfig(BaseModel):
    """Gaussian-process (optimal interpolation) estimator of the ERA5 residual wind.

    Prior: each residual component is a zero-mean GP with variance ``sigma_field**2`` and a covariance separable in
    horizontal distance, pressure altitude and time. Observation (along the heading): projection of the residual on the
    heading + per-aircraft airspeed bias ~ N(mean_bias, sigma_bias**2) + white noise sigma_noise.
    Defaults are the values retained from the leave-one-day-out marginal-likelihood fit (see README).
    """

    kernel: Literal["gaussian", "matern32"] = "gaussian"
    length_h_km: float = 40.0
    length_z_m: float = 300.0
    length_t_min: float = 15.0
    sigma_field: float = 2.0
    sigma_bias: float = 1.0
    mean_bias: float = -0.25
    sigma_noise: float = 1.2
    use_bias: bool = True
    flow_following: bool = False
    use_cross: bool = False
    sigma_cross_noise: float = 2.0
    sigma_cross_bias: float = 1.0
    cutoff: float = 3.0
    tile_minutes: float = 15.0
    max_neighbours: int = 3000


class RidgeConfig(BaseModel):
    """Prototype kernel-weighted ridge least-squares estimator (baseline)."""

    length_h_km: float = 75.0
    length_z_m: float = 500.0
    length_t_min: float = 20.0
    sigma_obs: float = 1.5
    sigma_prior: float = 2.0
    sigma_cross: float = 2.0
    use_cross: bool = False


class ExportConfig(BaseModel):
    """Gridded export (ERA5 horizontal grid, extra pressure levels, finer time step)."""

    half_width_deg: float = 2.5
    horizontal_refine: int = 2
    level_step_hpa: float = 5.0
    min_level_hpa: float = 150.0
    max_level_hpa: float = 400.0
    time_step_min: float = 10.0
    pad_before_min: float = 90.0
    pad_after_min: float = 10.0


class SourcesConfig(BaseModel):
    """Where a new day is fetched from. Remote paths are configuration (``skywind.toml``), not code."""

    remote_host: str | None = None
    opensky_pattern: str | None = None
    era5_pattern: str | None = None
    commb_padding_min: float = 45.0


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    site: Site = Field(default_factory=Site)
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    gp: GPConfig = Field(default_factory=GPConfig)
    ridge: RidgeConfig = Field(default_factory=RidgeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    max_workers: int = 8


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides(env: dict[str, str]) -> dict[str, Any]:
    paths = {field: env[var] for var, field in ENV_PATHS.items() if env.get(var)}
    return {"paths": paths} if paths else {}


def load_settings(config_file: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Defaults < ``skywind.toml`` < ``skywind.local.toml`` (current directory) < ``config_file`` < environment."""
    data: dict[str, Any] = {}
    candidates = [Path("skywind.toml"), Path("skywind.local.toml")]
    if config_file is not None:
        candidates.append(config_file)
    for path in candidates:
        if path.is_file():
            data = _merge(data, tomllib.loads(path.read_text()))
    data = _merge(data, _env_overrides(dict(os.environ) if env is None else env))
    return Settings.model_validate(data)
