from __future__ import annotations

from pathlib import Path

from skywind.config import Settings, load_settings


def test_defaults_are_relative_paths() -> None:
    s = Settings()
    assert not s.paths.data_root.is_absolute() and not s.paths.store_root.is_absolute()
    assert s.paths.commb("20240318").name == "commb_20240318.parquet"
    assert s.paths.opensky("20240318").name == "2024-03-18.parquet"
    assert s.paths.era5("20240318").name == "2024_03_18.pl.nc"


def test_environment_overrides_paths(tmp_path: Path) -> None:
    s = load_settings(env={"SKYWIND_DATA": str(tmp_path / "d"), "SKYWIND_CAMPAIGN": "c1"})
    assert s.paths.cache == tmp_path / "d" / "cache" / "c1"
