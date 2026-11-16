from __future__ import annotations

from pathlib import Path

import pytest

from skywind.config import Settings, load_settings

DAY = "20240206"


@pytest.fixture(scope="session")
def real_settings() -> Settings:
    s = load_settings()
    if not s.paths.commb(DAY).is_file():
        pytest.skip(f"input store not reachable ({s.paths.inputs})")
    return s


@pytest.fixture
def tmp_settings(real_settings: Settings, tmp_path: Path) -> Settings:
    """Real input store, working data redirected to a temporary directory."""
    s = real_settings.model_copy(deep=True)
    s.paths.data_root = tmp_path / "data"
    return s
