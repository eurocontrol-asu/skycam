"""Benchmark fixtures.

These extend conftest.py fixtures from tests/ for benchmarks.
"""

import sys
from pathlib import Path

# Add tests/ to path so we can reuse fixtures
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from conftest import (  # noqa: F401
    calibration_data_session,
    calibration_path_session,
    fixtures_path_session,
    projector_session,
    sample_image_session,
)
