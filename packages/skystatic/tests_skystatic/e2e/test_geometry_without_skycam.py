from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_geometry_files_work_when_skycam_cannot_be_imported(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"zenith{suffix}"
    zenith = np.array([[0, 45], [90, np.nan]], dtype=np.float32)
    if suffix == ".npy":
        np.save(path, zenith)
    else:
        np.savez(path, zenith_deg=zenith)
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockSkycam(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'skycam', 'cv2', 'numba', 'shapely', 'matplotlib'}:
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

sys.meta_path.insert(0, BlockSkycam())
from skystatic import SkyGeometry
geometry = SkyGeometry.from_file(Path(sys.argv[1]))
assert geometry.below(80).tolist() == [[True, True], [False, False]]
assert geometry.shape == (2, 2)
assert not any(name == 'skycam' or name.startswith('skycam.') for name in sys.modules)
try:
    SkyGeometry.from_skycam(None)
except ImportError as exc:
    assert 'skystatic[skycam]' in str(exc)
else:
    raise AssertionError('Missing optional dependency must be explained')
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
