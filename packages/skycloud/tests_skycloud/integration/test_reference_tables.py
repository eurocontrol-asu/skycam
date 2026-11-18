from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from skycloud.tables import EMBEDDED_SHA256, ReferenceTables, file_sha256, pack_reference_tables
from tests_skycloud.conftest import npz_bytes, synthetic_npz, thresholds

pytestmark = pytest.mark.integration


def test_embedded_tables_load_with_their_checksum() -> None:
    t = ReferenceTables.load()
    assert t.sha256 == EMBEDDED_SHA256
    assert t.ref.tables["Nc"].shape == (27, 91, 60)
    assert t.ref.tables["Nc"].dtype == np.float32
    assert t.grid_gp.shape == (16200,)
    assert t.metadata["archive"]["frames"] == 26028
    assert t.table("c9_knowable")["E_K"] == 6.5


def test_wrong_checksum_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ReferenceTables.load(sha256="0" * 64)
    f = tmp_path / "other.npz"
    f.write_bytes(npz_bytes(synthetic_npz()))
    with pytest.raises(ValueError, match="SHA-256"):
        ReferenceTables.load(f, sha256="1" * 64)
    t = ReferenceTables.load(f, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    assert t.metadata["camera"] == "synthetic"


def test_pack_is_lossless(tmp_path: Path) -> None:
    import json

    arrays = synthetic_npz()
    ref = tmp_path / "ref"
    ref.mkdir()
    for fname, prefix in (
        ("c1_ref.npz", "ref"), ("c14_spread.npz", "spread"), ("grid.npz", "grid"),
        ("c3_lowcount.npz", "lowcount"), ("c4_lowcount_ped.npz", "lowcount_ped"),
    ):  # fmt: skip
        np.savez(
            ref / fname,
            **{k.split("/")[1]: v for k, v in arrays.items() if k.startswith(prefix + "/")},
        )
    for name, content in thresholds().items():
        (ref / f"{name}.json").write_text(json.dumps(content))
    out = tmp_path / "packed.npz"
    sha = pack_reference_tables(ref, out, {"camera": "synthetic"})
    assert sha == file_sha256(out)
    t = ReferenceTables.load(out, sha256=sha)
    assert t.metadata["sources_sha256"]["c1_ref.npz"] == file_sha256(ref / "c1_ref.npz")
    with np.load(out) as d, np.load(ref / "c1_ref.npz") as s:
        assert d["ref/Nc"].tobytes() == s["Nc"].tobytes()
    assert t.table("c4_consistency") == thresholds()["c4_consistency"]
