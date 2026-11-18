"""The reflection table: lookup, visibility ramp, checksum."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skysun import ReflectionTable, packaged_table
from skysun.table import StructureInfo, VisibilityRamp
from tests_skysun.conftest import CENTRE, RING_RADIUS, table


def test_lookup_follows_the_sun_and_wraps_around() -> None:
    t = table(0.04)
    relative = np.array([0.0, 5.0, 85.0, 95.0, 180.0, 355.0, 360.0, -5.0], dtype=np.float32)
    values = t.lookup(0, 30.0, relative)
    assert values[0] == pytest.approx(0.04)  # between sectors 355 and 5, both lit
    assert values[1] == pytest.approx(0.04)
    assert values[2] == pytest.approx(0.04)
    assert values[3] == pytest.approx(0.0)  # sector 95: facing away
    assert values[4] == 0.0
    assert values[5] == values[6] == values[7] == pytest.approx(0.04)
    half = t.lookup(0, 30.0, np.array([90.0], dtype=np.float32))
    assert half[0] == pytest.approx(0.02)  # halfway between a lit and an unlit sector


def test_lookup_interpolates_elevation_and_holds_the_ends() -> None:
    t = table(0.04)
    t.contrast.flags.writeable = True
    t.contrast[0, 0] *= 0.0  # first elevation bin (0-20, centre 10) dark
    low, mid, high = (t.lookup(0, e, np.array([5.0], dtype=np.float32))[0] for e in (-5, 20, 30))
    assert low == 0.0
    assert mid == pytest.approx(0.02)  # halfway between the centres 10 and 30
    assert high == pytest.approx(0.04)


def test_visibility_ramp_reads_the_core_per_elevation_and_the_glow() -> None:
    ramp = VisibilityRamp(
        elevation_edges=[0.0, 20.0, 40.0], high=[0.4, 0.8], glow_low=0.1, glow_high=0.3, origin="t"
    )
    assert ramp.from_core(0.0, 30.0) == 0.0
    assert ramp.from_core(0.4, 5.0) == 1.0  # low sun: top 0.4
    assert ramp.from_core(0.4, 35.0) == pytest.approx(0.5)  # high sun: top 0.8
    assert ramp.from_core(0.4, 20.0) == pytest.approx(0.4 / 0.6)  # between the bin centres
    assert ramp.from_glow(0.05) == 0.0
    assert ramp.from_glow(0.2) == pytest.approx(0.5)
    assert ramp.from_glow(0.5) == 1.0
    with pytest.raises(ValueError, match="per elevation bin"):
        VisibilityRamp(
            elevation_edges=[0.0, 20.0], high=[0.4, 0.8], glow_low=0.1, glow_high=0.3, origin="t"
        )


def test_layer_scale_is_contrast_over_the_reference_clipped_at_1() -> None:
    t = table()
    scaled = t.layer_value(np.array([0.0, 0.04, 0.08, 0.2], dtype=np.float32))
    assert scaled.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])
    assert t.metadata.negligible_value == pytest.approx(0.003 / 0.08)


def test_saved_table_round_trips_and_detects_changes(tmp_path: Path, table_file: Path) -> None:
    loaded = ReflectionTable.from_file(table_file)
    assert loaded.metadata.sha256 == table().metadata.sha256
    assert np.array_equal(loaded.contrast, table().contrast)
    with pytest.raises(FileExistsError):
        loaded.save(table_file)
    with np.load(table_file) as stored:
        arrays = {k: stored[k] for k in stored.files}
    arrays["contrast"] = arrays["contrast"] * 1.01
    np.savez(tmp_path / "tampered.npz", **arrays)
    with pytest.raises(ValueError, match="checksum"):
        ReflectionTable.from_file(tmp_path / "tampered.npz")


def test_other_circles_are_refused() -> None:
    t = table()
    same = [StructureInfo(name="ring0", centre=(CENTRE, CENTRE), radius=RING_RADIUS, disc=False)]
    t.check_structures(same)
    moved = [
        StructureInfo(name="ring0", centre=(CENTRE + 5, CENTRE), radius=RING_RADIUS, disc=False)
    ]
    with pytest.raises(ValueError, match="differ"):
        t.check_structures(moved)


def test_the_embedded_table_loads_and_describes_its_calibration() -> None:
    t = packaged_table()
    meta = t.metadata
    assert meta.camera_id.startswith("gvccs")
    assert meta.frames > 100 and len(meta.days) > 20
    assert meta.image_size_wh == (2464, 2056)
    assert all(0 < h <= 1 for h in meta.visibility.high)
    assert 0 <= meta.visibility.glow_low < meta.visibility.glow_high <= 1
    assert 0.05 < meta.reference_contrast < 0.1  # the review measured 7.4 %
    assert t.contrast.shape == (len(meta.structures), len(meta.elevation_edges) - 1, meta.sectors)
    evaluation = {
        "20230927",
        "20231021",
        "20231101",
        "20240115",
        "20240318",
        "20240510",
        "20231012",
        "20231122",
        "20240206",
        "20240412",
    }
    assert not evaluation & set(meta.days)
