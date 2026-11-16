from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from skystatic import SkyGeometry, StaticConfig, ZipSource, open_source, run
from tests_skystatic.conftest import DIRT, STRUCTURE, write_day, write_zip_day, zenith_map

pytestmark = pytest.mark.integration

CONFIG = StaticConfig(parts=3, max_side=0, core_zenith=40, min_days=0.8, tolerance=1, workers=2)


def days(tmp_path: Path, dirty: set[int]) -> list[Path]:
    return [
        write_day(tmp_path / f"2024-01-0{i}", datetime(2024, 1, i, 9), dirty=i in dirty)
        for i in range(1, 6)
    ]


def test_directory_source_lists_frames_in_time_order(tmp_path: Path) -> None:
    folder = write_day(tmp_path / "2024-01-01", datetime(2024, 1, 1, 9), count=5)
    (folder / "notes.txt").write_text("not an image")
    (folder / "image_20240101080000.png").write_bytes(b"")  # earlier, empty file
    source = open_source(folder)
    frames = source.frames()
    assert source.label == "2024-01-01"
    assert [f.timestamp for f in frames] == sorted(f.timestamp for f in frames)
    assert len(frames) == 6
    assert source.read(frames[0]) == b""


def test_pattern_selects_one_exposure(tmp_path: Path) -> None:
    folder = write_day(tmp_path / "day", datetime(2024, 1, 1, 9), count=4)
    for path in list(folder.iterdir())[:2]:
        path.rename(path.with_name(path.stem + "_160.png"))
    assert len(open_source(folder, pattern="*_160.png").frames()) == 2


def test_zip_source_reads_nested_members(tmp_path: Path) -> None:
    archive = write_zip_day(tmp_path / "CAM_20240101.zip", datetime(2024, 1, 1, 9), count=6)
    source = open_source(archive)
    assert isinstance(source, ZipSource)
    assert source.label == "2024-01-01"
    frames = source.frames()
    assert len(frames) == 6
    assert source.read(frames[0])[:4] == b"\x89PNG"


def test_open_source_rejects_other_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path / "missing")


def test_run_separates_installation_from_one_day_of_dirt(tmp_path: Path) -> None:
    folders = days(tmp_path / "raw", dirty={3})
    report = run(
        [open_source(f) for f in folders],
        SkyGeometry.from_array(zenith_map()),
        CONFIG,
        tmp_path / "out",
        cache=tmp_path / "cache",
    )
    installation = np.asarray(Image.open(tmp_path / "out" / "installation.png"))
    assert (installation[STRUCTURE] == 1).all()
    assert (installation[DIRT] == 0).all()
    dirty = np.asarray(Image.open(tmp_path / "out" / "classes_2024-01-03.png"))
    clean = np.asarray(Image.open(tmp_path / "out" / "classes_2024-01-02.png"))
    assert (dirty[DIRT] == 2).all()
    assert not (clean == 2).any()
    by_day = {d.label: d for d in report.days}
    assert by_day["2024-01-03"].variable_components == 1
    assert by_day["2024-01-02"].variable_components == 0
    stored = json.loads((tmp_path / "out" / "report.json").read_text())
    assert stored["reference_days"] == [f.name for f in folders]
    assert stored["days"][0]["frames_daytime"] == 24


def test_dirt_present_on_every_reference_day_needs_longer_reference(tmp_path: Path) -> None:
    # The limit of the method: what never changes over the reference period is installation.
    folders = days(tmp_path / "raw", dirty={1, 2, 3, 4, 5})
    run(
        [open_source(f) for f in folders],
        SkyGeometry.from_array(zenith_map()),
        CONFIG,
        tmp_path / "out",
    )
    installation = np.asarray(Image.open(tmp_path / "out" / "installation.png"))
    assert (installation[DIRT] == 1).all()


def test_separate_reference_days_keep_dirt_variable(tmp_path: Path) -> None:
    dirty = days(tmp_path / "raw", dirty={1, 2, 3, 4, 5})
    clean = [
        write_day(tmp_path / "ref" / f"2025-06-0{i}", datetime(2025, 6, i, 9)) for i in (1, 2, 3)
    ]
    run(
        [open_source(dirty[0])],
        SkyGeometry.from_array(zenith_map()),
        CONFIG,
        tmp_path / "out",
        reference=[open_source(f) for f in clean],
    )
    classes = np.asarray(Image.open(tmp_path / "out" / "classes_2024-01-01.png"))
    assert (classes[DIRT] == 2).all()
    assert (classes[STRUCTURE] == 1).all()


def test_cache_is_reused_and_zip_gives_the_same_masks(tmp_path: Path) -> None:
    folder = write_day(tmp_path / "raw" / "2024-01-01", datetime(2024, 1, 1, 9))
    archive = write_zip_day(tmp_path / "2024-01-01.zip", datetime(2024, 1, 1, 9))
    geometry = SkyGeometry.from_array(zenith_map())
    run([open_source(folder)], geometry, CONFIG, tmp_path / "a", cache=tmp_path / "cache")
    cached = list((tmp_path / "cache").glob("*.npz"))
    assert len(cached) == 1
    stamp = cached[0].stat().st_mtime_ns
    with patch("skystatic.pipeline.aggregate_parts", side_effect=AssertionError("cache miss")):
        run([open_source(folder)], geometry, CONFIG, tmp_path / "b", cache=tmp_path / "cache")
    assert cached[0].stat().st_mtime_ns == stamp
    run([open_source(archive)], geometry, CONFIG, tmp_path / "c")
    first = np.asarray(Image.open(tmp_path / "a" / "classes_2024-01-01.png"))
    assert np.array_equal(first, np.asarray(Image.open(tmp_path / "b" / "classes_2024-01-01.png")))
    assert np.array_equal(first, np.asarray(Image.open(tmp_path / "c" / "classes_2024-01-01.png")))


def test_a_day_without_daylight_is_an_error(tmp_path: Path) -> None:
    folder = tmp_path / "night"
    folder.mkdir()
    for i in range(4):
        Image.fromarray(np.zeros((96, 96, 3), np.uint8)).save(
            folder / f"image_2024010100000{i}.png"
        )
    with pytest.raises(ValueError):
        run([open_source(folder)], SkyGeometry.from_array(zenith_map()), CONFIG, tmp_path / "out")


def test_a_foggy_reference_day_does_not_vote(tmp_path: Path) -> None:
    folders = days(tmp_path / "raw", dirty=set())
    fog = tmp_path / "raw" / "2024-01-09"
    fog.mkdir()
    for i in range(24):  # uniform grey: no skyline, nothing static
        Image.fromarray(np.full((96, 96, 3), 120, np.uint8)).save(
            fog / f"image_202401090900{i:02d}.png"
        )
    strict = CONFIG.model_copy(update={"min_days": 1.0})
    report = run(
        [open_source(folders[0])],
        SkyGeometry.from_array(zenith_map()),
        strict,
        tmp_path / "out",
        reference=[open_source(f) for f in [*folders, fog]],
    )
    assert report.rejected_reference_days == ["2024-01-09"]
    assert report.days[0].skyline_visible
    installation = np.asarray(Image.open(tmp_path / "out" / "installation.png"))
    assert (installation[STRUCTURE] == 1).all()
    votes = np.asarray(Image.open(tmp_path / "out" / "reference_votes.png"))
    assert votes[STRUCTURE].min() == 5


def test_a_foggy_processed_day_is_flagged(tmp_path: Path) -> None:
    folders = days(tmp_path / "raw", dirty=set())
    fog = tmp_path / "raw" / "2024-01-09"
    fog.mkdir()
    for i in range(24):
        Image.fromarray(np.full((96, 96, 3), 120, np.uint8)).save(
            fog / f"image_202401090900{i:02d}.png"
        )
    report = run(
        [open_source(fog), open_source(folders[0])],
        SkyGeometry.from_array(zenith_map()),
        CONFIG,
        tmp_path / "out",
        reference=[open_source(f) for f in folders],
    )
    assert [d.skyline_visible for d in report.days] == [False, True]
