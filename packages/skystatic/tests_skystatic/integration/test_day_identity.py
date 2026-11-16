"""Dates and source revisions must not alias outputs or cached averages."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from skystatic import SkyGeometry, StaticConfig, open_source, run
from tests_skystatic.conftest import DIRT, scene, write_day, write_zip_day, zenith_map

pytestmark = pytest.mark.integration
CONFIG = StaticConfig(max_side=0, workers=2)


def test_archive_tree_uses_image_dates_in_outputs_and_cache(tmp_path: Path) -> None:
    folders = [
        write_day(tmp_path / "2023/10/12", datetime(2023, 10, 12, 9)),
        write_day(tmp_path / "2024/04/12", datetime(2024, 4, 12, 9), dirty=True),
    ]
    report = run(
        [open_source(p) for p in folders],
        SkyGeometry.from_array(zenith_map()),
        CONFIG,
        tmp_path / "out",
        cache=tmp_path / "cache",
    )
    assert [day.label for day in report.days] == ["2023-10-12", "2024-04-12"]
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2
    clean = np.asarray(Image.open(tmp_path / "out/classes_2023-10-12.png"))
    dirty = np.asarray(Image.open(tmp_path / "out/classes_2024-04-12.png"))
    assert (clean[DIRT] == 0).all()
    assert (dirty[DIRT] == 2).all()


@pytest.mark.parametrize("reference_collision", [False, True])
def test_duplicate_label_fails_before_cache_or_output(
    tmp_path: Path, reference_collision: bool
) -> None:
    a = open_source(write_day(tmp_path / "a", datetime(2023, 10, 12, 9)), label="12")
    b = open_source(write_day(tmp_path / "b", datetime(2024, 4, 12, 9)), label="12")
    output, cache = tmp_path / "out", tmp_path / "cache"
    with patch("skystatic.pipeline.compute_day", side_effect=AssertionError("too late")):
        with pytest.raises(ValueError, match="Duplicate day label"):
            run(
                [a] if reference_collision else [a, b],
                SkyGeometry.from_array(zenith_map()),
                CONFIG,
                output,
                reference=[b] if reference_collision else [],
                cache=cache,
            )
    assert not output.exists()
    assert not cache.exists()


def test_same_day_can_be_processed_and_used_as_reference(tmp_path: Path) -> None:
    folder = write_day(tmp_path / "day", datetime(2024, 4, 12, 9))
    report = run(
        [open_source(folder)],
        SkyGeometry.from_array(zenith_map()),
        CONFIG,
        tmp_path / "out",
        reference=[open_source(folder)],
    )
    assert report.reference_days == ["2024-04-12"]


@pytest.mark.parametrize("archive", [False, True])
def test_mixed_dates_are_rejected_even_with_an_explicit_label(
    tmp_path: Path, archive: bool
) -> None:
    folder = write_day(tmp_path / "mixed", datetime(2024, 4, 12, 23, 59), count=4)
    if archive:
        import zipfile

        path = tmp_path / "mixed.zip"
        with zipfile.ZipFile(path, "w") as stored:
            for image in folder.iterdir():
                stored.write(image, image.name)
    else:
        path = folder
    with pytest.raises(ValueError, match="Expected one image date"):
        run(
            [open_source(path, label="override")],
            SkyGeometry.from_array(zenith_map()),
            CONFIG,
            tmp_path / "out",
        )


def test_same_date_at_another_source_does_not_reuse_cached_images(tmp_path: Path) -> None:
    a = write_day(tmp_path / "camera-a", datetime(2024, 4, 12, 9))
    b = write_day(tmp_path / "camera-b", datetime(2024, 4, 12, 9), dirty=True)
    geometry = SkyGeometry.from_array(zenith_map())
    run([open_source(a)], geometry, CONFIG, tmp_path / "a", cache=tmp_path / "cache")
    run([open_source(b)], geometry, CONFIG, tmp_path / "b", cache=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2
    clean = np.asarray(Image.open(tmp_path / "a/classes_2024-04-12.png"))
    dirty = np.asarray(Image.open(tmp_path / "b/classes_2024-04-12.png"))
    assert (clean[DIRT] == 0).all()
    assert (dirty[DIRT] != 0).all()


def test_source_revision_invalidates_cached_averages(tmp_path: Path) -> None:
    folder = write_day(tmp_path / "day", datetime(2024, 4, 12, 9))
    geometry = SkyGeometry.from_array(zenith_map())
    run([open_source(folder)], geometry, CONFIG, tmp_path / "a", cache=tmp_path / "cache")
    for index, image in enumerate(sorted(folder.iterdir())):
        Image.fromarray(scene(index, dirty=True)).save(image)
    run([open_source(folder)], geometry, CONFIG, tmp_path / "b", cache=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2


def test_zip_label_comes_from_frames_not_archive_name(tmp_path: Path) -> None:
    source = open_source(write_zip_day(tmp_path / "12.zip", datetime(2024, 4, 12, 9)))
    assert source.label == "2024-04-12"


def test_concurrent_runs_do_not_share_temporary_cache_files(tmp_path: Path) -> None:
    folder = write_day(tmp_path / "day", datetime(2024, 4, 12, 9))
    geometry = SkyGeometry.from_array(zenith_map())
    with ThreadPoolExecutor(2) as pool:
        reports = list(
            pool.map(
                lambda name: run(
                    [open_source(folder)],
                    geometry,
                    CONFIG,
                    tmp_path / name,
                    cache=tmp_path / "cache",
                ),
                ["a", "b"],
            )
        )
    assert len(reports) == 2
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    assert not list((tmp_path / "cache").glob("*.tmp.npz"))
