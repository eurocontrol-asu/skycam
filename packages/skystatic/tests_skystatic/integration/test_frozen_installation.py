"""Frozen products are portable, exclusive and bound to camera geometry and period."""

from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from skystatic import (
    FrozenInstallation,
    SkyGeometry,
    StaticConfig,
    freeze_installation,
    load_day_mask,
    open_source,
    run,
)
from tests_skystatic.conftest import DIRT, write_day, zenith_map

pytestmark = pytest.mark.integration
CONFIG = StaticConfig(max_side=0, workers=2)


@pytest.fixture
def frozen(tmp_path: Path) -> FrozenInstallation:
    folders = [write_day(tmp_path / f"reference-{i}", datetime(2024, 1, i, 9)) for i in (1, 2, 3)]
    run(
        [open_source(p) for p in folders],
        SkyGeometry.from_array(zenith_map()),
        CONFIG,
        tmp_path / "reference",
    )
    return freeze_installation(
        tmp_path / "reference",
        tmp_path / "installation.npz",
        camera_id="test-camera",
        version="v1",
        valid_from=date(2024, 1, 1),
        valid_until=date(2025, 1, 1),
    )


def test_frozen_installation_reuses_votes_without_reference_recalculation(
    tmp_path: Path, frozen: FrozenInstallation
) -> None:
    folder = write_day(tmp_path / "new-day", datetime(2024, 5, 10, 9), dirty=True)
    with patch("skystatic.pipeline.installation_mask", side_effect=AssertionError("new vote")):
        report = run(
            [open_source(folder)],
            SkyGeometry.from_array(zenith_map()),
            CONFIG,
            tmp_path / "out",
            installation=frozen,
        )
    assert report.installation_id == frozen.metadata.artifact_id
    assert report.reference_days == ["2024-01-01", "2024-01-02", "2024-01-03"]
    product = load_day_mask(tmp_path / "out/classes_2024-05-10.png")
    assert (product.classes[DIRT] == 2).all()
    assert product.metadata.installation_id == frozen.metadata.artifact_id
    assert np.array_equal(np.asarray(Image.open(tmp_path / "out/installation.png")), frozen.classes)
    assert not frozen.classes.flags.writeable
    assert not frozen.votes.flags.writeable


def test_frozen_version_cannot_be_overwritten(tmp_path: Path, frozen: FrozenInstallation) -> None:
    with pytest.raises(FileExistsError):
        freeze_installation(
            tmp_path / "reference",
            tmp_path / "installation.npz",
            camera_id="test-camera",
            version="v2",
            valid_from=date(2024, 1, 1),
            valid_until=date(2025, 1, 1),
        )
    assert FrozenInstallation.from_file(tmp_path / "installation.npz").metadata == frozen.metadata


@pytest.mark.parametrize("day", [datetime(2023, 12, 31, 9), datetime(2025, 1, 1, 9)])
def test_outside_validity_is_refused_before_computation(
    tmp_path: Path, frozen: FrozenInstallation, day: datetime
) -> None:
    folder = write_day(tmp_path / "outside", day)
    with patch("skystatic.pipeline.compute_day", side_effect=AssertionError("too late")):
        with pytest.raises(ValueError, match="outside frozen installation validity"):
            run(
                [open_source(folder)],
                SkyGeometry.from_array(zenith_map()),
                CONFIG,
                tmp_path / "out",
                installation=frozen,
            )
    assert not (tmp_path / "out").exists()


def test_another_camera_geometry_is_refused(tmp_path: Path, frozen: FrozenInstallation) -> None:
    folder = write_day(tmp_path / "day", datetime(2024, 5, 10, 9))
    with pytest.raises(ValueError, match="different camera geometry"):
        run(
            [open_source(folder)],
            SkyGeometry.from_array(zenith_map() + 1),
            CONFIG,
            tmp_path / "out",
            installation=frozen,
        )


def test_changed_segmentation_parameters_are_refused(
    tmp_path: Path, frozen: FrozenInstallation
) -> None:
    folder = write_day(tmp_path / "day", datetime(2024, 5, 10, 9))
    with pytest.raises(ValueError, match="incompatible parameters"):
        run(
            [open_source(folder)],
            SkyGeometry.from_array(zenith_map()),
            CONFIG.model_copy(update={"edge": 0.1}),
            tmp_path / "out",
            installation=frozen,
        )


def test_corrupted_frozen_pixels_are_detected(tmp_path: Path, frozen: FrozenInstallation) -> None:
    classes = frozen.classes.copy()
    classes[0, 0] = 1 if classes[0, 0] == 0 else 0
    np.savez(
        tmp_path / "corrupted.npz",
        classes=classes,
        votes=frozen.votes,
        metadata=np.array(frozen.metadata.model_dump_json()),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        FrozenInstallation.from_file(tmp_path / "corrupted.npz")


def test_frozen_installation_and_reference_days_are_mutually_exclusive(
    tmp_path: Path, frozen: FrozenInstallation
) -> None:
    folder = write_day(tmp_path / "day", datetime(2024, 5, 10, 9))
    with pytest.raises(ValueError, match="not both"):
        run(
            [open_source(folder)],
            SkyGeometry.from_array(zenith_map()),
            CONFIG,
            tmp_path / "out",
            reference=[open_source(folder)],
            installation=frozen,
        )


def test_empty_validity_interval_is_refused(tmp_path: Path, frozen: FrozenInstallation) -> None:
    with pytest.raises(ValueError, match="valid_until must be after"):
        freeze_installation(
            tmp_path / "reference",
            tmp_path / "invalid.npz",
            camera_id="test-camera",
            version="v2",
            valid_from=date(2024, 1, 1),
            valid_until=date(2024, 1, 1),
        )
    assert not (tmp_path / "invalid.npz").exists()
