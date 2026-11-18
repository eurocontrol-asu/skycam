"""The rings of the optics leave class 1 by default and travel as typed circles."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import binary_dilation

from skystatic import (
    RingFit,
    SkyGeometry,
    StaticConfig,
    freeze_installation,
    load_day_mask,
    load_optics,
    open_source,
    run,
)
from skystatic.optics import ring_mask
from tests_skystatic.conftest import DIRT, SIZE, write_day, zenith_map

pytestmark = pytest.mark.integration

FIT = RingFit(
    centre=(47.5, 47.5), noise=1e-4, rings=[(15.0, 15.0, 0.004)], eye=2.0, centres=[(47.0, 48.0)]
)
HALFWIDTH = 16.0  # 1.25 px at the synthetic width of 96 (given at 1232)


def fake_rings(voters: list[Any], fixed: NDArray[np.bool_], *_: Any) -> tuple[Any, RingFit]:
    """What the ring fit would find on a real camera: one circle and the eye."""
    return ring_mask(FIT, fixed.shape, HALFWIDTH * SIZE / 1232), FIT


def run_mode(tmp_path: Path, ring_class: str, name: str) -> Path:
    days = [
        write_day(tmp_path / name / f"d{i}", datetime(2024, 1, i, 9), dirty=i == 2)
        for i in (1, 2, 3)
    ]
    config = StaticConfig(max_side=0, workers=2, ring_halfwidth=HALFWIDTH, ring_class=ring_class)  # type: ignore[arg-type]
    with patch("skystatic.pipeline._permanent_rings", side_effect=fake_rings):
        run(
            [open_source(p) for p in days],
            SkyGeometry.from_array(zenith_map()),
            config,
            tmp_path / name / "out",
        )
    return tmp_path / name / "out"


@pytest.fixture
def outputs(tmp_path: Path) -> tuple[Path, Path]:
    return run_mode(tmp_path, "sky", "sky"), run_mode(tmp_path, "installation", "old")


def test_ring_bands_leave_class_1_and_nothing_else_changes(outputs: tuple[Path, Path]) -> None:
    new, old = outputs
    band = ring_mask(FIT, (SIZE, SIZE), HALFWIDTH * SIZE / 1232)
    for name in ("installation.png", "classes_2024-01-01.png", "classes_2024-01-02.png"):
        after = np.asarray(Image.open(new / name))
        before = np.asarray(Image.open(old / name))
        assert (before[band] == 1).all()  # behaviour before: the rings are installation
        assert np.array_equal(after[~band], before[~band])
        # default: the rings are sky, except where variable static lies on both sides
        assert set(np.unique(after[band]).tolist()) <= {0, 2}
        near_dirt = binary_dilation(before == 2, iterations=3)
        assert not (band & (after == 2) & ~near_dirt).any()
        assert (after[band & ~near_dirt] == 0).all()
    dirty = load_day_mask(new / "classes_2024-01-02.png")
    # the dirt crosses the ring: no line of sky is left inside it
    assert (dirty.classes[DIRT] == 2).all()


def test_circles_are_written_next_to_the_masks_and_scaled(outputs: tuple[Path, Path]) -> None:
    new, _ = outputs
    optics = load_optics(new)
    assert optics is not None
    assert optics.size_wh == (SIZE, SIZE)
    assert optics.circles[0].radius == 15.0
    assert optics.circles[0].centre == (47.0, 48.0)
    double = load_optics(new / "optics.json", size=(2 * SIZE, 2 * SIZE))
    assert double is not None
    assert double.circles[0].radius == 30.0
    assert double.circles[0].centre == (94.5, 96.5)  # pixel centres: (x + 0.5) * 2 - 0.5
    assert double.fit.eye == 4.0


def test_frozen_installation_carries_the_circles(
    tmp_path: Path, outputs: tuple[Path, Path]
) -> None:
    new, _ = outputs
    frozen = freeze_installation(
        new,
        tmp_path / "camera.npz",
        camera_id="test-camera",
        version="v1",
        valid_from=date(2024, 1, 1),
        valid_until=date(2025, 1, 1),
    )
    from_npz = load_optics(tmp_path / "camera.npz")
    assert from_npz is not None
    assert from_npz.fit == FIT
    assert frozen.optics == from_npz
    band = from_npz.band()
    assert (frozen.classes[band & (frozen.classes != 3)] == 0).all()
    day = write_day(tmp_path / "later", datetime(2024, 5, 10, 9), dirty=True)
    config = StaticConfig(max_side=0, workers=2, ring_halfwidth=HALFWIDTH)
    run(
        [open_source(day)],
        SkyGeometry.from_array(zenith_map()),
        config,
        tmp_path / "re",
        installation=frozen,
    )
    reused = load_day_mask(tmp_path / "re/classes_2024-05-10.png")
    near_dirt = binary_dilation(reused.classes == 2, iterations=3)
    assert (reused.classes[band & ~near_dirt] == 0).all()
    assert (reused.classes[DIRT] == 2).all()
