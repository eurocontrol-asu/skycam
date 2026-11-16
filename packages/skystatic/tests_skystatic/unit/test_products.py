"""Consumer semantics: exact dates, class labels, shape and quality."""

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from skystatic import MaskMetadata, load_day_mask, sky_mask
from skystatic.products import png_metadata


def write_product(
    root: Path,
    name: str = "2024-05-10",
    *,
    visible: bool = True,
    day: date = date(2024, 5, 10),
) -> Path:
    path = root / f"classes_{name}.png"
    metadata = MaskMetadata(
        day=day,
        label=name,
        skyline_visible=visible,
        variable_reliable=visible,
    )
    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(
        path, pnginfo=png_metadata(metadata)
    )
    return path


def test_sky_mask_selects_exact_date_and_resizes_labels(tmp_path: Path) -> None:
    write_product(tmp_path, name="custom-label")
    write_product(tmp_path, name="tomorrow", day=date(2024, 5, 11))
    mask = sky_mask(tmp_path, datetime(2024, 5, 10, 23, tzinfo=UTC), (4, 4))
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, expected)


def test_metadata_is_readable_without_report_json(tmp_path: Path) -> None:
    path = write_product(tmp_path, visible=False)
    mask = load_day_mask(path)
    assert mask.metadata.day == date(2024, 5, 10)
    assert not mask.metadata.skyline_visible
    assert not mask.metadata.variable_reliable
    assert np.array_equal(mask.classes, [[0, 1], [2, 3]])


def test_unreliable_day_requires_explicit_consumer_opt_in(tmp_path: Path) -> None:
    write_product(tmp_path, visible=False)
    with pytest.raises(ValueError, match="unreliable daily mask"):
        sky_mask(tmp_path, datetime(2024, 5, 10), (2, 2))
    assert sky_mask(tmp_path, datetime(2024, 5, 10), (2, 2), allow_unreliable=True).sum() == 1


@pytest.mark.parametrize("size", [(4, 2), (0, 2), (2, -1)])
def test_consumer_refuses_changed_view_or_nonpositive_size(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    path = write_product(tmp_path)
    with pytest.raises(ValueError):
        load_day_mask(path, size)


def test_missing_date_has_no_nearest_day_fallback(tmp_path: Path) -> None:
    write_product(tmp_path)
    with pytest.raises(KeyError, match="No daily mask"):
        sky_mask(tmp_path, datetime(2024, 5, 9), (2, 2))


def test_two_products_for_same_date_are_an_error(tmp_path: Path) -> None:
    write_product(tmp_path, name="a")
    write_product(tmp_path, name="b")
    with pytest.raises(ValueError, match="Several daily masks"):
        sky_mask(tmp_path, datetime(2024, 5, 10), (2, 2))


def test_legacy_png_without_quality_metadata_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "classes_12.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="regenerate this legacy mask"):
        load_day_mask(path)


def test_unknown_classes_are_refused(tmp_path: Path) -> None:
    path = write_product(tmp_path)
    metadata = load_day_mask(path).metadata
    Image.fromarray(np.full((2, 2), 255, np.uint8)).save(path, pnginfo=png_metadata(metadata))
    with pytest.raises(ValueError, match="unknown class"):
        load_day_mask(path)
