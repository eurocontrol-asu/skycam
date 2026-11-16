"""Dense calibrated reprojection, independent of legacy fixtures and geometry."""

from pathlib import Path

import numpy as np
import pytest

from skycam.domain import CalibratedReprojection, SamplingMap
from skycam.domain.models import CalibrationData


def plane_directions(east, north):
    """An independent equidistant sky-plane oracle, including canonical zenith."""
    return np.stack([np.arctan2(east, north), np.hypot(east, north)], axis=-1)


@pytest.fixture
def camera():
    y, x = np.indices((7, 9))
    directions = plane_directions((x - 4) * 0.08, (3 - y) * 0.08)
    return CalibratedReprojection(directions[..., 0], directions[..., 1])


def test_identity_non_square_xy_and_edges(camera):
    grid = camera.directions
    mapping = camera.sampling_map(grid, target_shape=(7, 9))
    y, x = np.indices((7, 9))
    np.testing.assert_allclose(mapping.source_xy, np.stack([x, y], axis=-1))
    assert mapping.valid.all()
    assert mapping.source_shape == (7, 9)
    assert mapping.target_shape == (7, 9)
    labels = (100 * y + x).astype(np.uint16)
    np.testing.assert_array_equal(mapping.sample_labels(labels), labels)
    image = np.stack([x, y, labels], axis=-1).astype(np.uint8)
    np.testing.assert_array_equal(mapping.sample_image(image), image)
    assert not mapping.source_xy.flags.writeable
    assert not mapping.valid.flags.writeable


def test_cardinal_orientation_and_point_order(camera):
    # x grows east, y grows south: these absolute checks catch axis transposition.
    directions = [[0, 0.16], [np.pi / 2, 0.16], [-np.pi / 2, 0.16], [np.pi, 0.16]]
    np.testing.assert_allclose(
        camera.directions_to_pixels(directions), [[4, 1], [6, 3], [2, 3], [4, 5]]
    )
    np.testing.assert_allclose(camera.pixels_to_directions([6, 3]), [np.pi / 2, 0.16])


def test_seam_and_multiple_wraps(camera):
    directions = [[-np.pi, 0.16], [np.pi, 0.16], [3 * np.pi, 0.16]]
    np.testing.assert_allclose(camera.directions_to_pixels(directions), [[4, 5]] * 3)
    queries = np.array([[-np.pi + 1e-5, 0.17], [np.pi - 1e-5, 0.17]])
    xy = camera.directions_to_pixels(queries)
    np.testing.assert_allclose(xy[:, 1], [5.125] * 2, atol=1e-9)
    assert xy[0, 0] < 4 < xy[1, 0]
    np.testing.assert_allclose(camera.pixels_to_directions(xy), queries, atol=1e-12)


def test_zenith_azimuth_is_irrelevant(camera):
    queries = np.column_stack([[-np.pi, -1, 0, 1, np.pi], np.zeros(5)])
    np.testing.assert_allclose(camera.directions_to_pixels(queries), [[4, 3]] * 5)
    np.testing.assert_array_equal(camera.pixels_to_directions([4, 3]), [0, 0])
    # Interpolating opposite azimuths around zenith must not average azimuths.
    np.testing.assert_allclose(
        camera.pixels_to_directions([4.1, 3]), [np.pi / 2, 0.008]
    )


def test_invalid_directions_and_outside_hull(camera):
    queries = [
        [np.nan, 0.1],
        [0, np.nan],
        [np.inf, 0.1],
        [0, -0.1],
        [0, np.pi],
        [0, 1.0],
        [np.pi / 4, 0.5],
    ]
    assert np.isnan(camera.directions_to_pixels(queries)).all()
    points = [[-0.1, 2], [9, 2], [1, 7], [1, np.inf], [np.nan, 1]]
    assert np.isnan(camera.pixels_to_directions(points)).all()
    grid = np.asarray(queries[:6]).reshape(2, 3, 2)
    mapping = camera.sampling_map(grid)
    assert not mapping.valid.any()
    np.testing.assert_array_equal(
        mapping.sample_labels(np.ones((7, 9), dtype=np.uint16), fill_value=60001),
        np.full((2, 3), 60001, dtype=np.uint16),
    )


def test_calibration_holes_are_not_bridged(camera):
    directions = camera.directions
    directions[2:5, 3:6] = np.nan
    holed = CalibratedReprojection(directions[..., 0], directions[..., 1])
    assert np.isnan(holed.directions_to_pixels([[0, 0], [np.pi / 2, 0.04]])).all()
    assert np.isnan(holed.pixels_to_directions([[3.5, 3], [2.5, 3]])).all()
    # A valid native point beside invalid pixels remains valid.
    np.testing.assert_allclose(holed.pixels_to_directions([2, 3]), [-np.pi / 2, 0.16])
    np.testing.assert_allclose(
        holed.directions_to_pixels([[-np.pi / 2, 0.16]]), [[2, 3]]
    )


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16, np.int64, np.uint64])
def test_nearest_preserves_bool_and_large_ids(camera, dtype):
    grid = camera.directions[:, ::-1]
    mapping = camera.sampling_map(grid)
    labels = np.zeros((7, 9), dtype=dtype)
    if dtype == np.bool_:
        labels[1:6, 2] = True
    else:
        labels[1:6, 2] = 60001
        labels[4, 3:8] = 65535
        if dtype in (np.int64, np.uint64):
            labels[2, 4] = 2**60 + 123  # Would be corrupted by a float detour.
    sampled = mapping.sample_labels(labels)
    assert sampled.dtype == labels.dtype
    np.testing.assert_array_equal(sampled, labels[:, ::-1])


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16])
def test_dense_curved_thin_instances_preserve_overlap(camera, dtype):
    masks = np.zeros((3, 7, 9), dtype=dtype)
    masks[0, 3, :] = True  # Single-pixel horizontal trail, becomes a curve.
    masks[1, :, 4] = True  # Overlap with first trail must remain two instances.
    masks[2, 1, :3] = True  # A disjoint third instance.
    y, x = np.indices((7, 9))
    source_y = y + np.sin(x * np.pi / 4)
    directions = plane_directions((x - 4) * 0.08, (3 - source_y) * 0.08)
    mapping = camera.sampling_map(directions)
    sampled = mapping.sample_instances(masks)
    assert sampled.shape == masks.shape
    assert sampled.dtype == masks.dtype
    assert sampled[0, 3, 4] and sampled[1, 3, 4]
    assert not np.any(sampled[0] & sampled[2])
    expected = np.zeros_like(sampled)
    inside = (source_y >= 0) & (source_y <= 6)
    rows = np.floor(source_y[inside] + 0.5).astype(int)
    expected[:, inside] = masks[:, rows, x[inside]]
    np.testing.assert_array_equal(sampled, expected)
    assert sampled[0, 2, 2] and sampled[0, 4, 6]
    assert not sampled[0, 3, 2] and not sampled[0, 3, 6]
    assert mapping.sample_instances(np.empty((0, 7, 9), dtype=bool)).shape == (0, 7, 9)


def test_continuous_bilinear_rgb_and_opacity(camera):
    xy = np.array([[[2.25, 1.75], [7.5, 5.5]], [[0, 0], [8, 6]]])
    mapping = camera.sampling_map(camera.pixels_to_directions(xy))
    y, x = np.indices((7, 9))
    rgb = np.stack([2 * x + 3 * y, x, y], axis=-1).astype(np.uint8)
    expected = np.stack(
        [2 * xy[..., 0] + 3 * xy[..., 1], xy[..., 0], xy[..., 1]], axis=-1
    )
    sampled = mapping.sample_image(rgb)
    assert sampled.dtype == np.float64
    np.testing.assert_allclose(sampled, expected, atol=1e-12)
    opacity = (x + y) / 20
    np.testing.assert_allclose(mapping.sample_image(opacity), xy.sum(axis=-1) / 20)
    nearest = mapping.sample_image(rgb, interpolation="nearest")
    indices = np.floor(xy + 0.5).astype(int)
    np.testing.assert_array_equal(nearest, rgb[indices[..., 1], indices[..., 0]])


def test_point_round_trip_with_nonlinear_calibration():
    y, x = np.indices((17, 23))
    east = (x - 11) * 0.035 * (1 + 0.002 * (y - 8))
    north = (8 - y) * 0.035 * (1 + 0.003 * (x - 11))
    grid = plane_directions(east, north)
    camera = CalibratedReprojection(grid[..., 0], grid[..., 1])
    xy = np.array([[3.12, 5.43], [11.4, 8.2], [20.11, 12.74]])
    np.testing.assert_allclose(
        camera.directions_to_pixels(camera.pixels_to_directions(xy)), xy, atol=0.002
    )
    assert camera._inverse is not None
    cached = camera._inverse
    camera.directions_to_pixels([[0, 0.1]])
    assert camera._inverse is cached


def test_explicit_crop_and_target_grid_used_exactly(camera):
    grid = np.pad(camera.directions, ((1, 1), (2, 3), (0, 0)), constant_values=np.nan)
    with pytest.raises(ValueError, match="explicit crop"):
        camera.sampling_map(grid, target_shape=(7, 9))
    mapping = camera.sampling_map(grid, crop=(1, 1, 2, 3), target_shape=(7, 9))
    assert mapping.valid.all()
    assert mapping.target_shape == (7, 9)
    shifted = camera.directions[1:5, 2:7]
    mapping = camera.sampling_map(shifted, target_shape=(4, 5))
    y, x = np.indices((4, 5))
    np.testing.assert_allclose(mapping.source_xy, np.stack([x + 2, y + 1], axis=-1))


def test_explicit_legacy_to_geographic_conversion(camera):
    calibration = CalibrationData(
        azimuth_array=camera.directions[..., 0],
        zenith_array=camera.directions[..., 1],
        image_size=(7, 9),
    )
    legacy = CalibratedReprojection.from_calibration(calibration)
    geographic = CalibratedReprojection.from_calibration(
        calibration, azimuth_offset=np.pi
    )
    np.testing.assert_allclose(legacy.directions_to_pixels([np.pi / 2, 0.16]), [6, 3])
    np.testing.assert_allclose(
        geographic.directions_to_pixels([-np.pi / 2, 0.16]), [6, 3]
    )
    np.testing.assert_allclose(
        geographic.pixels_to_directions([6, 3]), [-np.pi / 2, 0.16]
    )
    bad = calibration.model_copy(update={"image_size": (9, 7)})
    with pytest.raises(ValueError, match="image_size"):
        CalibratedReprojection.from_calibration(bad)


def test_map_save_reuse_and_invalid_fill(tmp_path: Path):
    xy = np.array([[[0.5, 0], [3, 1]], [[np.nan, 1], [-1, 0]]])
    mapping = SamplingMap((2, 4), xy)
    path = tmp_path / "sampling-map.bin"
    mapping.save(path)
    loaded = SamplingMap.load(path)
    np.testing.assert_array_equal(loaded.source_xy, mapping.source_xy)
    np.testing.assert_array_equal(loaded.valid, mapping.valid)
    image = np.arange(8, dtype=np.uint16).reshape(2, 4) * 9000
    np.testing.assert_array_equal(
        loaded.sample_labels(image, fill_value=65535), [[9000, 63000], [65535, 65535]]
    )
    sampled = loaded.sample_image(image, fill_value=np.nan)
    np.testing.assert_allclose(sampled[0], [4500, 63000])
    assert np.isnan(sampled[1]).all()
    assert loaded.source_shape == (2, 4)


def test_no_nan_bleed_from_zero_weight_corners():
    mapping = SamplingMap((2, 2), np.array([[[0, 0], [0.5, 0]]]))
    image = np.array([[1, np.nan], [np.nan, np.nan]])
    result = mapping.sample_image(image)
    assert result[0, 0] == 1
    assert np.isnan(result[0, 1])


@pytest.mark.parametrize("bad_crop", [(-1, 0, 0, 0), (7, 0, 0, 0), (1, 2, 3)])
def test_bad_crop(camera, bad_crop):
    with pytest.raises(ValueError, match="crop"):
        camera.sampling_map(camera.directions, crop=bad_crop)


def test_invalid_inputs(camera):
    with pytest.raises(ValueError, match="Points"):
        camera.directions_to_pixels([1, 2, 3])
    with pytest.raises(ValueError, match="matching"):
        CalibratedReprojection(np.ones((2, 3)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="azimuth_offset"):
        CalibratedReprojection(np.ones((2, 3)), np.ones((2, 3)), azimuth_offset=np.nan)
    with pytest.raises(ValueError, match="Target directions"):
        camera.sampling_map([[0, 0]])
    mapping = camera.sampling_map(camera.directions)
    with pytest.raises(ValueError, match="source image shape"):
        mapping.sample_labels(np.ones((9, 7), dtype=np.uint16))
    with pytest.raises(TypeError, match="Labels"):
        mapping.sample_labels(np.ones((7, 9), dtype=float))
    with pytest.raises(ValueError, match="dtype range"):
        mapping.sample_labels(np.ones((7, 9), dtype=np.uint16), fill_value=65536)
    with pytest.raises(ValueError, match="Boolean"):
        mapping.sample_labels(np.ones((7, 9), dtype=bool), fill_value=2)
    with pytest.raises(ValueError, match="Instance masks"):
        mapping.sample_instances(np.ones((7, 9), dtype=bool))
    with pytest.raises(ValueError, match="interpolation"):
        mapping.sample_image(np.ones((7, 9)), interpolation="cubic")
    with pytest.raises(TypeError, match="Images"):
        mapping.sample_image(np.ones((7, 9), dtype=complex))
    with pytest.raises(ValueError, match="nondegenerate"):
        CalibratedReprojection(np.zeros((3, 4)), np.zeros((3, 4))).directions_to_pixels(
            [0, 0]
        )
