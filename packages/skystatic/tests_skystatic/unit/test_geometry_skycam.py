from __future__ import annotations

import numpy as np
import pytest

from skystatic.geometry import SkyGeometry


def test_skycam_calibration_uses_radians_and_preserves_native_pixels() -> None:
    models = pytest.importorskip("skycam.domain.models")
    zenith = np.array([[0.0, np.pi / 4, np.pi / 2], [np.nan, np.inf, -np.inf]])
    calibration = models.CalibrationData(
        azimuth_array=np.zeros((2, 3)), zenith_array=zenith, image_size=(2, 3)
    )
    geometry = SkyGeometry.from_skycam(calibration)
    assert geometry.shape == (2, 3)
    assert geometry.zenith_deg.dtype == np.float32
    np.testing.assert_allclose(geometry.zenith_deg[0], [0, 45, 90])
    assert np.isinf(geometry.zenith_deg[1]).all()
    assert geometry.below(80).tolist() == [[True, True, False], [False, False, False]]
    assert zenith[0, 1] == pytest.approx(np.pi / 4)
    assert np.isnan(zenith[1, 0])


def test_skycam_calibration_rejects_inconsistent_image_size() -> None:
    models = pytest.importorskip("skycam.domain.models")
    calibration = models.CalibrationData(
        azimuth_array=np.zeros((2, 3)), zenith_array=np.zeros((2, 3)), image_size=(3, 2)
    )
    with pytest.raises(ValueError, match="image_size"):
        SkyGeometry.from_skycam(calibration)


def test_skycam_calibration_rejects_a_non_image_map() -> None:
    models = pytest.importorskip("skycam.domain.models")
    calibration = models.CalibrationData(
        azimuth_array=np.zeros(6), zenith_array=np.zeros(6), image_size=(2, 3)
    )
    with pytest.raises(ValueError, match="2-D"):
        SkyGeometry.from_skycam(calibration)


def test_skycam_constructor_rejects_unrelated_objects() -> None:
    pytest.importorskip("skycam.domain.models")
    with pytest.raises(TypeError, match="CalibrationData"):
        SkyGeometry.from_skycam(object())  # type: ignore[arg-type]
