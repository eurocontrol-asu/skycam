from __future__ import annotations

import numpy as np
import pytest

from skystatic.geometry import SkyGeometry


def test_radians_are_converted_and_invalid_pixels_are_outside() -> None:
    geometry = SkyGeometry.from_array(np.array([[0.0, np.pi / 2], [np.nan, np.pi / 4]]), "rad")
    assert geometry.zenith_deg[0, 1] == pytest.approx(90.0)
    assert np.isinf(geometry.zenith_deg[1, 0])
    assert geometry.below(90).tolist() == [[True, False], [False, True]]


def test_resized_samples_pixel_centres() -> None:
    zenith = np.arange(16, dtype=np.float32).reshape(4, 4)
    small = SkyGeometry.from_array(zenith).resized((2, 2))
    assert small.zenith_deg.tolist() == [[5.0, 7.0], [13.0, 15.0]]
    assert SkyGeometry.from_array(zenith).resized((4, 4)).zenith_deg is not None


def test_resized_rejects_another_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="different aspects"):
        SkyGeometry.from_array(np.zeros((4, 4))).resized((8, 2))


def test_a_zenith_map_must_be_two_dimensional() -> None:
    with pytest.raises(ValueError, match="2-D"):
        SkyGeometry.from_array(np.zeros(5))
