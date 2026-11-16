from __future__ import annotations

import numpy as np

from skystatic.render import CLASS_COLOURS, overlay


def test_overlay_marks_each_class() -> None:
    image = np.full((6, 6, 3), 20, np.float32)
    classes = np.zeros((6, 6), np.uint8)
    classes[0, 0] = 2
    classes[2:5, 2:5] = 1
    classes[5, 5] = 3
    picture = overlay(image, classes)
    assert tuple(picture[0, 0]) == CLASS_COLOURS[2]
    assert tuple(picture[2, 2]) == CLASS_COLOURS[1]  # outline of the installation
    assert picture[3, 3, 0] > picture[3, 3, 2]  # interior tinted red
    assert tuple(picture[1, 1]) == (110, 110, 110)  # dark sky brought to mid-grey
    assert picture[5, 5].max() < picture[1, 1].min()  # outside dimmed


def test_overlay_gain_can_be_fixed() -> None:
    image = np.full((2, 2, 3), 50, np.float32)
    assert overlay(image, np.zeros((2, 2), np.uint8), gain=2.0)[0, 0, 0] == 100
