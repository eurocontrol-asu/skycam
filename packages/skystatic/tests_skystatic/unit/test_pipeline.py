from __future__ import annotations

import numpy as np

from skystatic.pipeline import day_classes, skyline_glow

SIZE = 101


def circle(radius: float, halfwidth: float = 1.5) -> np.ndarray:
    y, x = np.indices((SIZE, SIZE))
    return np.abs(np.hypot(x - 50, y - 50) - radius) <= halfwidth


def test_sky_enclosed_by_ring_lines_stays_sky_and_a_mark_beside_a_ring_is_kept() -> None:
    domain = np.ones((SIZE, SIZE), bool)
    rings = circle(30) | circle(32.2)  # two rings of the installation, 2.2 px apart
    day = np.zeros((SIZE, SIZE), bool)
    day[10:16, 10:16] = True  # a mark in the open sky
    classes = day_classes(day, rings, domain, circle(0, 10), rings, tolerance=2)
    between = (np.abs(np.hypot(*(np.indices((SIZE, SIZE)) - 50)) - 31.1) < 0.4) & ~rings
    assert (classes[between] == 0).all()  # see-through ring lines: no enclosed sky filled
    assert (classes[10:16, 10:16] == 2).all()


def test_the_glow_of_a_lit_roof_is_not_a_mark_foliage_and_dirt_are() -> None:
    y, x = np.indices((SIZE, SIZE))
    zenith = (np.hypot(x - 50, y - 50) * 1.8).astype(np.float32)  # 90 degrees at r = 50
    installation = y >= 90  # the skyline: a roof along the bottom, zenith 72-90
    marks = np.zeros((SIZE, SIZE), bool)
    flips = np.zeros((SIZE, SIZE), bool)
    marks[85:90, 30:70] = True  # a band hugging the roof...
    flips[85:90, 30:70] = True  # ...that changes sign with its lighting: glow
    marks[85:90, 5:25] = True  # foliage of the day along the skyline: one sign
    marks[45:50, 45:50] = True  # dirt near the zenith that changes sign, touching nothing
    flips[45:50, 45:50] = True
    glow = skyline_glow(marks, flips, installation, zenith, 60.0, reach=3)
    assert glow[85:90, 30:70].all()
    assert not glow[85:90, 5:25].any() and not glow[45:50, 45:50].any()
