from __future__ import annotations

import numpy as np

from skystatic.optics import RingFit, concentric_rings, ring_mask, without_rings

SIZE = 241
CENTRE = (123.0, 118.0)  # off the image centre, as on the real camera


def scene(spot: bool = False) -> np.ndarray:
    rng = np.random.default_rng(2)
    y, x = np.indices((SIZE, SIZE))
    r = np.hypot(x - CENTRE[0], y - CENTRE[1])
    contrast = 0.0004 * rng.standard_normal((SIZE, SIZE))
    contrast += 0.005 * np.exp(-((r - 40) ** 2) / 2)  # bright line
    contrast += 0.004 * np.exp(-((r - 80) ** 2) / 2)  # bright line
    contrast -= 0.004 * np.exp(-((r - 86) ** 2) / 2)  # dark line
    if spot:
        contrast[115:125, 175:185] += 0.01  # a dirt spot on one radius only
    return contrast.astype(np.float32)


def test_the_centre_and_the_bright_lines_are_found_and_masked_narrowly() -> None:
    sky = np.ones((SIZE, SIZE), bool)
    mask, fit = concentric_rings(scene(), sky, (120.0, 120.0), 6, level=0.003, halfwidth=1.5)
    assert abs(fit.centre[0] - CENTRE[0]) <= 1 and abs(fit.centre[1] - CENTRE[1]) <= 1
    radii = sorted(r for r, _, _ in fit.rings)
    assert len(radii) == 2
    assert abs(radii[0] - 40) <= 1 and abs(radii[1] - 80) <= 1
    y, x = np.indices((SIZE, SIZE))
    r = np.hypot(x - CENTRE[0], y - CENTRE[1])
    assert mask[np.abs(r - 40) < 0.8].mean() > 0.95
    assert not mask[np.abs(r - 60) < 5].any()  # plain sky between the lines
    assert not mask[np.abs(r - 86) < 0.5].any()  # dark line: not masked in lines mode


def test_level_mode_masks_dark_lines_too() -> None:
    sky = np.ones((SIZE, SIZE), bool)
    _, fit = concentric_rings(scene(), sky, (120.0, 120.0), 6, level=0.003, halfwidth=0)
    assert any(peak < 0 for _, _, peak in fit.rings)


def test_a_spot_on_one_radius_is_not_a_ring() -> None:
    sky = np.ones((SIZE, SIZE), bool)
    _, fit = concentric_rings(scene(spot=True), sky, (120.0, 120.0), 6, level=0.003, halfwidth=1.5)
    radii = [r for r, _, _ in fit.rings]
    assert all(abs(r - 58) > 3 for r in radii)  # the spot sits at r ~ 58


def radius_map() -> np.ndarray:
    y, x = np.indices((SIZE, SIZE))
    return np.hypot(x - CENTRE[0], y - CENTRE[1])


def test_removing_the_rings_keeps_a_spot_that_crosses_one() -> None:
    fit = RingFit(centre=CENTRE, noise=0.0004, rings=[(40.0, 40.0, 0.005), (80.0, 80.0, 0.004)])
    contrast = scene()
    contrast[int(CENTRE[1]) - 2 : int(CENTRE[1]) + 3, 161:166] += 0.01  # a spot on r 40
    flat = without_rings(contrast, np.ones((SIZE, SIZE), bool), fit.centre, [40.0, 80.0], reach=10)
    r = radius_map()
    assert np.abs(flat[(np.abs(r - 80) < 0.8)]).mean() < 0.0015  # the line is gone
    assert flat[int(CENTRE[1]), 163] > 0.008  # the spot stays


def test_a_ring_off_the_common_centre_is_masked_along_its_whole_circle() -> None:
    # One reflection is a circle whose centre is 2 px from the others': fitted as its own circle,
    # it is masked all around, also where it merges with a concentric line.
    y, x = np.indices((SIZE, SIZE))
    contrast = scene()
    own = (CENTRE[0] + 1.7, CENTRE[1] + 1.1)
    r_own = np.hypot(x - own[0], y - own[1])
    contrast += (0.004 * np.exp(-((r_own - 44.5) ** 2) / 2)).astype(np.float32)
    mask, fit = concentric_rings(
        contrast, np.ones((SIZE, SIZE), bool), (120.0, 120.0), 6, level=0.003, halfwidth=1.5, own=4
    )
    circles = fit.circles()
    off = [(c, r) for c, r in circles if abs(r - 44.5) < 1]
    assert len(off) == 1
    (cx, cy), _ = off[0]
    assert abs(cx - own[0]) < 0.6 and abs(cy - own[1]) < 0.6
    assert mask[np.abs(r_own - 44.5) < 0.5].mean() > 0.97  # the whole circle
    assert any(abs(r - 40) < 1 for _, r in circles)  # the concentric neighbour is kept
    assert np.array_equal(ring_mask(fit, (SIZE, SIZE), 1.5), mask)  # a frozen fit draws the same


def test_two_rings_closer_than_the_masking_width_are_one_band() -> None:
    from skystatic.optics import circle_bands

    y, x = np.indices((SIZE, SIZE))
    inner = (CENTRE, 40.0)
    outer = ((CENTRE[0] + 1.2, CENTRE[1] + 0.7), 45.0)  # 3.6 px apart on one side, 6.4 on the other
    far = (CENTRE, 70.0)
    mask = circle_bands([inner, outer, far], (SIZE, SIZE), 1.5)
    d_in = np.hypot(x - CENTRE[0], y - CENTRE[1])
    d_out = np.hypot(x - outer[0][0], y - outer[0][1])
    assert mask[(d_in > 41.5) & (d_out < 43.5)].all()  # no sky between the two
    assert not mask[(d_in > 50) & (d_in < 68)].any()  # a ring 25 px away stays a separate band
