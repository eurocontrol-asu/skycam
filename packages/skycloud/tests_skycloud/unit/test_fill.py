from __future__ import annotations

import numpy as np

from skycloud.fill import fill_holes


def test_fill_holes_uses_the_measured_ring_and_widens_it() -> None:
    alpha = np.full((60, 60), 0.2, np.float32)
    conf = np.full((60, 60), 0.8, np.float32)
    valid = np.ones((60, 60), bool)
    hole = np.zeros((60, 60), bool)
    hole[25:35, 25:35] = True
    valid[hole] = False
    alpha[hole] = np.nan
    valid[20:40, 10:30] = False  # the left half of the near ring is not measured (skyline)
    alpha[:, 30:] = 0.4  # the measured ring reads 0.4 on its visible side
    info = fill_holes(alpha, conf, valid, hole, (3, 8), 20, core=hole)
    assert np.allclose(alpha[hole], 0.4) and np.allclose(conf[hole], 0.8)
    assert info["unfilled_px"] == 0 and info["core_value"] == 0.4


def test_a_hole_without_measured_ring_keeps_no_value() -> None:
    alpha = np.full((20, 20), np.nan, np.float32)
    hole = np.ones((20, 20), bool)
    info = fill_holes(alpha, np.zeros((20, 20), np.float32), ~hole, hole, (3,), 20)
    assert np.isnan(alpha).all() and info["unfilled_px"] == 400


def test_each_component_takes_its_own_ring() -> None:
    alpha = np.zeros((40, 80), np.float32)
    alpha[:, 40:] = 0.9
    hole = np.zeros((40, 80), bool)
    hole[18:22, 10:14] = True
    hole[18:22, 60:64] = True
    valid = ~hole
    fill_holes(alpha, np.ones((40, 80), np.float32), valid, hole, (3,), 20)
    assert np.allclose(alpha[18:22, 10:14], 0.0) and np.allclose(alpha[18:22, 60:64], 0.9)
