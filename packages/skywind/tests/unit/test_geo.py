from __future__ import annotations

import numpy as np
import pytest

from skywind.geo import (
    KT,
    angle_diff_deg,
    cas_from_mach,
    circular_mean_deg,
    circular_sd_deg,
    isa_altitude_m,
    isa_pressure_pa,
    local_xy_km,
    offset_latlon,
    unit_vectors,
)


def test_isa_round_trip_and_reference_values() -> None:
    h = np.array([0.0, 5000.0, 11000.0, 12500.0])
    np.testing.assert_allclose(isa_altitude_m(isa_pressure_pa(h)), h, atol=0.05)
    assert isa_pressure_pa(0.0) == pytest.approx(101325.0)
    assert isa_pressure_pa(11000.0) == pytest.approx(22632, rel=1e-4)


def test_cas_from_mach_at_sea_level_equals_tas() -> None:
    assert cas_from_mach(0.3, 101325.0) == pytest.approx(0.3 * 340.294, rel=1e-6)


def test_angles() -> None:
    assert angle_diff_deg(10.0, 350.0) == pytest.approx(20.0)
    assert circular_mean_deg(np.sin(np.radians(359.0)), np.cos(np.radians(359.0))) == pytest.approx(359.0)
    assert circular_sd_deg(0.0, 1.0) == pytest.approx(0.0)
    hx, hy, qx, qy = unit_vectors(90.0)
    np.testing.assert_allclose([hx, hy, qx, qy], [1, 0, 0, -1], atol=1e-12)


def test_local_projection_and_offsets_are_consistent() -> None:
    la, lo = offset_latlon(48.0, 2.0, 1000.0, 2000.0)
    x, y = local_xy_km(la, lo, 48.0, 2.0)
    assert x == pytest.approx(1.0, rel=1e-3) and y == pytest.approx(2.0, rel=1e-6)
    assert KT == pytest.approx(1852 / 3600, rel=1e-5)
