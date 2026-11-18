from __future__ import annotations

import numpy as np

from skycloud.tables import ReferenceTables
from tests_skycloud.conftest import ENODES, table_value


def test_lookup_is_exact_at_the_nodes_and_linear_between(tiny_tables: ReferenceTables) -> None:
    # bin centres: angle g at (i + 0.5) x 2 deg, zenith at (j + 0.5) x 1.5 deg
    g = np.array([[3.0, 4.0]])
    z = np.array([[3.75, 3.75]])
    (nc,) = tiny_tables.ref.lookup(("Nc",), g, z, float(ENODES[2]))
    assert np.allclose(nc, [[table_value(2, 1, 2, 0), table_value(2, 1.5, 2, 0)]], atol=1e-6)
    elev = 0.5 * (ENODES[2] + ENODES[3])
    (no,) = tiny_tables.ref.lookup(("No",), np.array([3.0]), np.array([3.75]), float(elev))
    assert np.isclose(no[0], table_value(2.5, 1, 2, 1), atol=1e-6)


def test_spread_family_is_separate(tiny_tables: ReferenceTables) -> None:
    (a,) = tiny_tables.ref.lookup(("Lo_sd",), np.array([1.0]), np.array([0.75]), 0.0)
    (b,) = tiny_tables.spread.lookup(("Lo_sd",), np.array([1.0]), np.array([0.75]), 0.0)
    assert not np.isclose(a[0], b[0])  # different index of the field in each family


def test_low_count_spread_is_monotone_and_zero_beyond_the_top(tiny_tables: ReferenceTables) -> None:
    s = tiny_tables.low_count_spread(np.array([10.0, 50.0, 100.0, 1000.0]))
    assert np.allclose(
        s, [0.03, 0.03, 0.03, 0.0]
    )  # empty first bin filled, non-increasing, top cut
    assert tiny_tables.low_count_spread_ped(np.array([20.0]))[0] == 0.0


def test_thresholds_are_read(tiny_tables: ReferenceTables) -> None:
    assert tiny_tables.table("c9_knowable")["E_K"] == 6.5
    assert tiny_tables.metadata["camera"] == "synthetic"
