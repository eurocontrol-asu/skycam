"""Downstream reproduction of the embedded c16 tables from the prototype's stored intermediates.

Needs the prototype's working folder (``clouds_v3``: ``ref/binmeans_feat.npz``, ``feat/``,
``ref/grid.npz``), given by the environment variable ``SKYCLOUD_C16_PROTOTYPE``; skipped
without it. Read-only. About 4 minutes and 2 GB.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.lib import format as npformat

from skycloud.reference_build.binning import BinMeans
from skycloud.reference_build.features import DayFeatures, load_day_features
from skycloud.reference_build.labels import label_frames
from skycloud.reference_build.params import BuildParams
from skycloud.reference_build.postfit import postfit_tables, table_grid
from skycloud.reference_build.raw_tables import raw_tables
from skycloud.reference_build.smoothing import adapted_spread, rebuild_spreads, smooth_tables
from skycloud.reference_build.thresholds import (
    frame_stats,
    knowable,
    lift_colour,
    release_gate,
    shape_table,
)
from skycloud.tables import REF_FIELDS, SPREAD_FIELDS, ReferenceTables

pytestmark = pytest.mark.integration

PROTO = os.environ.get("SKYCLOUD_C16_PROTOTYPE", "")


def npz_memmap(path: Path) -> dict[str, Any]:
    """Memory-map the members of an UNCOMPRESSED .npz (the prototype's bin means, 853 MB)."""
    out = {}
    with zipfile.ZipFile(path) as z, path.open("rb") as f:
        for info in z.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            f.seek(info.header_offset + 26)
            name_len, extra_len = np.frombuffer(f.read(4), "<u2")
            f.seek(info.header_offset + 30 + int(name_len) + int(extra_len))
            version = npformat.read_magic(f)
            read = (
                npformat.read_array_header_1_0
                if version == (1, 0)
                else npformat.read_array_header_2_0
            )
            shape, fortran, dtype = read(f)
            out[info.filename[:-4]] = np.memmap(
                path, dtype, "r", f.tell(), shape, "F" if fortran else "C"
            )
    return out


@pytest.fixture(scope="module")
def proto() -> Path:
    if not PROTO or not (Path(PROTO) / "ref" / "binmeans_feat.npz").exists():
        pytest.skip("set SKYCLOUD_C16_PROTOTYPE to the prototype's clouds_v3 folder to run this")
    return Path(PROTO)


def test_c16_tables_are_rebuilt_bit_identical_from_the_stored_bin_means(proto: Path) -> None:
    p = BuildParams()
    bm = BinMeans.from_arrays(npz_memmap(proto / "ref" / "binmeans_feat.npz"))
    with np.load(proto / "ref" / "grid.npz") as g:
        az, zen = g["az"].astype(np.float64), g["zen"].astype(np.float64)

    def days() -> Iterator[DayFeatures]:
        for f in sorted((proto / "feat").glob("*.npz")):
            yield load_day_features(f)

    lb = label_frames(bm, p.bins, p.labels)
    with np.load(proto / "ref" / "labels_feat.npz") as stored:
        assert lb.lab.tobytes() == stored["lab"].tobytes()
    labels = lb.by_stamp()
    sm = smooth_tables(raw_tables(bm, lb.lab, p.bins, p.smooth.table_min_frames), p.bins, p.smooth)
    ref = {**sm, **adapted_spread(bm, lb.lab, sm, p.bins, p.smooth)}
    pf, n = postfit_tables(days(), labels, az, zen, sm, p.bins, p.smooth, p.cloudmap)
    ref.update(pf)
    assert n == 3813
    spread = rebuild_spreads(ref, p.smooth)
    emb = ReferenceTables.load()
    for k in REF_FIELDS:
        assert ref[k].tobytes() == emb.ref.tables[k].tobytes(), k
    for k in SPREAD_FIELDS:
        assert spread[k].tobytes() == emb.spread.tables[k].tobytes(), k
    grid3 = table_grid(ref)
    st = frame_stats(days(), labels, az, zen, grid3, p.bins, p.thresholds)
    mine = {
        "c7_shape": shape_table(st, p.thresholds),
        "c8_release": release_gate(st),
        "c11_lift_colour": lift_colour(st, p.thresholds),
        "c9_knowable": knowable(grid3, az, zen, p.thresholds, p.cloudmap),
    }
    for name, content in mine.items():
        expected = {k: v for k, v in emb.table(name).items() if k != "source"}
        assert json.loads(json.dumps(content)) == expected, name
