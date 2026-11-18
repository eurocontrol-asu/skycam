"""Equivalence with the frozen prototype c16 on its golden outputs.

The golden folder (env ``SKYCLOUD_GOLDEN``) holds the c16 products of the prototype on a frozen
list of 42 frames, the day ledgers the prototype built, and ``manifest.json`` with the inputs
(raw frames, skystatic masks, calibration maps, site). The tests render the same frames with
the package and require the delivered uint8 products to be EQUAL, byte for byte: no tolerance.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from skycloud import CameraGeometry, RenderSetup, Site, build_ledgers, render
from skycloud.ledger import DayLedger, LedgerSet

pytestmark = pytest.mark.integration

GOLDEN = Path(os.environ.get("SKYCLOUD_GOLDEN", "/nonexistent"))
MANIFEST = GOLDEN / "manifest.json"


def _manifest() -> dict[str, Any]:
    if not MANIFEST.is_file():
        pytest.skip(
            "golden c16 outputs not available: set SKYCLOUD_GOLDEN to the folder "
            f"holding manifest.json ({MANIFEST})"
        )
    m: dict[str, Any] = json.loads(MANIFEST.read_text())
    for key in ("images", "masks_with_metadata", "geometry"):
        if not (GOLDEN / m[key]).exists():
            pytest.skip(f"golden input missing: {GOLDEN / m[key]}")
    return m


def _setup(m: dict[str, Any], ledgers: Path) -> RenderSetup:
    geo = GOLDEN / m["geometry"]
    return RenderSetup(
        geometry=CameraGeometry.from_files(geo, geo),
        site=Site(**m["site"]),
        images=GOLDEN / m["images"],
        mask_dir=GOLDEN / m["masks_with_metadata"],
        ledger_dir=ledgers,
        exclude=frozenset(m["exclude"]),
        allow_unreliable_mask=m["allow_unreliable_mask"],
        workers=int(os.environ.get("SKYCLOUD_TEST_WORKERS", "1")),
    )


def _png(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p))


def _differences(gold: Path, port: Path) -> list[str]:
    out = []
    for f in sorted(gold.rglob("*.png")):
        q = port / f.relative_to(gold)
        if not q.exists():
            out.append(f"missing {f.relative_to(gold)}")
            continue
        a, b = _png(f), _png(q)
        if a.shape != b.shape or not np.array_equal(a, b):
            d = np.abs(a.astype(int) - b.astype(int))
            out.append(f"{f.relative_to(gold)}: {int((d > 0).sum())} px differ, max {int(d.max())}")
    return out


def test_render_equals_the_golden_c16_products(tmp_path: Path) -> None:
    m = _manifest()
    ledgers = tmp_path / "ledgers"
    for legacy, src in ((False, "c10"), (True, "legacy")):
        for f in (GOLDEN / m["ledgers_golden"][src]).glob("*.npz"):
            dst = LedgerSet.path(ledgers, f.stem, legacy)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
    targets = (GOLDEN / m["targets"]).read_text().split()
    out = tmp_path / "out"
    report = render(targets, _setup(m, ledgers), out)
    assert not [r for r in report.frames if "error" in r]
    gold = GOLDEN / m["products"]
    assert _differences(gold, out) == []
    assert json.loads((out / "states.json").read_text()) == json.loads(
        (gold / "states.json").read_text()
    )


def test_ledger_build_equals_the_prototype_ledger(tmp_path: Path) -> None:
    m = _manifest()
    day = os.environ.get("SKYCLOUD_TEST_LEDGER_DAY", "20230927")
    setup = _setup(m, tmp_path)
    build_ledgers([day], setup)
    for legacy, src in ((False, "c10"), (True, "legacy")):
        a = DayLedger.load(GOLDEN / m["ledgers_golden"][src] / f"{day}.npz")
        b = DayLedger.load(LedgerSet.path(tmp_path, day, legacy))
        for k in ("t0", "t1", "delta", "support", "var", "frames", "ped", "anchors"):
            x, y = getattr(a, k), getattr(b, k)
            assert x.shape == y.shape and np.array_equal(x, y), (src, k)
        assert json.loads(a.params) == json.loads(b.params)
