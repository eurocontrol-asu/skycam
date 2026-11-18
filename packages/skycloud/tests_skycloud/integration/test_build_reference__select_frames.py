"""The whole build on a tiny synthetic camera written to disk: steps, packing, resume."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from skycloud.geometry import CameraGeometry, Site
from skycloud.reference_build import (
    BuildInputs,
    BuildParams,
    CarriedTables,
    build_reference,
    select_frames,
)
from skycloud.tables import ReferenceTables
from tests_skycloud.conftest import (
    SYN_PARAMS,
    npz_bytes,
    synthetic_npz,
    thresholds,
    write_synthetic_archive,
)

pytestmark = pytest.mark.integration


def inputs(root: Path) -> BuildInputs:
    carry = root / "carry.npz"
    carry.write_bytes(npz_bytes(synthetic_npz()))
    return BuildInputs(
        camera="synthetic fisheye",
        geometry=CameraGeometry.from_files(root / "geo.npz", root / "geo.npz"),
        sky_mask=np.asarray(Image.open(root / "static_mask.png")) == 0,
        site=Site(latitude_deg=48.6, longitude_deg=2.35),
        frames=select_frames(images=root / "img"),
        carried=CarriedTables.from_archive(carry, hashlib.sha256(carry.read_bytes()).hexdigest()),
    )


def test_build_packs_loadable_tables_and_resumes(tmp_path: Path) -> None:
    write_synthetic_archive(tmp_path)
    params = BuildParams.model_validate(SYN_PARAMS)
    out = tmp_path / "ref.npz"
    report = build_reference(inputs(tmp_path), tmp_path / "work", out, params)
    t = ReferenceTables.load(out, sha256=report.sha256)
    assert t.metadata["archive"]["frames"] == 48 and t.metadata["archive"]["days"] == 2
    assert t.metadata["camera"] == "synthetic fisheye"
    assert t.metadata["params"]["labels"]["node_min_frames"] == 2
    assert t.ref.tables["Nc"].shape == (27, 91, 60) and np.isfinite(t.ref.tables["Nc"]).all()
    # the clear colour index of the blue day, (90 - 60) / 150, and the grey day's 0
    assert np.isclose(np.median(t.ref.tables["Nc"]), 0.2, atol=1e-3)
    assert np.isclose(np.median(t.ref.tables["No"]), 0.0, atol=1e-3)
    assert t.grid_gp.shape == (16200,)
    assert t.table("c4_consistency") == thresholds()["c4_consistency"]  # carried verbatim
    assert set(t.table("c8_release")) == {"T_o", "T_c", "source"}
    assert report.steps["midscale"]["frames_read"] == 24
    assert set(t.table("c6_midscale")) == {"elev", "sd", "sd_p90", "K", "source"}
    assert all(not s["reused"] for s in report.steps.values())
    assert report.steps["labels"]["clear"] > 0 and report.steps["labels"]["overcast"] > 0

    again = build_reference(inputs(tmp_path), tmp_path / "work", out, params)
    assert all(s["reused"] for s in again.steps.values()) and again.sha256 == report.sha256

    changed = params.model_copy(
        update={"labels": params.labels.model_copy(update={"clear_q50_max": 0.1})}
    )
    third = build_reference(inputs(tmp_path), tmp_path / "work", out, changed)
    assert third.steps["features"]["reused"] and third.steps["binmeans"]["reused"]
    assert not third.steps["labels"]["reused"] and not third.steps["postfit"]["reused"]


def test_build_stops_after_a_step_and_needs_carried_tables(tmp_path: Path) -> None:
    write_synthetic_archive(tmp_path)
    i = inputs(tmp_path)
    work = tmp_path / "work"
    report = build_reference(i, work, tmp_path / "x.npz", until="binmeans")
    assert list(report.steps) == ["grid", "features", "binmeans"] and not report.sha256
    assert np.load(tmp_path / "work/binmeans/MN.npy", mmap_mode="r").shape == (48, 5460)
    with pytest.raises(ValueError, match="no clear frame"):  # default thresholds: 40 frames
        build_reference(i, work, tmp_path / "x.npz")
    bare = BuildInputs(i.camera, i.geometry, i.sky_mask, i.site, i.frames)
    with pytest.raises(ValueError, match="not measured by the build"):
        build_reference(bare, tmp_path / "work2", tmp_path / "y.npz")
    assert not (tmp_path / "work2").exists()


def test_select_frames_from_a_list_excludes_days_and_refuses_duplicates(tmp_path: Path) -> None:
    stamps = write_synthetic_archive(tmp_path)["stamps"]
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join([stamps[0], stamps[30], stamps[0]]))
    with pytest.raises(ValueError, match="listed twice"):
        select_frames(images=tmp_path / "img", list_file=lst)
    f = select_frames(images=tmp_path / "img", list_file=lst, duplicates="keep")
    assert f.n_frames == 3 and [t for t, _ in f.listed] == [stamps[0], stamps[30], stamps[0]]
    f = select_frames(images=tmp_path / "img", exclude_days=["20240311"])
    assert sorted(f.days) == ["20240310"] and f.stamps == stamps[:24]
