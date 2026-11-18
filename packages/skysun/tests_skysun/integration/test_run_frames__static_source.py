"""A run on files: skystatic's products in, one set of layers per frame out, resumable."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from skystatic import SkyGeometry, StaticConfig, freeze_installation, open_source, run
from skysun import CameraGeometry, SunCamera
from skysun.frames import FrameFile, frame_time, list_frames
from skysun.products import frame_paths, load_layer, run_frames
from skysun.static import StaticSource
from tests_skysun.conftest import MORNING, SITE, frame, maps, optics, table

pytestmark = pytest.mark.integration


def write_frames(
    folder: Path, geometry: CameraGeometry, start: datetime, count: int, step_s: int = 1200
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        time = start + timedelta(seconds=step_s * i)
        Image.fromarray(frame(geometry, time)).save(folder / f"image_{time:%Y%m%d%H%M%S}.png")


def static_run(tmp_path: Path, geometry: CameraGeometry) -> Path:
    """skystatic on the synthetic camera (one day), with the synthetic ring's optics.json."""
    day = tmp_path / "raw" / "2024-03-18"
    write_frames(day, geometry, MORNING, 12)
    output = tmp_path / "static"
    run(
        [open_source(day)],
        SkyGeometry.from_array(maps()[1]),
        StaticConfig(max_side=200, workers=2),
        output,
    )
    (output / "optics.json").write_text(optics().model_dump_json())
    return output


def test_frames_get_their_layers_and_a_rerun_skips_them(
    tmp_path: Path, geometry: CameraGeometry, mocker: object
) -> None:
    static = StaticSource(static_run(tmp_path, geometry))
    camera = SunCamera(geometry, SITE, static.optics(geometry.size), table(0.04))
    write_frames(tmp_path / "burst", geometry, MORNING + timedelta(seconds=5), 3, step_s=30)
    frames = list_frames([tmp_path / "burst"])
    broken = tmp_path / "burst" / "image_20240318120000.png"
    broken.write_bytes(b"not an image")
    frames = [*frames, FrameFile(broken, frame_time(broken))]
    result = run_frames(
        frames, camera, lambda f: static.sky(f.time.date(), geometry.size), tmp_path / "out"
    )
    infos = result.done
    assert len(infos) == 3
    assert result.unreadable == [broken]
    paths = frame_paths(tmp_path / "out", frames[0])
    zone, refl = load_layer(paths["zone"]), load_layer(paths["reflections"])
    assert zone.shape == refl.shape == geometry.shape
    assert zone.max() == 1.0  # the saturated sun
    assert 0.48 < refl.max() < 0.52  # the predicted ring: contrast 0.04 over the reference 0.08
    assert not paths["json"].with_name(paths["json"].name.replace(".json", ".png")).exists()
    info = json.loads(paths["json"].read_text())
    assert info["profile_frames"] >= 2  # the run renders with its neighbours' median profile
    patched = mocker.patch("skysun.products.measure_frame", side_effect=AssertionError)  # type: ignore[attr-defined]
    again = run_frames(
        frames, camera, lambda f: static.sky(f.time.date(), geometry.size), tmp_path / "out"
    )
    assert [i.time for i in again.done] == [i.time for i in infos]
    assert not patched.called


def test_frozen_installation_gives_sky_and_circles(
    tmp_path: Path, geometry: CameraGeometry
) -> None:
    output = static_run(tmp_path, geometry)
    (output / "optics.json").unlink()
    freeze_installation(
        output,
        tmp_path / "camera.npz",
        camera_id="synthetic",
        version="v1",
        valid_from=date(2024, 1, 1),
        valid_until=date(2025, 1, 1),
    )
    static = StaticSource(tmp_path / "camera.npz")
    sky = static.sky(date(2024, 3, 18), geometry.size)
    assert sky.shape == geometry.shape
    assert sky[200, 200] and not sky[0, 0]
    circles = static.optics(geometry.size)  # skystatic's own fit on the synthetic frames
    assert circles is not None and circles.size_wh == geometry.size
    with pytest.raises(ValueError, match="validity"):
        static.sky(date(2026, 1, 1), geometry.size)
