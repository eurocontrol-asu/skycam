"""Command line: ``skysun run``, ``skysun measure`` and ``skysun build-table``."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import cyclopts

from skysun.calibrate import BuildConfig, MeasureConfig, build_table, measure_archive
from skysun.frames import list_frames
from skysun.geometry import CameraGeometry
from skysun.pipeline import SunCamera
from skysun.products import run_frames
from skysun.static import StaticSource
from skysun.sun import Site
from skysun.table import ReflectionTable, packaged_table

__all__ = ["app", "main"]

app = cyclopts.App(name="skysun", help="Where and how much the sun disturbs each frame.")

Geometry = Annotated[
    Path, cyclopts.Parameter(help=".npz with azimuth_deg and zenith_deg per native pixel")
]
Static = Annotated[
    Path,
    cyclopts.Parameter(help="skystatic run folder (daily masks, optics.json) or frozen NPZ"),
]
Latitude = Annotated[float, cyclopts.Parameter(help="Site latitude, degrees north")]
Longitude = Annotated[float, cyclopts.Parameter(help="Site longitude, degrees east")]


def _fail(exc: Exception) -> None:
    print(f"skysun: {exc}", file=sys.stderr)  # noqa: T201
    raise SystemExit(1) from exc


@app.command(name="run")
def run_command(
    frames: Annotated[list[Path], cyclopts.Parameter(help="Frame files or folders of frames")],
    *,
    geometry: Geometry,
    static: Static,
    latitude: Latitude,
    longitude: Longitude,
    output: Annotated[Path, cyclopts.Parameter(help="Where the products are written")],
    table: Annotated[
        Path | None,
        cyclopts.Parameter(help="Reflection table NPZ; default: the embedded GVCCS table"),
    ] = None,
    no_reflections: Annotated[
        bool, cyclopts.Parameter(help="Sun zone only (another camera without a table)")
    ] = False,
    allow_unreliable: Annotated[
        bool, cyclopts.Parameter(help="Accept daily masks whose skyline was not visible")
    ] = False,
) -> None:
    """Zone and reflections layers of every frame (resumes: frames already written are skipped)."""
    try:
        camera_geometry = CameraGeometry.from_file(geometry)
        source = StaticSource(static, allow_unreliable=allow_unreliable)
        reflection_table = None
        if not no_reflections:
            reflection_table = (
                packaged_table() if table is None else ReflectionTable.from_file(table)
            )
        camera = SunCamera(
            camera_geometry,
            Site(latitude_deg=latitude, longitude_deg=longitude),
            source.optics(camera_geometry.size),
            reflection_table,
        )
        result = run_frames(
            list_frames(frames),
            camera,
            lambda f: source.sky(f.time.date(), camera_geometry.size),
            output,
        )
    except (ValueError, KeyError, OSError) as exc:
        _fail(exc)
    lit = sum(1 for i in result.done if i.sun_elevation_deg > 0)
    print(f"{len(result.done)} frames, {lit} with the sun above the horizon -> {output}")  # noqa: T201
    for path in result.unreadable:
        print(f"skysun: unreadable frame skipped: {path}", file=sys.stderr)  # noqa: T201


@app.command(name="measure")
def measure_command(
    days: Annotated[list[Path], cyclopts.Parameter(help="Archive day folders .../YYYY/MM/DD")],
    *,
    geometry: Geometry,
    static: Static,
    latitude: Latitude,
    longitude: Longitude,
    labels: Annotated[Path, cyclopts.Parameter(help="NPZ of frame labels: ts, lab")],
    output: Annotated[Path, cyclopts.Parameter(help="One measurement NPZ per day")],
    exclude_day: Annotated[
        list[date] | None, cyclopts.Parameter(help="Days never to read (evaluation days)")
    ] = None,
    min_elevation: float = 3.0,
    workers: Annotated[int, cyclopts.Parameter(help="Worker processes")] = 1,
) -> None:
    """Measure the zone profile and the ring sectors of labelled archive frames, per day."""
    try:
        config = MeasureConfig(
            geometry=geometry,
            static=static,
            latitude=latitude,
            longitude=longitude,
            labels=labels,
            min_elevation=min_elevation,
            exclude_days=exclude_day or [],
        )
        done = measure_archive(days, config, output, workers)
    except (ValueError, KeyError, OSError) as exc:
        _fail(exc)
    print(f"{len(done)} days measured now, {sum(done.values())} frames -> {output}")  # noqa: T201


@app.command(name="build-table")
def build_command(
    measurements: Annotated[Path, cyclopts.Parameter(help="Folder written by skysun measure")],
    destination: Annotated[Path, cyclopts.Parameter(help="New table NPZ; never overwritten")],
    *,
    camera_id: str,
    width: int,
    height: int,
) -> None:
    """Build the reflection table (and its visibility ramp) from the measurements."""
    try:
        config = BuildConfig(camera_id=camera_id, command=" ".join(sys.argv))
        table = build_table(measurements, config, image_size=(width, height))
        table.save(destination)
    except (ValueError, KeyError, OSError) as exc:
        _fail(exc)
    meta = table.metadata
    print(  # noqa: T201
        f"{meta.frames} frames on {len(meta.days)} days, ramp tops "
        f"{[round(h, 3) for h in meta.visibility.high]}, glow ramp "
        f"{meta.visibility.glow_low:.3f}..{meta.visibility.glow_high:.3f}, reference contrast "
        f"{meta.reference_contrast:.4f}, sha256 {meta.sha256}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
