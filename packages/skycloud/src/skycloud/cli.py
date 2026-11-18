"""Command line: ``skycloud render``, ``skycloud ledger`` and ``skycloud build-reference``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
import numpy as np
from numpy.typing import NDArray

from skycloud.geometry import CameraGeometry, Site
from skycloud.ledger_build import build_ledgers
from skycloud.reference_build import (
    BuildInputs,
    BuildParams,
    CarriedTables,
    build_reference,
    select_frames,
)
from skycloud.render import RenderSetup, render
from skycloud.tables import ReferenceTables

__all__ = ["app", "main"]

app = cyclopts.App(name="skycloud", help="Continuous cloud map of a fixed all-sky camera.")


def _setup(
    *,
    images: Path,
    mask_dir: Path,
    ledger_dir: Path,
    azimuth: Path,
    zenith: Path,
    latitude: float,
    longitude: float,
    azimuth_key: str,
    zenith_key: str,
    unit: Literal["deg", "rad"],
    tables: Path | None,
    tables_sha256: str | None,
    exclude: Path | None,
    allow_unreliable_mask: bool,
    workers: int,
) -> RenderSetup:
    geometry = CameraGeometry.from_files(
        azimuth, zenith, azimuth_key=azimuth_key, zenith_key=zenith_key, unit=unit
    )
    return RenderSetup(
        geometry=geometry,
        site=Site(latitude_deg=latitude, longitude_deg=longitude),
        images=images,
        mask_dir=mask_dir,
        ledger_dir=ledger_dir,
        tables=ReferenceTables.load(tables, sha256=tables_sha256),
        exclude=frozenset(exclude.read_text().split()) if exclude else frozenset(),
        allow_unreliable_mask=allow_unreliable_mask,
        workers=workers,
    )


@app.command(name="render")
def render_command(
    stamps: Annotated[
        list[str] | None, cyclopts.Parameter(help="Timestamps YYYYmmddHHMMSS")
    ] = None,
    *,
    out: Annotated[Path, cyclopts.Parameter(help="Output folder (products, frames.json)")],
    images: Annotated[Path, cyclopts.Parameter(help="Frames: ROOT/YYYY/MM/DD/image_<ts>.jp2")],
    mask_dir: Annotated[Path, cyclopts.Parameter(help="skystatic daily masks (classes_<day>.png)")],
    ledger_dir: Annotated[Path, cyclopts.Parameter(help="Cache of the day gain ledgers")],
    azimuth: Annotated[Path, cyclopts.Parameter(help=".npy/.npz azimuth map")],
    zenith: Annotated[Path, cyclopts.Parameter(help=".npy/.npz zenith map")],
    latitude: Annotated[float, cyclopts.Parameter(help="Camera latitude, deg")],
    longitude: Annotated[float, cyclopts.Parameter(help="Camera longitude, deg east")],
    list_file: Annotated[
        Path | None, cyclopts.Parameter(name="--list", help="File of timestamps")
    ] = None,
    azimuth_key: Annotated[
        str, cyclopts.Parameter(help="Array name inside a .npz")
    ] = "azimuth_deg",
    zenith_key: Annotated[str, cyclopts.Parameter(help="Array name inside a .npz")] = "zenith_deg",
    unit: Annotated[Literal["deg", "rad"], cyclopts.Parameter()] = "deg",
    tables: Annotated[Path | None, cyclopts.Parameter(help="External reference archive")] = None,
    tables_sha256: Annotated[str | None, cyclopts.Parameter(help="Its expected SHA-256")] = None,
    exclude: Annotated[
        Path | None, cyclopts.Parameter(help="File of timestamps never read")
    ] = None,
    allow_unreliable_mask: Annotated[
        bool, cyclopts.Parameter(help="Accept days whose skyline was not visible")
    ] = False,
    workers: Annotated[int, cyclopts.Parameter(help="Worker processes (about 3.5 GB each)")] = 1,
) -> None:
    """Render frames with their temporal context into OUT."""
    try:
        targets = list(stamps or []) + (list_file.read_text().split() if list_file else [])
        setup = _setup(
            images=images, mask_dir=mask_dir, ledger_dir=ledger_dir, azimuth=azimuth, zenith=zenith,
            latitude=latitude, longitude=longitude, azimuth_key=azimuth_key, zenith_key=zenith_key,
            unit=unit, tables=tables, tables_sha256=tables_sha256, exclude=exclude,
            allow_unreliable_mask=allow_unreliable_mask, workers=workers,
        )  # fmt: skip
        report = render(targets, setup, out)
    except (ValueError, KeyError, OSError) as exc:
        print(f"skycloud: {exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
    errors = sum("error" in r for r in report.frames)
    print(f"rendered {len(report.frames)} frames into {out} ({errors} errors)")  # noqa: T201


@app.command(name="ledger")
def ledger_command(
    days: Annotated[list[str], cyclopts.Parameter(help="Days YYYYmmdd")],
    *,
    images: Annotated[Path, cyclopts.Parameter(help="Frames: ROOT/YYYY/MM/DD/image_<ts>.jp2")],
    ledger_dir: Annotated[Path, cyclopts.Parameter(help="Cache of the day gain ledgers")],
    azimuth: Annotated[Path, cyclopts.Parameter(help=".npy/.npz azimuth map")],
    zenith: Annotated[Path, cyclopts.Parameter(help=".npy/.npz zenith map")],
    latitude: Annotated[float, cyclopts.Parameter(help="Camera latitude, deg")],
    longitude: Annotated[float, cyclopts.Parameter(help="Camera longitude, deg east")],
    azimuth_key: Annotated[
        str, cyclopts.Parameter(help="Array name inside a .npz")
    ] = "azimuth_deg",
    zenith_key: Annotated[str, cyclopts.Parameter(help="Array name inside a .npz")] = "zenith_deg",
    unit: Annotated[Literal["deg", "rad"], cyclopts.Parameter()] = "deg",
    tables: Annotated[Path | None, cyclopts.Parameter(help="External reference archive")] = None,
    tables_sha256: Annotated[str | None, cyclopts.Parameter(help="Its expected SHA-256")] = None,
    exclude: Annotated[
        Path | None, cyclopts.Parameter(help="File of timestamps never read")
    ] = None,
    workers: Annotated[int, cyclopts.Parameter(help="Worker processes")] = 1,
    overwrite: Annotated[bool, cyclopts.Parameter(help="Rebuild cached days")] = False,
) -> None:
    """Build the gain ledgers (c10 and legacy) of DAYS into LEDGER_DIR."""
    try:
        setup = _setup(
            images=images, mask_dir=Path(), ledger_dir=ledger_dir, azimuth=azimuth, zenith=zenith,
            latitude=latitude, longitude=longitude, azimuth_key=azimuth_key, zenith_key=zenith_key,
            unit=unit, tables=tables, tables_sha256=tables_sha256, exclude=exclude,
            allow_unreliable_mask=False, workers=workers,
        )  # fmt: skip
        written = build_ledgers(days, setup, overwrite=overwrite)
    except (ValueError, KeyError, OSError) as exc:
        print(f"skycloud: {exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
    print(f"{len(written)} ledger file(s) written into {ledger_dir}")  # noqa: T201


def _sky_mask(path: Path, sky_class: int) -> NDArray[np.bool_]:
    """The a-priori static sky mask: pixels whose value is ``sky_class``."""
    if path.suffix == ".npy":
        values = np.load(path)
    else:
        from PIL import Image

        with Image.open(path) as im:
            values = np.asarray(im)
    mask: NDArray[np.bool_] = np.asarray(values == sky_class, dtype=bool)
    return mask


def _days(path: Path | None) -> list[str]:
    return path.read_text().split() if path else []


@app.command(name="build-reference")
def build_reference_command(
    *,
    out: Annotated[Path, cyclopts.Parameter(help="Reference archive to write (.npz)")],
    work: Annotated[Path, cyclopts.Parameter(help="Work folder (intermediates; a build resumes)")],
    camera: Annotated[str, cyclopts.Parameter(help="Camera name recorded in the archive")],
    sky_mask: Annotated[
        Path, cyclopts.Parameter(help="A-priori static mask of the camera (.png/.npy)")
    ],
    azimuth: Annotated[Path, cyclopts.Parameter(help=".npy/.npz azimuth map")],
    zenith: Annotated[Path, cyclopts.Parameter(help=".npy/.npz zenith map")],
    latitude: Annotated[float, cyclopts.Parameter(help="Camera latitude, deg")],
    longitude: Annotated[float, cyclopts.Parameter(help="Camera longitude, deg east")],
    images: Annotated[
        Path | None, cyclopts.Parameter(help="Archive folder (every image_*.jp2 below)")
    ] = None,
    list_file: Annotated[
        Path | None,
        cyclopts.Parameter(name="--list", help="Frame list (paths, or timestamps under --images)"),
    ] = None,
    exclude_days: Annotated[
        Path | None, cyclopts.Parameter(help="File of days YYYYmmdd never read (evaluation days)")
    ] = None,
    calibration_list: Annotated[
        Path | None,
        cyclopts.Parameter(help="Frames of dense 30 s days for the low-count tables"),
    ] = None,
    carry: Annotated[
        Path | None, cyclopts.Parameter(help="Archive the unmeasured tables are carried from")
    ] = None,
    carry_sha256: Annotated[str | None, cyclopts.Parameter(help="Its expected SHA-256")] = None,
    params: Annotated[
        Path | None, cyclopts.Parameter(help="JSON of BuildParams overrides (default: c16)")
    ] = None,
    sky_class: Annotated[int, cyclopts.Parameter(help="Value of the sky in the mask")] = 0,
    azimuth_key: Annotated[
        str, cyclopts.Parameter(help="Array name inside a .npz")
    ] = "azimuth_deg",
    zenith_key: Annotated[str, cyclopts.Parameter(help="Array name inside a .npz")] = "zenith_deg",
    unit: Annotated[Literal["deg", "rad"], cyclopts.Parameter()] = "deg",
    until: Annotated[str | None, cyclopts.Parameter(help="Stop after this step")] = None,
    keep_duplicates: Annotated[
        bool, cyclopts.Parameter(help="Read a timestamp listed twice twice (the c16 build)")
    ] = False,
    workers: Annotated[int, cyclopts.Parameter(help="Worker processes of the measurements")] = 1,
) -> None:
    """Build the reference tables of a camera from an archive of its frames into OUT."""
    try:
        prm = BuildParams.model_validate_json(params.read_text()) if params else BuildParams()
        if keep_duplicates:
            prm = prm.model_copy(update={"duplicates": "keep"})
        excluded = _days(exclude_days)
        frames = select_frames(
            images=images, list_file=list_file, exclude_days=excluded, duplicates=prm.duplicates
        )
        calib = None
        if calibration_list is not None:
            calib = select_frames(
                images=images, list_file=calibration_list, exclude_days=excluded,
                duplicates=prm.duplicates,
            )  # fmt: skip
        if carry is not None and carry_sha256 is None:
            raise ValueError("--carry needs --carry-sha256")
        inputs = BuildInputs(
            camera=camera,
            geometry=CameraGeometry.from_files(
                azimuth, zenith, azimuth_key=azimuth_key, zenith_key=zenith_key, unit=unit
            ),
            sky_mask=_sky_mask(sky_mask, sky_class),
            site=Site(latitude_deg=latitude, longitude_deg=longitude),
            frames=frames,
            calibration_frames=calib,
            carried=CarriedTables.from_archive(carry, carry_sha256)
            if carry and carry_sha256
            else None,
        )
        report = build_reference(
            inputs, work, out, prm, until=until, workers=workers,
            log=lambda s: print(s, file=sys.stderr, flush=True),  # noqa: T201
        )  # fmt: skip
    except (ValueError, KeyError, OSError) as exc:
        print(f"skycloud: {exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
    if report.sha256:
        print(f"{out}: SHA-256 {report.sha256}")  # noqa: T201
    else:
        print(f"stopped after step {until} in {work}")  # noqa: T201


def main() -> None:
    """Entry point of the ``skycloud`` command."""
    app()
