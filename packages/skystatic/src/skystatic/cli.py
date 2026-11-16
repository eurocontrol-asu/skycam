"""Command line: ``skystatic run``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

import cyclopts

from skystatic.frames import open_source
from skystatic.geometry import SkyGeometry
from skystatic.pipeline import StaticConfig, run

__all__ = ["app", "main"]

app = cyclopts.App(name="skystatic", help="Static parts of a fixed all-sky camera.")


@app.command(name="run")
def run_command(
    days: Annotated[list[Path], cyclopts.Parameter(help="Day folders or zip archives to process")],
    *,
    zenith: Annotated[Path, cyclopts.Parameter(help=".npy/.npz zenith map of the camera")],
    output: Annotated[Path, cyclopts.Parameter(help="Where masks and report.json are written")],
    reference: Annotated[
        list[Path] | None,
        cyclopts.Parameter(
            help="Days defining the installation; default: the processed days",
            consume_multiple=True,
        ),
    ] = None,
    zenith_key: Annotated[str, cyclopts.Parameter(help="Array name inside a .npz")] = "zenith_deg",
    zenith_unit: Annotated[Literal["deg", "rad"], cyclopts.Parameter()] = "deg",
    pattern: Annotated[
        str | None, cyclopts.Parameter(help="File name pattern, e.g. '*_160.jpg'")
    ] = None,
    cache: Annotated[Path | None, cyclopts.Parameter(help="Cache of per-day averages")] = None,
    aggregate: Annotated[Literal["mean", "geometric"], cyclopts.Parameter()] = "geometric",
    parts: Annotated[int, cyclopts.Parameter(help="Parts of a day that must agree")] = 3,
    max_side: Annotated[int, cyclopts.Parameter(help="0 = native resolution")] = 1232,
    zenith_limit: Annotated[float, cyclopts.Parameter(help="Degrees")] = 90.0,
    edge: Annotated[float, cyclopts.Parameter(help="Edge strength threshold")] = 0.05,
    min_days: Annotated[float, cyclopts.Parameter(help="Fraction of reference days")] = 0.6,
    jobs: Annotated[int, cyclopts.Parameter(help="Days processed at the same time")] = 1,
) -> None:
    """Compute the installation mask and the variable static of each day."""
    try:
        config = StaticConfig(
            aggregate=aggregate,
            parts=parts,
            max_side=max_side,
            zenith_limit=zenith_limit,
            edge=edge,
            min_days=min_days,
            jobs=jobs,
        )
        geometry = SkyGeometry.from_file(zenith, zenith_key, zenith_unit)
        report = run(
            [open_source(day, pattern) for day in days],
            geometry,
            config,
            output,
            reference=[open_source(day, pattern) for day in reference or []],
            cache=cache,
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"skystatic: {exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
    print(f"installation: {100 * report.installation_fraction:.2f}% of the domain")  # noqa: T201
    for day in report.days:
        note = "" if day.skyline_visible else " [skyline not visible]"
        print(  # noqa: T201
            f"{day.label}: static {100 * day.static_fraction:.2f}%, "
            f"variable {100 * day.variable_fraction:.3f}% in {day.variable_components} "
            f"components ({day.frames_daytime} daytime frames){note}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
