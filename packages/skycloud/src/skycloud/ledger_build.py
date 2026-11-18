"""Build and cache the day ledgers of a render (both the c10 and the legacy form)."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skycloud.images import frame_path, pedestal, read_rgb
from skycloud.ledger import LedgerSet, build_day_ledger, sky8, small_frame, step_delta

if TYPE_CHECKING:
    from skycloud.render import RenderSetup

__all__ = ["build_ledgers", "day_frames", "ensure_ledgers"]

_S: dict[str, Any] = {}


def day_frames(setup: RenderSetup, day: str) -> list[str]:
    """The day's frames (time order), excluded ones dropped, sun above -4 deg."""
    folder = frame_path(setup.images, day + "000000").parent
    ts = sorted(f.stem[6:20] for f in folder.glob("image_*.jp2"))
    return [t for t in ts if t not in setup.exclude and setup.elev(t) > -4.0]


def _init(root: Path, sky: Any) -> None:
    _S["root"], _S["sky"] = root, sky


def _ped(ts: str) -> tuple[str, float | None]:
    try:
        return ts, float(pedestal(read_rgb(frame_path(_S["root"], ts)))[0])
    except Exception:  # unreadable frame: no pedestal
        return ts, None


def _delta(args: tuple[str, str, dict[str, Any], bool]) -> tuple[str, str, Any, float, Any]:
    t0, t1, prm, c10 = args
    try:
        f0 = small_frame(read_rgb(frame_path(_S["root"], t0)), _S["sky"])
        f1 = small_frame(read_rgb(frame_path(_S["root"], t1)), _S["sky"])
        d, sup, var = step_delta(f0, f1, prm, c10)
        return t0, t1, d, sup, var
    except Exception:  # unreadable pair: no step
        return t0, t1, None, 0.0, None


def build_ledgers(
    days: Iterable[str], setup: RenderSetup, *, overwrite: bool = False
) -> list[Path]:
    """Build the c10 and legacy ledgers of ``days`` into ``setup.ledger_dir``."""
    sky = sky8(setup.geometry.zen99)
    prm = setup.tables.table("c5_ledger_params")
    p = setup.params
    written = []
    with ProcessPoolExecutor(
        max(setup.workers, 1), initializer=_init, initargs=(setup.images, sky)
    ) as pool:
        for day in days:
            for legacy in (False, True):
                path = LedgerSet.path(setup.ledger_dir, day, legacy)
                if path.exists() and not overwrite:
                    continue
                c10 = not legacy
                ledger = build_day_ledger(
                    day_frames(setup, day),
                    lambda ts: dict(pool.map(_ped, ts, chunksize=16)),
                    lambda steps, c10=c10: pool.map(_delta, [(a, b, prm, c10) for a, b in steps]),  # type: ignore[misc]
                    setup.elev,
                    prm,
                    c10=c10,
                    e_period=p.e_period,
                    shape8=sky.shape,
                )
                ledger.save(path)
                written.append(path)
    return written


def ensure_ledgers(days: Iterable[str], setup: RenderSetup) -> list[Path]:
    """Build the ledgers of the days that have none in the cache."""
    missing = [
        d
        for d in days
        if not all(LedgerSet.path(setup.ledger_dir, d, lg).exists() for lg in (False, True))
    ]
    return build_ledgers(missing, setup) if missing else []
