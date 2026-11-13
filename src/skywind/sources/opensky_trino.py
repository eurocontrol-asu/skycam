"""Raw Comm-B replies from the OpenSky Trino database (pyopensky, optional extra ``skywind[opensky]``).

Credentials come from pyopensky's own configuration (``~/.config/pyopensky/settings.conf``). If they are missing or
rejected the source raises :class:`SourceError` instead of starting an interactive browser login.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from skywind.sources.base import SourceError

__all__ = ["TrinoCommB"]

_COLUMNS = ["mintime", "rawmsg", "icao24", "serials"]


def _trino(cache_dir: Path) -> Any:
    os.environ.setdefault("OPENSKY_CACHE_NO_EXPIRE", "1")  # pyopensky otherwise purges its cache at import time
    try:
        import pyopensky.trino as trino_module
        from pyopensky.trino import Trino
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SourceError("pyopensky is not installed: uv sync --extra opensky") from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    trino_module.cache_path = cache_dir

    class AuthenticatedTrino(Trino):  # type: ignore[misc]
        def token(self, **kwargs: Any) -> Any:
            try:
                result = super().token(timeout=30, **kwargs)
            except Exception as exc:
                raise SourceError(f"OpenSky Trino authentication failed: {exc}") from exc
            if result is None:
                raise SourceError("OpenSky Trino credentials missing or rejected; no browser login is attempted")
            return result

    return AuthenticatedTrino()


class TrinoCommB:
    """DF20 / DF21 112-bit replies (first hex digit 'a'), with the receiver serials (space-joined), hour by hour."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def fetch(self, start: pd.Timestamp, stop: pd.Timestamp, icao24: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
        from pyopensky.schema import RollcallRepliesData4
        from sqlalchemy import literal_column

        api = _trino(self.cache_dir)
        serials = literal_column("array_join(transform(sensors, s -> cast(s.serial as varchar)), ' ')").label("serials")
        frames, chunks = [], []
        for h0 in pd.date_range(start.floor("1h"), stop, freq="1h"):
            a, b = max(h0, start), min(h0 + pd.Timedelta(hours=1), stop)
            if a >= b:
                continue
            tic = time.time()
            df = api.rawdata(a, b, RollcallRepliesData4.rawmsg.like("a%"), icao24=icao24, Table=RollcallRepliesData4,
                             extra_columns=(serials,))
            n = 0 if df is None else len(df)
            chunks.append({"start": str(a), "stop": str(b), "rows": n, "seconds": round(time.time() - tic, 1)})
            if n:
                frames.append(df[_COLUMNS])
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLUMNS)
        out = out[out["rawmsg"].str.len() == 28].drop_duplicates(["mintime", "rawmsg", "icao24"])
        return out, {"chunks": chunks}
