"""Files copied from another host over ssh (e.g. the state-vector and ERA5 archives of a data server)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from skywind.sources.base import SourceError

__all__ = ["RemoteCopySource"]


@dataclass(frozen=True)
class RemoteCopySource:
    """``pattern`` is a remote path with ``{Y}``, ``{m}``, ``{d}`` fields, e.g. ``/archive/{Y}/{Y}-{m}-{d}.parquet``."""

    host: str
    pattern: str
    timeout_s: float = 3600.0

    def remote_path(self, day: str) -> str:
        return self.pattern.format(Y=day[:4], m=day[4:6], d=day[6:])

    def fetch(self, day: str, dest: Path) -> Path:
        scp = shutil.which("scp")
        if scp is None:
            raise SourceError("scp not found")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        cmd = [scp, "-q", "-o", "BatchMode=yes", f"{self.host}:{self.remote_path(day)}", str(tmp)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, check=False)  # noqa: S603
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise SourceError(f"{self.host}:{self.remote_path(day)}: {res.stderr.strip() or 'copy failed'}")
        tmp.replace(dest)
        return dest
