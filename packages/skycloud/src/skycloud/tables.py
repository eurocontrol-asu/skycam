"""Reference tables of one camera: one frozen, checksummed, compressed archive.

The cloud map reads, for every pixel, what clear sky and overcast look like at that place and
time: medians and spreads of the two cues (colour index N, log brightness L) on a grid of 27 sun
elevations x 91 angles to the sun (2 deg) x 60 zenith angles (1.5 deg), trilinear in between
and continuous in sun elevation. The archive also holds the archive's 16,200 grid directions,
the measured low-count spread of the colour index, and the small threshold tables of the
method (JSON, kept verbatim). Arrays are stored at their original dtype: loading gives arrays
bit-identical to the files they were packed from. A SHA-256 of the whole archive is checked at
load (the embedded archive's digest is fixed in this module).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

__all__ = [
    "EMBEDDED_NAME",
    "EMBEDDED_SHA256",
    "ReferenceTables",
    "file_sha256",
    "pack_reference_tables",
]

F32 = NDArray[np.float32]

EMBEDDED_NAME = "gvccs_bretigny_c16.npz"
EMBEDDED_SHA256 = "a5babedcdff44802dc9d67001f18e465a270cf4d5f2b07b3bab15948b897b933"

#: The clear-sky / overcast tables the engine reads (c1_ref.npz; b4_smooth.py, c1_postfit.py).
REF_FIELDS = (
    "Nc", "No", "Lc", "Lo", "Nc_sd", "No_sd", "Lc_sd", "Lo_sd",
    "Nc_sda", "Lc_sda", "Nc_sdf", "Nc_bias",
)  # fmt: skip
#: The rebuilt spread tables read by the confidence only (c14_spread.npz; c14_spread.py).
SPREAD_FIELDS = ("Nc_sdf", "Nc_sd", "Lc_sda", "Lc_sd", "No_sd", "Lo_sd")
#: The JSON threshold files read at run time (kept verbatim).
JSON_FILES = (
    "c4_consistency", "c5_ledger_params", "c6_midscale", "c7_shape",
    "c8_release", "c9_knowable", "c11_lift_colour",
)  # fmt: skip


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class _Grid3:
    """One family of smooth tables with its elevation nodes and bin sizes."""

    tables: dict[str, F32]
    enodes: NDArray[np.float64]
    gb: float
    zb: float

    def lookup(self, names: tuple[str, ...], g: Any, z: Any, elev: float) -> list[F32]:
        """Tables ``names`` at angle to sun ``g``, zenith ``z`` (deg) and sun elevation."""
        x = float(np.interp(elev, self.enodes, np.arange(len(self.enodes))))
        e0 = min(int(np.floor(x)), len(self.enodes) - 2)
        t = x - e0
        co = np.stack(
            [np.asarray(g, np.float32) / self.gb - 0.5, np.asarray(z, np.float32) / self.zb - 0.5]
        )
        out = []
        for k in names:
            tab = self.tables[k]
            plane = (1 - t) * tab[e0] + t * tab[e0 + 1]
            out.append(np.asarray(ndimage.map_coordinates(plane, co, order=1, mode="nearest")))
        return out


def _monotone_spread(
    s: NDArray[np.float64], e: NDArray[np.float64], q: Any, s_max: float
) -> tuple[F32, F32]:
    """The measured low-count spread e(S): bins without data take the nearest measured bin
    below, made non-increasing in S, zero above ``s_max`` (sparse top bins)."""
    e = np.where(np.isfinite(q), e, np.nan)
    e[: np.argmax(np.isfinite(e))] = e[np.argmax(np.isfinite(e))]
    e = np.minimum.accumulate(np.nan_to_num(e))
    e[s > s_max] = 0.0
    return s.astype(np.float32), e.astype(np.float32)


@dataclass(frozen=True)
class ReferenceTables:
    """The reference tables of one camera and version, with their metadata."""

    ref: _Grid3
    spread: _Grid3
    grid_gp: NDArray[np.int64]
    grid_az: NDArray[np.float32]
    grid_zen: NDArray[np.float32]
    lowcount: tuple[F32, F32]
    lowcount_ped: tuple[F32, F32]
    thresholds: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""

    @classmethod
    def load(cls, path: Path | None = None, *, sha256: str | None = None) -> ReferenceTables:
        """The embedded tables (``path`` None) or an external archive. The archive's SHA-256
        must equal ``sha256`` (the embedded digest by default); an external archive without
        an expected digest is read after recording its digest."""
        if path is None:
            data = resources.files("skycloud").joinpath("reference", EMBEDDED_NAME).read_bytes()
            expected: str | None = EMBEDDED_SHA256 if sha256 is None else sha256
        else:
            data = Path(path).read_bytes()
            expected = sha256
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(
                f"reference tables: SHA-256 {digest} differs from the expected {expected}"
            )
        return cls.from_npz(np.load(io.BytesIO(data), allow_pickle=False), digest)

    @classmethod
    def from_npz(cls, d: Any, digest: str = "") -> ReferenceTables:
        """Build from an opened archive (``np.load`` result)."""
        meta = json.loads(str(d["__metadata__"]))

        def grid3(prefix: str, names: tuple[str, ...]) -> _Grid3:
            return _Grid3(
                {k: np.asarray(d[f"{prefix}/{k}"]).astype(np.float32) for k in names},
                np.asarray(d[f"{prefix}/enodes"]).astype(np.float64),
                float(d[f"{prefix}/GB"]),
                float(d[f"{prefix}/ZB"]),
            )

        lc = _monotone_spread(d["lowcount/S"], d["lowcount/e"], d["lowcount/q_step"], 300.0)
        lp = _monotone_spread(
            d["lowcount_ped/S"], d["lowcount_ped/e"], d["lowcount_ped/q_step"], 11.0
        )
        return cls(
            ref=grid3("ref", REF_FIELDS),
            spread=grid3("spread", SPREAD_FIELDS),
            grid_gp=np.asarray(d["grid/gp"]),
            grid_az=np.asarray(d["grid/az"]),
            grid_zen=np.asarray(d["grid/zen"]),
            lowcount=lc,
            lowcount_ped=lp,
            thresholds=meta["thresholds"],
            metadata={k: v for k, v in meta.items() if k != "thresholds"},
            sha256=digest,
        )

    def low_count_spread(self, s: Any) -> F32:
        """c3: extra clear-sky spread of the colour index at window count level S = B + R."""
        return np.asarray(np.interp(s, self.lowcount[0], self.lowcount[1]), np.float32)

    def low_count_spread_ped(self, x: Any) -> F32:
        """c4/c9: the same spread against the signal in pedestal units (gain-free)."""
        return np.asarray(np.interp(x, self.lowcount_ped[0], self.lowcount_ped[1]), np.float32)

    def table(self, name: str) -> Any:
        """One threshold table (the content of the JSON file of that name)."""
        return self.thresholds[name]


def pack_reference_tables(ref_dir: Path, out: Path, metadata: dict[str, Any]) -> str:
    """Merge the run-time files of ``ref_dir`` (c1_ref.npz, c14_spread.npz, grid.npz,
    c3_lowcount.npz, c4_lowcount_ped.npz and the JSON tables) into one compressed archive with
    ``metadata``; the JSON tables are kept verbatim. Returns the archive's SHA-256."""
    ref_dir = Path(ref_dir)
    arrays: dict[str, Any] = {}
    sources = {}
    for prefix, fname, names in (
        ("ref", "c1_ref.npz", (*REF_FIELDS, "enodes", "GB", "ZB")),
        ("spread", "c14_spread.npz", (*SPREAD_FIELDS, "enodes", "GB", "ZB")),
        ("grid", "grid.npz", ("gp", "az", "zen")),
        ("lowcount", "c3_lowcount.npz", ("S", "e", "q_step", "q_none")),
        ("lowcount_ped", "c4_lowcount_ped.npz", ("S", "e", "q_step", "q_none")),
    ):
        with np.load(ref_dir / fname) as d:
            for k in names:
                arrays[f"{prefix}/{k}"] = d[k]
        sources[fname] = file_sha256(ref_dir / fname)
    thresholds = {}
    for name in JSON_FILES:
        f = ref_dir / f"{name}.json"
        thresholds[name] = json.loads(f.read_text())
        sources[f.name] = file_sha256(f)
    meta = {**metadata, "sources_sha256": sources, "thresholds": thresholds}
    arrays["__metadata__"] = np.array(json.dumps(meta, sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return file_sha256(out)
