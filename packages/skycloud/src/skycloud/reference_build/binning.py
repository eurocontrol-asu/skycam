"""Step 2, per-frame bin means (prototype ``b1_binmeans.py``).

Per frame, the means of the colour index N, the log brightness L and the relative texture
sd(G) / (G - pedestal G) over the directions of each (angle to the sun, zenith) bin; a bin with
fewer than ``bin_min_dirs`` readable directions is NaN. Stored float16, one row per frame,
in a folder of ``.npy`` files that later steps memory-map (26,028 frames x 5,460 bins x 3
cues = 853 MB for the c16 archive: never loaded whole).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from skycloud.cues import cues
from skycloud.geometry import angle_to_sun
from skycloud.reference_build.features import DayFeatures
from skycloud.reference_build.params import BinParams

__all__ = ["BIN_FIELDS", "BinMeans", "bin_keys", "day_bin_means", "frame_ok", "write_bin_means"]

BIN_FIELDS = ("ts", "sun", "corner", "MN", "ML", "MT")


def frame_ok(feat: Any, valid_n: Any, zen: Any, p: BinParams) -> Any:
    """The readable directions of a frame: enough sky in the window, inside the rim, no
    saturated pixel, no window-mean channel near saturation."""
    ok0 = (valid_n >= p.window_min_sky) & (zen < p.zenith_max)
    return ok0 & (feat[:, 3] == 0) & (feat[:, :3].max(1) < p.bright_max)


def bin_keys(g: Any, zi: Any, p: BinParams) -> Any:
    """Flat bin index (angle-to-sun bin x n_zenith + zenith bin)."""
    return np.clip((g // p.gamma_bin_deg).astype(int), 0, p.n_gamma - 1) * p.n_zenith + zi


def day_bin_means(day: DayFeatures, az: Any, zen: Any, p: BinParams) -> tuple[Any, Any, Any]:
    """(MN, ML, MT) float16 (frames, bins) of one day. ``az``, ``zen``: grid directions."""
    nb = p.n_gamma * p.n_zenith
    zi = np.clip((zen // p.zenith_bin_deg).astype(int), 0, p.n_zenith - 1)
    mn = np.full((len(day.ts), nb), np.nan, np.float32)
    ml, mt = mn.copy(), mn.copy()
    for i in range(len(day.ts)):
        f, cor = day.feat[i], day.corner[i]
        m = frame_ok(f, day.valid_n, zen, p)
        g = angle_to_sun(az, zen, day.sun[i, 0], day.sun[i, 1])
        key = bin_keys(g[m], zi[m], p)
        n_cue, l_cue = cues(f[m, 0], f[m, 1], f[m, 2], cor)
        n = np.bincount(key, minlength=nb)
        tex = f[m, 4] / np.maximum(f[m, 1] - cor[1], 1.0)
        for out, v in ((mn, n_cue), (ml, l_cue), (mt, tex)):
            s = np.bincount(key, v.astype(np.float64), nb)
            out[i] = np.where(n >= p.bin_min_dirs, s / np.maximum(n, 1), np.nan)
    return mn.astype(np.float16), ml.astype(np.float16), mt.astype(np.float16)


def write_bin_means(
    folder: Path, days: Iterable[DayFeatures], n_frames: int, az: Any, zen: Any, p: BinParams
) -> int:
    """Stream the days' bin means into ``folder`` (one ``.npy`` per field). Returns the
    number of frames written."""
    from numpy.lib.format import open_memmap

    folder.mkdir(parents=True, exist_ok=True)
    nb = p.n_gamma * p.n_zenith
    arrays: dict[str, Any] = {
        k: open_memmap(folder / f"{k}.tmp.npy", "w+", np.float16, (n_frames, nb))
        for k in ("MN", "ML", "MT")
    }
    ts: list[str] = []
    sun, corner = [], []
    k = 0
    for day in days:
        mn, ml, mt = day_bin_means(day, az, zen, p)
        j = k + len(day.ts)
        arrays["MN"][k:j], arrays["ML"][k:j], arrays["MT"][k:j] = mn, ml, mt
        ts.extend(str(t) for t in day.ts)
        sun.append(day.sun)
        corner.append(day.corner)
        k = j
    if k != n_frames:
        raise ValueError(f"bin means: {k} frames read, {n_frames} expected")
    for name, a in arrays.items():
        a.flush()
        del a
        (folder / f"{name}.tmp.npy").replace(folder / f"{name}.npy")
    np.save(folder / "ts.npy", np.array(ts))
    np.save(folder / "sun.npy", np.concatenate(sun).astype(np.float64))
    np.save(folder / "corner.npy", np.concatenate(corner).astype(np.float32))
    return k


@dataclass(frozen=True)
class BinMeans:
    """The archive's bin means, memory-mapped (``MN``, ``ML``, ``MT`` float16)."""

    ts: NDArray[np.str_]
    sun: NDArray[np.float64]
    corner: NDArray[np.float32]
    MN: Any
    ML: Any
    MT: Any

    @classmethod
    def open(cls, folder: Path) -> BinMeans:
        """Memory-map a folder written by ``write_bin_means``."""
        load = {k: np.load(folder / f"{k}.npy", mmap_mode="r") for k in ("MN", "ML", "MT")}
        return cls(
            ts=np.load(folder / "ts.npy"),
            sun=np.load(folder / "sun.npy"),
            corner=np.load(folder / "corner.npy"),
            **load,
        )

    @classmethod
    def from_arrays(cls, d: Any) -> BinMeans:
        """From a mapping of arrays (the prototype's ``binmeans_feat.npz``, or a test)."""
        return cls(**{k: d[k] for k in BIN_FIELDS})

    @property
    def elev(self) -> NDArray[np.float64]:
        """Sun elevation of every frame, deg."""
        return np.asarray(90 - self.sun[:, 1])
