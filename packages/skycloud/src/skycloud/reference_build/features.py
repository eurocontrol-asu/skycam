"""Step 1, per-frame measurements (prototype ``a2_features.py``).

On the archive's grid directions (one native pixel per azimuth step x zenith step, the pixel
nearest to that direction), the means over a window of a-priori sky pixels of R, G, B, the
share of saturated pixels and the sd of G; and the sensor pedestal (mean and sd of the four
corners outside the fisheye circle). One file per day, the frames in the order given.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from skycloud.geometry import CameraGeometry, Site, to_vectors
from skycloud.images import pedestal, read_rgb, when_of
from skycloud.reference_build.params import BinParams
from skycloud.sun import sun_position

__all__ = [
    "DayFeatures",
    "FeatureGrid",
    "corner_sd",
    "frame_features",
    "grid_pixels",
    "load_day_features",
    "write_day_features",
]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]


def grid_pixels(geometry: CameraGeometry, p: BinParams) -> NDArray[np.int64]:
    """Flat index of the native pixel nearest to each grid direction (index
    ``iz + n_zenith_dirs * ia``), as the labelling kit's ``sky_auto.grid_pixels``."""
    az, zen = geometry.azimuth_deg, geometry.zenith_deg
    ok = np.isfinite(zen)
    tree = cKDTree(to_vectors(np.nan_to_num(az), np.nan_to_num(zen))[ok])
    n_az = round(360.0 / p.azimuth_step_deg)
    n_zen = round(90.0 / p.zenith_step_deg)
    ga = np.repeat(np.arange(n_az) * p.azimuth_step_deg, n_zen).astype(np.float64)
    gz = np.tile(np.arange(n_zen) * p.zenith_step_deg, n_az).astype(np.float64)
    return np.asarray(np.flatnonzero(ok.ravel())[tree.query(to_vectors(ga, gz))[1]], np.int64)


@dataclass(frozen=True)
class FeatureGrid:
    """The grid directions on the frame: their pixel, azimuth and zenith (the ``grid.npz`` of
    the reference archive) and the sky pixels of their windows."""

    gp: NDArray[np.int64]
    az: F32
    zen: F32
    idx: NDArray[np.int64]
    m: B
    n: F32

    @classmethod
    def build(cls, geometry: CameraGeometry, sky: B, p: BinParams) -> FeatureGrid:
        """``sky``: the a-priori static sky mask of the camera (True = sky)."""
        if sky.shape != geometry.shape:
            raise ValueError(f"sky mask {sky.shape} and geometry {geometry.shape} differ")
        gp = grid_pixels(geometry, p)
        h, w = sky.shape
        y, x = np.divmod(gp, w)
        off = np.arange(p.window_px) - p.window_px // 2
        yy = np.clip(y[:, None, None] + off[None, :, None], 0, h - 1)
        xx = np.clip(x[:, None, None] + off[None, None, :], 0, w - 1)
        idx = (yy * w + xx).reshape(len(y), p.window_px * p.window_px)
        m = np.asarray(sky).ravel()[idx]
        return cls(
            gp=gp,
            az=geometry.azimuth_deg.ravel()[gp].astype(np.float32),
            zen=geometry.zenith_deg.ravel()[gp].astype(np.float32),
            idx=idx,
            m=m,
            n=np.maximum(m.sum(1), 1).astype(np.float32),
        )

    @property
    def valid_n(self) -> NDArray[np.uint8]:
        """Sky pixels per window."""
        return np.asarray(self.m.sum(1).astype(np.uint8))

    def save(self, path: Path) -> None:
        """The archive's ``grid.npz`` (gp, az, zen)."""
        np.savez(path, gp=self.gp, az=self.az, zen=self.zen)


CORNER_PX = 60  # the corner squares of skycloud.images.pedestal (the renderer's pedestal)


def corner_sd(img: Any, k: int = CORNER_PX) -> F32:
    """The sd of each channel over the four corner squares of side ``k``."""
    c = np.concatenate(
        [img[:k, :k].reshape(-1, 3), img[:k, -k:].reshape(-1, 3),
         img[-k:, :k].reshape(-1, 3), img[-k:, -k:].reshape(-1, 3)]
    ).astype(np.float32)  # fmt: skip
    return np.asarray(c.std(0))


def frame_features(img: Any, grid: FeatureGrid, p: BinParams) -> F32:
    """(directions, 5) float32: window means of R, G, B over its sky pixels, saturated share,
    sd of G."""
    x = img.reshape(-1, 3)[grid.idx].astype(np.float32)
    m, n = grid.m, grid.n
    out = np.empty((x.shape[0], 5), np.float32)
    out[:, :3] = (x * m[..., None]).sum(1) / n[:, None]
    out[:, 3] = ((x.max(-1) >= p.saturation) & m).sum(1) / n
    g = x[..., 1]
    out[:, 4] = np.sqrt(np.maximum((g * g * m).sum(1) / n - out[:, 1] ** 2, 0))
    return out


@dataclass(frozen=True)
class DayFeatures:
    """One day's measurements (the prototype's ``feat/<day>.npz``)."""

    ts: NDArray[np.str_]
    sun: NDArray[np.float64]  # (frames, 2): sun azimuth and zenith angle, deg (stored float32)
    feat: F32  # (frames, directions, 5), stored float16
    corner: F32  # (frames, 3): pedestal R, G, B
    valid_n: NDArray[np.uint8]


def write_day_features(
    path: Path,
    frames: list[tuple[str, Path]],
    grid: FeatureGrid,
    site: Site,
    p: BinParams,
    on_error: Callable[[str, Exception], None] | None = None,
) -> int:
    """Measure ``frames`` (timestamp, file) and write the day file; unreadable frames are
    skipped (reported to ``on_error``). Returns the number of frames written."""
    rows = []
    for ts, f in frames:
        try:
            img = read_rgb(f)
        except Exception as exc:  # unreadable frame: skipped, as the prototype did
            if on_error is not None:
                on_error(ts, exc)
            continue
        rows.append((ts, frame_features(img, grid, p), pedestal(img), corner_sd(img)))
    if not rows:
        return 0
    stamps = [r[0] for r in rows]
    sun = np.array(
        [sun_position(when_of(t), site.latitude_deg, site.longitude_deg) for t in stamps],
        np.float32,
    )
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp, ts=np.array(stamps), sun=sun,
        feat=np.stack([r[1] for r in rows]).astype(np.float16),
        corner=np.stack([r[2] for r in rows]), corner_sd=np.stack([r[3] for r in rows]),
        valid_n=grid.valid_n,
    )  # fmt: skip
    tmp.replace(path)
    return len(rows)


def load_day_features(path: Path) -> DayFeatures:
    """A day file, at the precision the later steps read it (float32 features)."""
    with np.load(path) as d:
        return DayFeatures(
            ts=d["ts"],
            sun=d["sun"].astype(np.float64),
            feat=d["feat"].astype(np.float32),
            corner=d["corner"].astype(np.float32),
            valid_n=d["valid_n"],
        )
