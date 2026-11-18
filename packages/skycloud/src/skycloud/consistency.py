"""Temporal consistency (method stage "consistency cap", round c4/c5) and the unchanged-sky test.

A value that is not reproduced 30 s later on an unchanged sky is not known. For a frame and
each +-30 s neighbour: where the sky did not change between them (normalised brightness and
ledger-corrected colour index within 3 noise sd, at 1/8 resolution, over the whole 2-sigma
neighbourhood) but the smoothed value differs by more than D0 (99th percentile of that
difference on development-day pairs without a gain step, ref ``c4_consistency``), the
confidence is capped at 0.29 on those pixels. The pass works on the rendered products
(``<out>/<ts>.png``, ``<ts>_conf.png``, the neighbours in ``<out>/_neighbours/``) and keeps the
uncapped confidence in ``<out>/_uncapped/``; it can be re-run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skycloud.images import frame_path, read_rgb, shift_stamp
from skycloud.ledger import LedgerSet, SmallFrame, small_frame

__all__ = ["SkyChange", "apply_cap"]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]
GROW: Any = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)  # radius 4 cells at 1/4 res = 2 sigma


@dataclass
class SkyChange:
    """Reads frames at 1/8 resolution and tells where the sky did not change between two."""

    root: Path
    sky8: B
    ledgers: LedgerSet
    elev: Any  # Callable[[str], float]
    r_step: float
    n_noise: float

    def small(self, ts: str) -> SmallFrame:
        return small_frame(read_rgb(frame_path(self.root, ts)), self.sky8)

    def same(self, t: str, n: str, shape: tuple[int, int]) -> B:
        """The sky did not change between ``t`` and ``n`` where BOTH its normalised brightness
        and its ledger-corrected colour index changed by less than 3 x their noise on
        unchanged sky (R_STEP, 3 N_NOISE); at ``shape`` (nearest)."""
        a, b = self.small(t), self.small(n)
        n0, n1 = a.n, b.n
        o0, _ = self.ledgers.offset(t, self.elev(t))
        o1, _ = self.ledgers.offset(n, self.elev(n))
        if o0 is not None:
            n0 = n0 + o0
        if o1 is not None:
            n1 = n1 + o1
        rel = (b.g - a.g) / np.maximum((b.g + a.g) / 2.0, 1e-3)
        same = a.sky & b.sky & (np.abs(rel) < self.r_step) & (np.abs(n1 - n0) < 3.0 * self.n_noise)
        out = cv2.resize(
            same.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return np.asarray(out > 0)


def _load_val(d: Path, ts: str) -> F32 | None:
    p = Path(d) / f"{ts}.png"
    if not p.exists():
        return None
    v = np.asarray(Image.open(p))
    h, w = v.shape
    v4 = np.where(v == 255, np.nan, v / 254.0).astype(np.float32)
    return np.asarray(cv2.resize(v4, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST))


def _smooth_d(d: Any, ok: B, sig: float) -> Any:
    m = ok.astype(np.float32)
    num = cv2.GaussianBlur(np.where(ok, d, 0.0).astype(np.float32), (0, 0), sig)
    return num / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 1e-3)


def _cap_mask(ok: B, same: B, d: Any, d0: float, sig: float) -> B:
    """The cap where the sky is unchanged over the whole 2-sigma neighbourhood and the
    smoothed value differs."""
    erode: Any = cv2.erode  # the stubs do not type a scalar border value
    eroded = erode(
        (ok & same).astype(np.uint8), GROW, borderType=cv2.BORDER_CONSTANT, borderValue=1
    )
    same = (eroded > 0) & ok
    core = same & (_smooth_d(d, ok, sig) > d0)
    return np.asarray((cv2.dilate(core.astype(np.uint8), GROW) > 0) & same)


def _cap_one(
    out: Path, t: str, dirs: list[Path], change: SkyChange, d0: float, sig: float
) -> tuple[B, list[str], F32]:
    vt = _load_val(out, t)
    assert vt is not None  # noqa: S101  # the caller checked
    capmask = np.zeros(vt.shape, bool)
    used = []
    for s in (-30, 30):
        n = shift_stamp(t, s)
        vn = next((v for v in (_load_val(dd, n) for dd in dirs) if v is not None), None)
        if (
            vn is None
            or not frame_path(change.root, n).exists()
            or not frame_path(change.root, t).exists()
        ):
            continue
        ok = np.isfinite(vt) & np.isfinite(vn)
        d = np.abs(vt - vn)
        capmask |= _cap_mask(ok, change.same(t, n, ok.shape), d, d0, sig)
        used.append(n)
    return capmask, used, vt


def apply_cap(
    out: Path, targets: Iterable[str], change: SkyChange, d0: float, cap: float, sig: float
) -> dict[str, dict[str, Any]]:
    """Cap the confidence of every target where a +-30 s neighbour (a target, or a frame
    rendered in ``out/_neighbours``) shows an unchanged sky but a different value."""
    out = Path(out)
    dirs = [out, out / "_neighbours"]
    info: dict[str, dict[str, Any]] = {}
    for t in targets:
        if _load_val(out, t) is None:
            continue
        capmask, used, vt = _cap_one(out, t, dirs, change, d0, sig)
        cp = out / f"{t}_conf.png"
        bk = out / "_uncapped" / f"{t}_conf.png"  # the confidence before the check (re-runnable)
        if not bk.exists():
            bk.parent.mkdir(exist_ok=True)
            bk.write_bytes(cp.read_bytes())
        else:
            cp.write_bytes(bk.read_bytes())
        if used and capmask.any():
            c = np.asarray(Image.open(cp)).astype(np.float32) / 255.0
            h, w = c.shape
            cm = cv2.resize(capmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            c = np.where(cm, np.minimum(c, cap), c)
            Image.fromarray(np.rint(255 * c).astype(np.uint8)).save(cp)
        valid = np.isfinite(vt)
        share = round(float(capmask[valid].mean()), 4) if used and valid.any() else None
        info[t] = {"neighbours": len(used), "capped_share": share}
        if not used:
            info[t]["note"] = (
                "no +-30 s neighbour image: confidence not checked for temporal consistency"
            )
    return info
