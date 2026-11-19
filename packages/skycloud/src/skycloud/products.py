"""The products of a render and their encoding (one folder per run).

| File | Content |
|---|---|
| ``<ts>.png`` | value, uint8: code / 254 (0 clear .. 254 thick cloud); 255 = not sky (*) |
| ``_unknown/<ts>.png`` | 255 = the unknown verdict (final confidence under the contract) |
| ``<ts>_conf.png`` | published confidence, code / 255 |
| ``<ts>_lens.png`` | 255 = lens ghost found by the detector (the value there is left as measured) |
| ``_unmasked/<ts>.png`` | the value before the unknown rule (255: no value at all) |
| ``_known/<ts>.png`` | diffuse knowability K x 255, below E_K |
| ``_cert/<ts>.png`` | certified structure below +10 deg: 255 x (0.9-1.0, 0.8 field, 0.5 line) |
| ``_region/<ts>.png`` | 255 = diffuse-known region, below E_K |
| ``_rings/<ts>.png`` | 255 = coherent lens-ring field (side layer, not a flag) |
| ``_capped/``, ``_uncapped/`` | the confidence after / before the consistency cap |
| ``_neighbours/`` | the +-30 s neighbours rendered for the temporal passes |
| ``states.json``, ``frames.json`` | frame states (levels) and per-frame diagnostics |

(*) with ``unknown_state="layer"`` (default) a low confidence never removes the value: the
verdict is the complementary mask ``_unknown/``. With ``"mask"`` (version c16) the verdict is
written into the value as 255 and ``_unknown/`` is not written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skycloud.engine import FrameResult

__all__ = ["FrameProducts", "load_products", "value_code", "write_frame"]

U8 = NDArray[np.uint8]


def value_code(alpha: Any) -> U8:
    """Value in [0, 1] (NaN = no value) to its uint8 code."""
    out: U8 = np.rint(254.0 * np.nan_to_num(alpha)).astype(np.uint8)
    out[~np.isfinite(alpha)] = 255
    return out


def _layer(x: Any) -> U8:
    out: U8 = np.rint(255.0 * np.clip(x.astype(np.float32), 0, 1)).astype(np.uint8)
    return out


def write_frame(out: Path, ts: str, res: FrameResult) -> None:
    """Write one rendered frame (value, confidence, lens flag and the side layers)."""
    out.mkdir(parents=True, exist_ok=True)
    if res.lens is not None:
        Image.fromarray(np.where(res.lens, 255, 0).astype(np.uint8)).save(out / f"{ts}_lens.png")
    Image.fromarray(value_code(res.alpha)).save(out / f"{ts}.png")
    for layer, sub in ((res.known, "_known"), (res.cert, "_cert")):
        if layer is not None:
            (out / sub).mkdir(exist_ok=True)
            Image.fromarray(_layer(layer)).save(out / sub / f"{ts}.png")
    if res.rings is not None:
        (out / "_rings").mkdir(exist_ok=True)
        Image.fromarray(np.where(res.rings, 255, 0).astype(np.uint8)).save(
            out / "_rings" / f"{ts}.png"
        )
    Image.fromarray(np.rint(255 * res.conf).astype(np.uint8)).save(out / f"{ts}_conf.png")


@dataclass(frozen=True)
class FrameProducts:
    """The delivered products of one frame, decoded: value (float32, NaN = unknown or not
    sky), unmasked value (NaN = not sky), confidence (float32) and lens flag (bool)."""

    value: NDArray[np.float32]
    unmasked: NDArray[np.float32]
    confidence: NDArray[np.float32]
    lens: NDArray[np.bool_]


def _decode(code: U8) -> NDArray[np.float32]:
    return np.where(code == 255, np.nan, code / 254.0).astype(np.float32)


def load_products(out: Path, ts: str) -> FrameProducts:
    """Read the delivered products of one target frame of a render folder."""
    out = Path(out)
    v = np.asarray(Image.open(out / f"{ts}.png"))
    um = out / "_unmasked" / f"{ts}.png"
    u = np.asarray(Image.open(um)) if um.exists() else v
    c = np.asarray(Image.open(out / f"{ts}_conf.png")).astype(np.float32) / 255.0
    lp = out / f"{ts}_lens.png"
    lens = np.asarray(Image.open(lp)) > 0 if lp.exists() else np.zeros(v.shape, bool)
    return FrameProducts(_decode(v), _decode(u), c, lens)
