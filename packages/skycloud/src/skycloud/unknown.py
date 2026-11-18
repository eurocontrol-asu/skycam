"""The explicit "unknown" state (method stage "unknown state", rounds c6-c16).

A pixel whose final confidence is below 0.30 gets NO VALUE (code 255) in ``<ts>.png``; the
full value is kept in ``_unmasked/``. The published confidence is read with the +-30 s
neighbours on unchanged sky (median of the three frames before the cap; a capped pixel keeps
its cap). In daytime (no ``_known/`` layer) the boundary is the confidence smoothed at the fill
kernel's scale, upsampled bilinearly, plus the strict per-pixel contract. Below E_K the
DIFFUSE sky is decided as a region (table knowability K at least 0.45 and the confidence at the
diffuse scale at the contract, made well-formed and monotone through a twilight), certified
structures that persist in a neighbour (lines, fields, bodies) are delivered inside it, frame
factors (ambiguity, contradiction, overcast rim, gap-free deck) lower the confidence of the
uncertified sky, and small unresolved fragments are removed.

The pass works on the rendered products of an output folder and is idempotent (it starts
from ``_unmasked/`` and ``_capped/``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skycloud.images import shift_stamp
from skycloud.params import CloudMapParams

__all__ = ["FrameFactors", "UnknownContext", "apply_unknown", "temporal_conf"]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]
U8 = NDArray[np.uint8]


@dataclass(frozen=True)
class FrameFactors:
    """Frame-level factors of the uncertified sky: ambiguity A, contradiction C, overcast rim
    weight W and gap-free deck weight Wg (``frames.ambiguity``)."""

    amb: float = 0.0
    contra: float = 0.0
    rim: float = 0.0
    deck: float = 0.0


@dataclass
class UnknownContext:
    """What the pass needs besides the products: the parameters, the unchanged-sky test
    (``same(t, n, shape)``), the sun elevation of a frame and the zenith map (99 outside)."""

    p: CloudMapParams
    same: Callable[[str, str, tuple[int, int]], B]
    elev: Callable[[str], float]
    zen99: F32
    min_area: int = field(init=False)

    def __post_init__(self) -> None:
        self.min_area = int(np.pi * self.p.unknown_sig * self.p.unknown_sig)

    def zen(self, shape: tuple[int, int]) -> F32:
        z = self.zen99
        if z.shape == shape:
            return z
        return np.asarray(cv2.resize(z, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST))


def _png(path: Path) -> U8:
    return np.asarray(Image.open(path))


def _save_conf(path: Path, c: Any) -> F32:
    """Publish a confidence (uint8 code) and return it as read back."""
    Image.fromarray(np.rint(255 * c).astype(np.uint8)).save(path)
    return np.asarray(_png(path)).astype(np.float32) / 255.0


def _clean(mask: B, min_area: int) -> B:
    """Remove components (of mask and of its complement) smaller than min_area."""
    for target in (True, False):
        m = (mask == target).astype(np.uint8)
        _, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < min_area) + 1
        if small.size:
            mask = np.where(np.isin(lab, small), not target, mask)
    return mask


def _clean_valued(valued: B, min_area: int) -> B:
    """Valued components smaller than min_area become unknown (False)."""
    _, lab, st, _ = cv2.connectedComponentsWithStats(valued.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < min_area) + 1
    return np.where(np.isin(lab, small), False, valued) if small.size else valued


def _read(out: Path, ts: str) -> tuple[U8, F32] | None:
    """(value code, confidence BEFORE the cap) of a target or a rendered neighbour, or None."""
    for d in (out, out / "_neighbours"):
        vp = d / f"{ts}.png"
        if vp.exists():
            bk = d / "_unmasked" / f"{ts}.png"
            v = _png(bk if bk.exists() else vp)
            cu, ck = d / "_uncapped" / f"{ts}_conf.png", d / "_capped" / f"{ts}_conf.png"
            src = cu if cu.exists() else ck if ck.exists() else d / f"{ts}_conf.png"
            return v, _png(src).astype(np.float32) / 255.0
    return None


def temporal_conf(out: Path, ts: str, v: U8, c_capped: F32, ctx: UnknownContext) -> tuple[F32, int]:
    """The confidence of the frame read with its +-30 s neighbours: on pixels whose sky did
    not change, the median of the three confidences before the cap (the mean of two with one
    unchanged neighbour), the frame's own elsewhere; a pixel the cap lowered keeps its cap.
    The unchanged-sky weight is upsampled bilinearly (no 8 px staircase)."""
    own = _read(out, ts)
    cu = own[1] if own is not None else c_capped
    capped = c_capped < cu - 1.0 / 255.0
    has = v != 255
    h, w = v.shape
    nb: dict[int, tuple[F32, Any]] = {}
    for s in (-30, 30):
        n = shift_stamp(ts, s)
        r = _read(out, n)
        if r is None:
            continue
        try:
            same8 = ctx.same(ts, n, (h // 8, w // 8))
        except Exception:  # noqa: S112  # a neighbour without its image: not used
            continue
        same = cv2.resize(same8.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        nb[s] = (r[1], same * ((r[0] != 255) & has))
    eff: Any = cu
    if -30 in nb and 30 in nb:
        (a, sa), (b, sb) = nb[-30], nb[30]
        med = np.median(np.stack([a, cu, b]), axis=0)
        eff = (
            sa * sb * med
            + sa * (1 - sb) * 0.5 * (a + cu)
            + (1 - sa) * sb * 0.5 * (b + cu)
            + (1 - sa) * (1 - sb) * cu
        )
    elif nb:
        a, sa = next(iter(nb.values()))
        eff = sa * 0.5 * (a + cu) + (1 - sa) * cu
    return np.where(capped, c_capped, eff).astype(np.float32), len(nb)


# ---------------------------------------------------------------------------------- shapes


def _lines(m: B, near: B, ctx: UnknownContext) -> B:
    """c10: the components of m that are LINES (length >= line_l, mean width <= line_w,
    length / width >= line_elong), PERSIST (line_persist of their pixels near a neighbour's
    certification) and lie above the rim band (median zenith < line_zen)."""
    p = ctx.p
    mu = m.astype(np.uint8)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.line_w + 1, p.line_w + 1))
    wide = cv2.morphologyEx(mu, cv2.MORPH_OPEN, disc) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats((m & ~wide).astype(np.uint8), connectivity=8)
    out = np.zeros(m.shape, bool)
    zen = ctx.zen(m.shape)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) < p.line_l * 0.7 or a < p.line_l:
            continue
        sub = lab[y : y + h, x : x + w] == i
        if not _is_line(sub, a, p):
            continue
        if near[y : y + h, x : x + w][sub].mean() < p.line_persist:
            continue
        if np.median(zen[y : y + h, x : x + w][sub]) >= p.line_zen:
            continue
        out[y : y + h, x : x + w] |= sub
    return out


def _is_line(sub: B, a: int, p: CloudMapParams) -> bool:
    yy, xx = np.nonzero(sub)
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xx, yy], 1).astype(np.float32))
    length = max(rw, rh, 1.0)
    return not (
        length < p.line_l or length / max(a / length, 1.0) < p.line_elong or a / length > p.line_w
    )


def _enclosed(st: Any, i: int, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """The 4 px padded box of component ``i``, None when it touches the image edge."""
    x, y, w, h, _ = st[i]
    hh, ww = shape
    x0, y0, x1, y1 = max(x - 4, 0), max(y - 4, 0), min(x + w + 4, ww), min(y + h + 4, hh)
    if x0 == 0 or y0 == 0 or x1 == ww or y1 == hh:
        return None
    return x0, y0, x1, y1


def _ring(sub: B) -> B:
    return np.asarray((cv2.dilate(sub.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~sub)


def _drop_fragments(unk: B, has: B, v: U8, p: CloudMapParams) -> B:
    """c12 (L5): a VALUED fragment enclosed by unknown, smaller than frag_area and than the
    unknown region around it, with the unmasked value of its surroundings, becomes unknown."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((~unk & has).astype(np.uint8), connectivity=8)
    if n <= 1:
        return unk
    _, labu, stu, _ = cv2.connectedComponentsWithStats((unk & has).astype(np.uint8), connectivity=4)
    unk = unk.copy()
    for i in range(1, n):
        a = st[i, 4]
        box = None if a >= p.frag_area else _enclosed(st, i, unk.shape)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        sub = lab[y0:y1, x0:x1] == i
        ring = _ring(sub)
        hs, us = has[y0:y1, x0:x1], unk[y0:y1, x0:x1]
        if not (hs[ring].all() and (~us[ring]).mean() <= p.island_ring_unk):
            continue
        reg = np.bincount(np.asarray(labu[y0:y1, x0:x1][ring], np.int64)).argmax()
        if reg == 0 or stu[reg, cv2.CC_STAT_AREA] <= a:
            continue
        vs = v[y0:y1, x0:x1].astype(np.float32) / 254.0
        if abs(float(vs[sub].mean()) - float(vs[ring].mean())) > p.island_dv:
            continue
        us[sub] = True
    return unk


def _island_ok(sub: B, ring: B, a: int, vs: F32, p: CloudMapParams) -> bool:
    mi, mr = float(vs[sub].mean()), float(vs[ring].mean())
    if abs(mi - mr) > p.island_dv:
        return False
    # c12 (L5): a clear-reading hole is filled only when small
    return not (mr < 0.4 and a >= p.frag_area)


def _fill_islands(unk: B, has: B, v: U8, c: F32, p: CloudMapParams) -> tuple[B, F32]:
    """c10/c12: an UNKNOWN island enclosed by valued pixels, smaller than the valued body
    around it, with the unmasked value of its ring, is valued with its ring's confidence."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((unk & has).astype(np.uint8), connectivity=8)
    if n <= 1:
        return unk, c
    _, labv, stv, _ = cv2.connectedComponentsWithStats(
        (~unk & has).astype(np.uint8), connectivity=4
    )
    unk, c = unk.copy(), c.copy()
    for i in range(1, n):
        a = st[i, 4]
        box = _enclosed(st, i, unk.shape)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        sub = lab[y0:y1, x0:x1] == i
        ring = _ring(sub)
        hs, us = has[y0:y1, x0:x1], unk[y0:y1, x0:x1]
        if not (hs[ring].all() and us[ring].mean() <= p.island_ring_unk):
            continue
        bid = np.bincount(np.asarray(labv[y0:y1, x0:x1][ring], np.int64)).argmax()
        if bid == 0 or stv[bid, cv2.CC_STAT_AREA] <= a:
            continue
        if not _island_ok(sub, ring, a, v[y0:y1, x0:x1].astype(np.float32) / 254.0, p):
            continue
        cs = c[y0:y1, x0:x1]
        cs[sub] = np.maximum(cs[sub], float(np.median(cs[ring])))
        us[sub] = False
    return unk, c


def _close_structures(unk: B, has: B, c: F32, struct: B, p: CloudMapParams) -> tuple[B, F32]:
    """c16 (G): an UNKNOWN hole smaller than cert_minarea, enclosed by valued pixels mostly
    certified, is a gap the cue cannot resolve: delivered with its ring's confidence."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((unk & has).astype(np.uint8), connectivity=4)
    if n <= 1:
        return unk, c
    unk, c = unk.copy(), c.copy()
    for i in np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < p.cert_minarea) + 1:
        box = _enclosed(st, int(i), unk.shape)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        sub = lab[y0:y1, x0:x1] == i
        ring = _ring(sub)
        hs, us = has[y0:y1, x0:x1], unk[y0:y1, x0:x1]
        if (
            not (hs[ring].all() and us[ring].mean() <= p.island_ring_unk)
            or struct[y0:y1, x0:x1][ring].mean() < 0.5
        ):
            continue
        cs = c[y0:y1, x0:x1]
        cs[sub] = np.maximum(cs[sub], float(np.median(cs[ring])))
        us[sub] = False
    return unk, c


def _drop_unresolved(unk: B, has: B, p: CloudMapParams) -> B:
    """c16 (G): a delivered component smaller than cert_minarea that is not a line becomes
    unknown."""
    _, lab, st, _ = cv2.connectedComponentsWithStats((~unk & has).astype(np.uint8), connectivity=8)
    unk = unk.copy()
    for i in np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < p.cert_minarea) + 1:
        x, y, w, h, a = st[i]
        sub = lab[y : y + h, x : x + w] == i
        yy, xx = np.nonzero(sub)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xx, yy], 1).astype(np.float32))
        length = max(rw, rh, 1.0)
        if length >= p.line_l and length / max(a / length, 1.0) >= p.line_elong:
            continue
        unk[y : y + h, x : x + w] |= sub
    return unk


def _deck_guard(unk: B, has: B, v: U8, deck: float, p: CloudMapParams) -> B:
    """c16 (F): in a confirmed gap-free deck no delivered pixel reads in [alpha_clear, 0.5)."""
    if deck >= p.deck_on:
        return unk | (has & (v >= p.alpha_clear * 254) & (v < 0.5 * 254))
    return unk


# ---------------------------------------------------------------------------------- region


def _region_conf(out: Path, ts: str, has: B, ctx: UnknownContext) -> Any:
    """c16 (B): a frame's confidence read with its neighbours, at the diffuse scale."""
    r = _read(out, ts)
    if r is None:
        return None
    v, _ = r
    ck = None
    for d in (out, out / "_neighbours"):
        for q in (d / "_capped" / f"{ts}_conf.png", d / f"{ts}_conf.png"):
            if q.exists() and (d / f"{ts}.png").exists():
                ck = _png(q).astype(np.float32) / 255.0
                break
        if ck is not None:
            break
    if ck is None or ck.shape != has.shape:
        return None
    c, _ = temporal_conf(out, ts, v, ck, ctx)
    m = has.astype(np.float32)
    s = ctx.p.region_sig
    num = cv2.GaussianBlur(np.where(has, c, 0).astype(np.float32), (0, 0), s)
    return num / np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)


def _form(kn: B, has: B, ctx: UnknownContext) -> B:
    """c15 (B): a smooth front (K mask smoothed at 1.5 deg, kept above 0.5), known islands
    smaller than the fill kernel's disc dropped (the hole filling of c15 is OFF in c16)."""
    s = cv2.GaussianBlur(kn.astype(np.float32), (0, 0), ctx.p.wf_sig) > 0.5
    s &= has
    _, lab, st, _ = cv2.connectedComponentsWithStats(s.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < 64 * ctx.min_area) + 1
    if small.size:
        s &= ~np.isin(lab, small)
    return np.asarray(s)


def _wellformed(known: B, has: B, out: Path, ts: str, ctx: UnknownContext) -> B:
    """c15 (B): the diffuse-known region made well-formed; monotone through a twilight: the
    region of the -30 s frame is a floor at dawn (UTC morning) and a ceiling at dusk."""
    p = ctx.p
    kn = _form(known, has, ctx)
    prev = shift_stamp(ts, -30)
    for dd in (out, out / "_neighbours"):
        q = dd / "_known" / f"{prev}.png"
        if q.exists():
            kp = _png(q).astype(np.float32) / 255.0
            if kp.shape == kn.shape:
                k_prev = kp >= p.known_thr
                sp = _region_conf(out, prev, has, ctx)
                if sp is not None:
                    k_prev &= sp >= p.unknown_conf
                pk = _form(k_prev, has, ctx)
                kn = (kn | pk) if int(ts[8:10]) < 12 else (kn & pk)
            break
    return kn


# ---------------------------------------------------------------------------------- the pass


@dataclass
class _Frame:
    out: Path
    ts: str
    v: U8
    has: B
    c: F32
    cp: Path
    elev: float


def _apply_factors(f: _Frame, ff: FrameFactors, ctx: UnknownContext) -> None:
    """c10-c16: frame factors on the uncertified sky (ambiguity on cloud-like values,
    contradiction everywhere, overcast rim, gap-free deck below 0.5)."""
    p = ctx.p
    if not (ff.amb > 0 or ff.contra > 0 or ff.rim > 0 or ff.deck > 0):
        return
    cq = f.out / "_cert" / f"{f.ts}.png"
    certd = (_png(cq) > 0) if cq.exists() else np.zeros_like(f.has)
    has, v = f.has, f.v
    dif = has & ~certd & (v > p.alpha_clear * 254)
    fac = np.where(dif, 1.0 - ff.amb, 1.0) * np.where(has & ~certd, 1.0 - ff.contra, 1.0)
    if ff.rim > 0:
        zz = ctx.zen(has.shape)
        z0, z1 = p.rim_z
        rimf = np.clip((zz - z0) / (z1 - z0), 0.0, 1.0) * np.clip(
            (p.rim_v - v / 254.0) / 0.2, 0.0, 1.0
        )
        fac = fac * np.where(has & ~certd, 1.0 - ff.rim * rimf, 1.0)
    if ff.deck > 0:
        fac = fac * np.where(has & ~certd & (v < 0.5 * 254), 1.0 - ff.deck, 1.0)
    if (fac < 1.0).any():
        f.c = _save_conf(f.cp, (f.c * fac).astype(np.float32))


def _neighbour_cert(f: _Frame) -> list[B]:
    nb_c = []
    for sft in (-30, 30):
        n = shift_stamp(f.ts, sft)
        for dd in (f.out, f.out / "_neighbours"):
            q = dd / "_cert" / f"{n}.png"
            if q.exists():
                nb_c.append(_png(q) > 0)
                break
    return nb_c


@dataclass
class _Cert:
    """The certification read from ``_cert/`` and what persists in a neighbour."""

    carr: U8
    cert: B
    lines: B
    keep_s: B
    near: B | None = None


def _structures(f: _Frame, ctx: UnknownContext) -> _Cert:
    """c9-c15: persistent certified structure (lines, fields, strong certification) and the
    confidence floors it carries."""
    p = ctx.p
    cp_ = f.out / "_cert" / f"{f.ts}.png"
    carr = _png(cp_) if cp_.exists() else np.zeros(f.has.shape, np.uint8)
    field_ = (carr > 200) & (carr < 225)
    strong, weak = carr >= 225, carr > 0
    s = _Cert(carr, carr > 200, np.zeros_like(f.has), np.zeros_like(f.has))
    if weak.any():
        nb_c = _neighbour_cert(f)
        if nb_c:
            k = 2 * p.cert_r + 1
            s.near = np.zeros_like(s.cert)
            for cn in nb_c:
                ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                s.near |= cv2.dilate(cn.astype(np.uint8), ell) > 0
            s.cert = s.cert & s.near
    if field_.any() and s.near is not None:
        fp = field_ & s.near
        if fp.any():
            f.c = _save_conf(
                f.cp, np.where(fp, np.maximum(f.c, p.line_conf), f.c).astype(np.float32)
            )
        if f.elev > 0.0:
            s.keep_s = s.keep_s | (field_ & s.near)
    if strong.any() and s.near is not None and f.elev > 0.0:
        sp = strong & s.near
        s.keep_s = s.keep_s | sp
        if sp.any():
            snr = np.clip((carr.astype(np.float32) - 230.0) / 25.0, 0.0, 1.0)
            floor = p.line_conf + (p.struct_cmax - p.line_conf) * snr
            f.c = _save_conf(f.cp, np.where(sp, np.maximum(f.c, floor), f.c).astype(np.float32))
    if weak.any() and f.elev > 0.0:
        _deliver_lines(f, s, weak, ctx)
    return s


def _deliver_lines(f: _Frame, s: _Cert, weak: B, ctx: UnknownContext) -> None:
    """c10/c11: a persistent certified LINE is delivered, grown along its certified texture,
    with its confidence floored at line_conf."""
    p = ctx.p
    if s.near is not None:
        s.lines = _lines(weak & f.has, s.near, ctx)
        if s.lines.any():
            k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (p.cert_r // 3) + 1,) * 2)
            for _ in range(p.line_grow):
                s.lines = (cv2.dilate(s.lines.astype(np.uint8), k_) > 0) & weak & f.has
    if s.lines.any():
        s.cert = s.cert | s.lines
        f.c = _save_conf(
            f.cp, np.where(s.lines, np.maximum(f.c, p.line_conf), f.c).astype(np.float32)
        )


def _diffuse_region(f: _Frame, c_reg: F32, ctx: UnknownContext) -> B:
    """c9/c16: K at least known_thr and the confidence at the diffuse scale at the contract,
    made well-formed; written to ``_region/``."""
    p = ctx.p
    k = _png(f.out / "_known" / f"{f.ts}.png").astype(np.float32) / 255.0
    known = k >= p.known_thr
    m_ = f.has.astype(np.float32)
    s = p.region_sig
    num = cv2.GaussianBlur(np.where(f.has, c_reg, 0).astype(np.float32), (0, 0), s)
    big = num / np.maximum(cv2.GaussianBlur(m_, (0, 0), s), 1e-3)
    known = known & (big >= p.unknown_conf)
    known = _wellformed(known, f.has, f.out, f.ts, ctx)
    (f.out / "_region").mkdir(exist_ok=True)
    Image.fromarray(np.where(known & f.has, 255, 0).astype(np.uint8)).save(
        f.out / "_region" / f"{f.ts}.png"
    )
    return known


def _low_sun(f: _Frame, c_reg: F32, ff: FrameFactors, ctx: UnknownContext) -> B:
    """Below E_K: diffuse region + delivered structure + the pixel contract with its band."""
    p = ctx.p
    s = _structures(f, ctx)
    known = _diffuse_region(f, c_reg, ctx)
    if s.keep_s.any():
        _, lab_, st_, _ = cv2.connectedComponentsWithStats(
            (s.keep_s & ~known).astype(np.uint8), connectivity=8
        )
        small_ = np.flatnonzero(st_[1:, cv2.CC_STAT_AREA] < p.cert_minarea) + 1
        if small_.size:
            drop = np.isin(lab_, small_) & ~s.lines
            s.keep_s = s.keep_s & ~drop
            s.cert = s.cert & ~drop
    has = f.has
    unk = has & (((~known) & ~s.cert) | (f.c < p.unknown_conf - p.contract_band))
    val_ = ~unk & has
    unk = ~(_clean_valued(val_, 64 * ctx.min_area) | (val_ & (s.lines | s.keep_s))) & has
    if f.elev < p.island_e:
        unk = _drop_fragments(unk, has, f.v, p) & ~(s.keep_s & ~unk)
        unk, c2 = _fill_islands(unk, has, f.v, f.c, p)
        if (c2 != f.c).any():
            _save_conf(f.cp, c2)
            f.c = c2
    if f.elev > 0.0:
        unk, c2 = _close_structures(unk, has, f.c, (s.keep_s | s.lines | s.cert) & has, p)
        if (c2 != f.c).any():
            _save_conf(f.cp, c2)
    unk = _deck_guard(unk, has, f.v, ff.deck, p)
    if f.elev > 0.0:
        unk = _drop_unresolved(unk, has, p)
    return unk


def _daytime(f: _Frame, ff: FrameFactors, ctx: UnknownContext) -> B:
    """Above E_K: the smoothed-confidence boundary and the strict per-pixel contract."""
    p = ctx.p
    has, c = f.has, f.c
    h, w = has.shape
    small = (w // 8, h // 8)
    m8 = cv2.resize(has.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    c8 = cv2.resize(np.where(has, c, 0).astype(np.float32), small, interpolation=cv2.INTER_AREA)
    cs = cv2.GaussianBlur(c8, (0, 0), p.unknown_sig) / np.maximum(
        cv2.GaussianBlur(m8, (0, 0), p.unknown_sig), 1e-3
    )
    csf = cv2.resize(cs.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    unk = _clean((csf < p.unknown_conf) | ~has, 64 * ctx.min_area) & has
    unk |= has & (c < p.unknown_conf)
    unk = ~_clean_valued(~unk & has, 64 * ctx.min_area) & has
    if f.elev < p.island_e:
        unk = _drop_fragments(unk, has, f.v, p)
        unk, c2 = _fill_islands(unk, has, f.v, c, p)
        if (c2 != c).any():
            _save_conf(f.cp, c2)
    return _deck_guard(unk, has, f.v, ff.deck, p)


def _apply_one(out: Path, ts: str, ff: FrameFactors, ctx: UnknownContext) -> float | None:
    vp, cp = out / f"{ts}.png", out / f"{ts}_conf.png"
    if not vp.exists():
        return None
    bk = out / "_unmasked" / f"{ts}.png"
    if bk.exists():
        v = _png(bk)
    else:
        v = _png(vp)
        bk.parent.mkdir(exist_ok=True)
        Image.fromarray(v).save(bk)
    ck = out / "_capped" / f"{ts}_conf.png"  # the confidence after the cap, before this pass
    if not ck.exists():
        ck.parent.mkdir(exist_ok=True)
        ck.write_bytes(cp.read_bytes())
    c = _png(ck).astype(np.float32) / 255.0
    has = v != 255
    # c7: the published confidence is the one read with the neighbours
    c, _ = temporal_conf(out, ts, v, c, ctx)
    c = _save_conf(cp, c)
    f = _Frame(out, ts, v, has, c, cp, ctx.elev(ts))
    c_reg = c.copy()  # c16 (B): the region's confidence, before the frame factors
    _apply_factors(f, ff, ctx)
    if (out / "_known" / f"{ts}.png").exists():
        unk = _low_sun(f, c_reg, ff, ctx)
    else:
        unk = _daytime(f, ff, ctx)
    Image.fromarray(np.where(unk, 255, v).astype(np.uint8)).save(vp)
    return round(float(unk.sum() / max(has.sum(), 1)), 4)


def apply_unknown(
    out: Path, targets: list[str], factors: dict[str, FrameFactors], ctx: UnknownContext
) -> dict[str, float | None]:
    """Apply the unknown state to every target of a rendered folder, in time order; returns the
    unknown share of each target's sky."""
    return {t: _apply_one(Path(out), t, factors.get(t, FrameFactors()), ctx) for t in targets}
