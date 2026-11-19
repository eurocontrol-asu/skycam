"""The per-frame cloud map: value, confidence and the layers the unknown state reads.

``render_frame`` follows the method's stages in order: the sun and the valid sky; the two cues
through the edge-preserving filter (with the gain-ledger offset); the references at the
frame's clear-sky level; the diffuse knowability K below E_K; the cues' clear-sky spreads; the
single-cue opacities and their precisions (colour, brightness, texture, mid-scale); the
low-sun overcast lift; the fusion; the discriminability and the confidence; the lens flags; the
low-information fill; the aureole rule; the agreement of the cues with the value; the shape
rule; the structure certification. Its result is per frame; what links frames (the temporal
consistency cap, the unknown state) runs on the rendered products (``consistency``,
``unknown``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from skycloud.cues import (
    LFLOOR,
    PED_BR,
    clear_baseline,
    colour_opacity,
    fill_low_info,
    fuse,
    guided,
    level_basis,
    midscale,
    noise_sigma,
    ped_g,
    texture,
)
from skycloud.fill import fill_holes
from skycloud.geometry import CameraGeometry, angle_to_sun
from skycloud.images import pedestal
from skycloud.lens import lens_flags
from skycloud.level import NB
from skycloud.params import CloudMapParams
from skycloud.structure import (
    certification,
    cloud_objects,
    lobe_filtered,
    mid_neighbours,
    shape_clear,
)
from skycloud.sun import (
    AureoleInput,
    SunArea,
    artefact_confidence,
    aureole_confidence,
    detect_core,
)
from skycloud.tables import ReferenceTables

__all__ = ["FrameInputs", "FrameResult", "render_frame"]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]
U8 = NDArray[np.uint8]
BANDS = ((0, 40), (40, 60), (60, 70), (70, 80), (80, 85))


@dataclass(frozen=True)
class FrameInputs:
    """What one frame's render reads besides the image: its sun position, its static sky
    mask, its smoothed level, the gain-ledger offset of its colour index (full resolution;
    None: not corrected) with its note, loaders of its +-30 s neighbour images, and
    (``unseen_sky='fill'``) skysun's angle to the sun and saturated core, and the variable
    static of the day (skystatic class 2), filled at the end, never measured."""

    saz: float
    szen: float
    sky_mask: B
    level: dict[str, Any]
    offset: F32 | None
    offset_note: str
    neighbours: dict[int, Callable[[], U8 | None]]
    sun_area: SunArea | None = None
    spoilt: B | None = None


@dataclass
class FrameResult:
    """Value (float32 in [0, 1], NaN = no value), confidence (float32), the lens ghost flag,
    the lens ring side layer, the diffuse knowability K (below E_K) and the certification
    layer (below +10 deg), with the frame's diagnostics."""

    alpha: F32
    conf: F32
    info: dict[str, Any]
    lens: B | None = None
    rings: B | None = None
    known: F32 | None = None
    cert: F32 | None = None


@dataclass
class _F:
    """The arrays one frame's stages pass to each other."""

    img: U8
    elev: float
    saz: float
    szen: float
    info: dict[str, Any]
    v: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, k: str) -> Any:
        try:
            return self.__dict__["v"][k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def set(self, **kw: Any) -> None:
        self.v.update(kw)


class _Engine:
    """Geometry, tables and parameters of a render, with the stage functions."""

    def __init__(
        self, geometry: CameraGeometry, tables: ReferenceTables, p: CloudMapParams
    ) -> None:
        self.geo = geometry
        self.t = tables
        self.p = p

    def refs(self, names: tuple[str, ...], saz: float, szen: float, beta: Any) -> list[Any]:
        """Reference tables on the pixels (read at quarter resolution) and the frame's
        clear-sky level of N (level_basis . beta), upsampled bilinearly."""
        g4 = angle_to_sun(self.geo.az4, self.geo.zen4, saz, szen)
        out = self.t.ref.lookup(names, g4, self.geo.zen4, 90.0 - szen)
        out.append(
            (level_basis(g4, self.geo.zen4) @ np.asarray(beta, np.float64)).astype(np.float32)
        )
        h, w = self.geo.shape
        return [cv2.resize(o, (w, h), interpolation=cv2.INTER_LINEAR) for o in out]

    # ------------------------------------------------------------------ stage: sun, valid sky
    def layers(self, f: _F, mask: B) -> None:
        geo = self.geo
        sky = geo.inside & mask
        sun = detect_core(f.img, f.saz, f.szen, geo)
        core = sun.core & sky
        kconf = artefact_confidence(f.img, sun, geo.inside, sky)
        sat = f.img.max(-1) >= self.p.sat
        valid = sky & ~core & ~sat
        f.set(sun=sun, kconf=kconf, sat=sat, valid=valid, sky=sky)

    # ------------------------------------------------------------------ stage: cues
    def smoothed_cues(self, f: _F, x: FrameInputs) -> None:
        p = self.p
        valid = f.valid
        ped = pedestal(f.img)
        ped[1] = ped_g(ped)  # the G pedestal from the stable R corner
        fl = f.img.astype(np.float32)
        red, gr, bl = fl[..., 0], fl[..., 1], fl[..., 2]
        wv = valid.astype(np.float32)
        sig = max(noise_sigma(gr, valid), 0.8)
        gain = float(ped[1]) / 7.0  # the pedestal follows the gain (~7 counts of G at low gain)
        r = int(np.clip(round(p.r0 * np.sqrt(max(gain, 1.0))), p.r0, p.r_max))
        eps = max((p.eps_k * sig) ** 2, 1.0)
        rh, bh, gh = guided(gr, [red, bl, gr], wv, r, eps)
        # dark sky: the window is widened (x3) where B + R is low
        dark = np.clip((p.dark_hi - (bh + rh)) / (p.dark_hi - p.dark_lo), 0.0, 1.0)
        if dark.max() > 0:
            r2, b2, g2 = guided(gr, [red, bl, gr], wv, 3 * r, eps)
            rh, bh, gh = rh + dark * (r2 - rh), bh + dark * (b2 - bh), gh + dark * (g2 - gh)
        reff = r * (1.0 + 2.0 * dark)  # the effective radius, per pixel
        nb = (bh - rh) / np.maximum(bh + rh, 1.0)  # the smooth base of the colour cue
        # luminance detail: what the smoothing flattened in G is put back into N
        # (N x smooth G / fine G)
        k1 = cv2.GaussianBlur(wv, (0, 0), p.fine)
        gf = cv2.GaussianBlur(gr * wv, (0, 0), p.fine) / np.maximum(k1, 1e-4)
        q = np.clip(gh / np.maximum(gf, 1.0), 0.7, 1.4)
        f.info["gain_ledger"] = x.offset_note
        if x.offset is not None:
            nb = nb + x.offset
        n = nb * q
        lum = np.log(np.maximum(gh - ped[1], LFLOOR * ped[1]) / ped[1])
        f.set(
            ped=ped,
            g=gr,
            sig=sig,
            gain=gain,
            rh=rh,
            bh=bh,
            gh=gh,
            reff=reff,
            nb=nb,
            q=q,
            n=n,
            lum=lum,
        )
        f.set(ledger_used=x.offset is not None)

    # ------------------------------------------------------------------ stage: references
    def references(self, f: _F, level: dict[str, Any]) -> None:
        p, geo = self.p, self.geo
        beta = level.get("beta", [level.get("aN", 0.0)] + [0.0] * (NB - 1))
        names = (
            "Nc",
            "No",
            "Lc",
            "Lo",
            "Nc_sdf",
            "Lc_sda",
            "Nc_sd",
            "Lc_sd",
            "No_sd",
            "Lo_sd",
            "Nc_bias",
        )
        nc, no, lc, lo, sn, sl, snu, slu, sno, slo, bias, lvn = self.refs(
            names, f.saz, f.szen, beta
        )
        # c13 (L7): the table part of the confidence's discriminability, and its smooth version
        dtab = (
            np.maximum(nc - no, 0.0) ** 2 / (sn**2 + np.maximum(sno, 1e-3) ** 2)
            + p.kappa_clear * (lo - lc) ** 2 / (sl**2 + np.maximum(slo, 1e-2) ** 2)
        ).astype(np.float32)
        # c14 (L7): the same from the rebuilt spread tables (the confidence only)
        h, wd = geo.shape
        g4 = angle_to_sun(geo.az4, geo.zen4, f.saz, f.szen)
        s14 = [
            cv2.resize(o, (wd, h), interpolation=cv2.INTER_LINEAR)
            for o in self.t.spread.lookup(
                ("Nc_sdf", "Lc_sda", "No_sd", "Lo_sd"), g4, geo.zen4, f.elev
            )
        ]
        dnum = (
            np.maximum(nc - no, 0.0) ** 2 / (s14[0] ** 2 + np.maximum(s14[2], 1e-3) ** 2)
            + p.kappa_clear * (lo - lc) ** 2 / (s14[1] ** 2 + np.maximum(s14[3], 1e-2) ** 2)
        ).astype(np.float32)
        ins = geo.inside.astype(np.float32)
        sg = 2.5 * geo.px_deg
        dtab_s = cv2.GaussianBlur(dnum * ins, (0, 0), sg) / np.maximum(
            cv2.GaussianBlur(ins, (0, 0), sg), 1e-3
        )
        dratio = np.clip(dtab_s / np.maximum(dtab, 1e-6), 0.5, 2.0).astype(np.float32)
        f.set(nc=nc, no=no, lc=lc, lo=lo, sn=sn, sl=sl, snu=snu, slu=slu, sno=sno, slo=slo)
        f.set(bias=bias, lvn=lvn, dratio=dratio)

    # ------------------------------------------------------------------ stage: knowability K
    def knowability(self, f: _F, level: dict[str, Any], un: float, ul: float) -> F32 | None:
        """c9 (item 1): below E_K, the DIFFUSE KNOWABILITY from the tables and the frame's
        smoothed state only (no per-pixel term): smooth in space and time."""
        p = self.p
        ek = self.t.table("c9_knowable")["E_K"]
        if f.elev >= ek:
            return None
        con_n = np.maximum(f.nc - f.no, 0.0)
        wnc = (con_n / np.maximum(f.sno, 1e-3)) ** 2
        bk = 1.0 - np.clip(1.0 - 2.0 / np.sqrt(wnc + 1e-6), 0.0, 1.0)
        dk = (
            con_n**2 / (f.sn**2 + un * f.snu**2 + np.maximum(f.sno, 1e-3) ** 2)
            + p.kappa_clear
            * (f.lo - f.lc) ** 2
            / (f.sl**2 + ul * f.slu**2 + np.maximum(f.slo, 1e-2) ** 2)
            * bk
        )
        kmap = dk / (dk + p.fill_eps)
        # c9: an OVERCAST-SHAPED sky is readable as cloud whatever the colour separation: the
        # shape lift (c10: K + lift (1 - K), each pixel crosses the threshold at its own lift)
        _, r_k = self.shape(f, level)
        if r_k is not None:
            tab = self.t.table("c7_shape")
            r1k = float(np.interp(f.elev, tab["elev"], tab["R1"]))
            lift = float(np.clip((r_k - r1k) / r1k, 0.0, 1.0))
            kmap = 1.0 - (1.0 - kmap) * (1.0 - lift)
        # c9: K reaches 1 continuously at E_K (linear over the last degree)
        kmap = 1.0 - (1.0 - kmap) * float(np.clip(ek - f.elev, 0.0, 1.0))
        return np.asarray(kmap.astype(np.float32))

    def shape(self, f: _F, level: dict[str, Any]) -> tuple[float, float | None]:
        return shape_clear(
            self.geo, f.lum, f.lc, f.lo, f.valid, (f.saz, f.szen), level, self.t.table("c7_shape")
        )

    # ------------------------------------------------------------------ stage: spreads
    def spreads(self, f: _F, level: dict[str, Any], un: float, ul: float) -> None:
        p = self.p
        nf = np.sqrt(3.0) / (2 * f.reff + 1)
        sn = np.sqrt(
            f.sn**2 + un * f.snu**2 + (nf * 1.41 * f.sig / np.maximum(f.bh + f.rh, 1.0)) ** 2
        )
        # c3: the measured low-count spread e(S) of the colour index across gain changes
        sn = np.sqrt(sn**2 + self.t.low_count_spread(f.bh + f.rh) ** 2)
        # c2: an uncorrected gain mismatch with the neighbours: extra colour spread, confidence only
        sn_c = np.sqrt(sn**2 + (p.k_gain * level.get("gain_dev", 0.0)) ** 2)
        if f.ledger_used:
            sn_c = sn  # c7 (R6-2): the ledger corrected the frame: no double count
        if f.elev < 10.0:
            # c9: in the confidence, the low-count spread in PEDESTAL units (gain-free)
            e_raw = self.t.low_count_spread(f.bh + f.rh)
            e_ped = self.t.low_count_spread_ped(
                (f.bh + f.rh) / max(float(f.ped[0]) * (1.0 + PED_BR), 1.0)
            )
            sn_c = np.sqrt(np.maximum(sn_c**2 - e_raw**2, 0.0) + e_ped**2)
        signal = np.maximum(f.gh - f.ped[1], 0.0)
        sl = np.sqrt(f.sl**2 + ul * f.slu**2 + (nf * f.sig / np.maximum(signal, 0.3)) ** 2)
        f.set(nf=nf, sn=sn, sn_c=sn_c, sl=sl, signal=signal)

    # ------------------------------------------------------------------ stage: colour opacity
    def colour(self, f: _F) -> None:
        p, geo = self.p, self.geo
        con = f.nc - f.no
        conf_ = np.maximum(con, p.con_floor)
        lvn = f.lvn + f.bias  # the archive's median clear-sky residual after the level fit
        ab = (f.nc + lvn - f.nb) / conf_
        sa = f.sn / conf_
        wt = cv2.erode(f.valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        lg = np.log(np.maximum(f.g - f.ped[1], LFLOOR * f.ped[1]))
        t = texture(lg, wt)

        def _nb(a: Any, sg: float) -> Any:
            return cv2.GaussianBlur(a * wt, (0, 0), sg) / np.maximum(
                cv2.GaussianBlur(wt, (0, 0), sg), 1e-4
            )

        loc = np.clip((_nb(lg, 2.0) - _nb(lg, 8.0)) / np.maximum(t, 1e-4), 0.0, 1.0)
        t0 = np.maximum(p.t0, p.t_noise * np.sqrt(max(f.gain, 1.0)) / np.maximum(f.signal, 1.0))
        base = clear_baseline(ab, f.valid & (t < t0), geo.px_deg, p.ref_deg, p.base_q)
        a_n = colour_opacity(ab, sa, base, p.shrink)
        a_n = a_n + f.nb * (1.0 - f.q) / conf_
        wn_cloud = (np.maximum(con, 0.0) / f.sn) ** 2
        wn_clear = (np.maximum(con, 0.0) / np.maximum(f.sno, 1e-3)) ** 2
        f.set(
            con=con,
            conf_=conf_,
            lvn=lvn,
            ab=ab,
            sa=sa,
            wt=wt,
            lg=lg,
            t=t,
            loc=loc,
            t0=t0,
            base=base,
        )
        f.set(a_n=a_n, wn_cloud=wn_cloud, wn_clear=wn_clear)

    # ------------------------------------------------------------------ stage: brightness
    def brightness(self, f: _F, level: dict[str, Any]) -> None:
        p = self.p
        dl_ref = f.lo - f.lc
        dl = f.lum - f.lc - level.get("aL", 0.0)
        span = np.sqrt(dl_ref**2 + f.slo**2)
        tau = p.tau_l * f.sl
        a_l = (np.sqrt(dl * dl + tau * tau) - tau) / np.maximum(span, p.dl_floor)
        f.set(dl_ref=dl_ref, dl=dl, span=span, tau=tau, a_l=a_l)

    def cloud_lean(self, f: _F, level: dict[str, Any], s_clear: float) -> None:
        """c5: in a frame where no direction is jointly clear (gate < 1) AND the brightness says
        cloud, the shrinkage pulls towards CLOUD; c7: not in a clear-shaped sky."""
        p = self.p
        if level.get("gate", 1.0) >= 1.0:
            return
        wpr = (1.0 - level.get("gate", 1.0)) * np.clip(f.a_l, 0.0, 1.0)
        wpr = wpr * (1.0 - s_clear)
        b0 = 1.0 - (f.ab - f.base)
        a_nc = 1.0 - b0**3 / (b0 * b0 + p.shrink * f.sa * f.sa) + f.nb * (1.0 - f.q) / f.conf_
        f.set(a_n=(1.0 - wpr) * f.a_n + wpr * a_nc)

    # ------------------------------------------------------------------ stage: weights
    def weights(self, f: _F, level: dict[str, Any]) -> None:
        p = self.p
        wl_cloud = p.kappa * (f.span / f.sl) ** 2
        wl_clear = p.kappa_clear * (f.dl_ref / np.maximum(f.slo, 1e-2)) ** 2
        # hierarchy: brightness and texture speak to the extent the colour cannot exclude a cloud;
        # c3: without a jointly clear direction (gate -> 0) the colour cannot exclude one at all
        back = 1.0 - np.clip(1.0 - 2.0 / np.sqrt(f.wn_clear + 1e-6), 0.0, 1.0)
        g_fr = level.get("gate", 1.0)
        back = back + (1.0 - g_fr) * (1.0 - back)
        has = np.clip(f.signal / np.maximum(4.0 * f.sig * f.nf, 0.3), 0.0, 1.0)
        wn_cloud, wn_clear = f.wn_cloud * has, f.wn_clear * has
        wl_cloud, wl_clear = wl_cloud * has * back, wl_clear * has * back
        a_t = np.clip((f.t - f.t0) / p.t_span, 0.0, 1.0) * f.loc
        w_t = p.w_t * back * a_t * f.wt * f.kconf * f.kconf * float(np.clip(f.elev / 2.0, 0.0, 1.0))
        f.set(
            wl_cloud=wl_cloud,
            wl_clear=wl_clear,
            back=back,
            has=has,
            wn_cloud=wn_cloud,
            wn_clear=wn_clear,
        )
        f.set(a_t=a_t, w_t=w_t)

    # ------------------------------------------------------------------ stage: mid-scale cue
    def mid_scale(self, f: _F, x: FrameInputs) -> None:
        """c6/c7: mid-scale structure of the log brightness (1-3 deg band-pass) beyond the clear
        sky's 99.5 % tail, against the frame's own noise, persistent in the +-30 s neighbours;
        only at low sun (full below 5 deg, none above 7 deg)."""
        bpm, cov = midscale(f.lg, f.valid, self.geo.px_deg)
        mid = self.t.table("c6_midscale")
        sd_m, k_m = (
            float(np.interp(f.elev, mid["elev"], mid["sd"])),
            float(np.interp(f.elev, mid["elev"], mid["K"])),
        )
        w_low = float(np.clip((7.0 - f.elev) / 2.0, 0.0, 1.0))
        if w_low > 0:
            r_step = self.t.table("c5_ledger_params")["R_STEP"]
            bpm, nnote = mid_neighbours(
                x.neighbours, bpm, f.valid, cov, f.g, self.geo.px_deg, r_step
            )
            f.info["mid_nbr"] = nnote
            sd_m = max(sd_m, nnote.get("noise", 0.0))
        a_m = w_low * np.clip((np.abs(bpm) / sd_m - k_m) / k_m, 0.0, 1.0) * (cov > 0.9)
        f.set(bpm=bpm, cov=cov, sd_m=sd_m, k_m=k_m, w_low=w_low, a_m=a_m)

    # ------------------------------------------------------------------ stage: overcast lift
    def _lift_colour(self, elev: float) -> tuple[float, float]:
        """c11: (m0, m1): the archive clear frames' 95th and overcast frames' 5th percentile of
        the frame median colour opacity at this sun elevation (the +3.5 deg values below it)."""
        t = self.t.table("c11_lift_colour")
        e = max(elev, 3.5)
        m0, m1 = (
            float(np.interp(e, t["e"], t["clear_p95"])),
            float(np.interp(e, t["e"], t["ov_p5"])),
        )
        return m0, max(m1, m0 + 0.05)

    def _overflow_shift(
        self, f: _F, level: dict[str, Any], med: float, q5: float, p90: float
    ) -> float:
        """c9/c10: an overflowing bulk (p90 > 1) is held by its cells only where the frame has
        clear-coloured gaps (b_gap from the time-smoothed q5)."""
        shift = 1.0 - med
        if p90 > 1.0:
            b_ov = float(np.clip((med - 0.85) / 0.15, 0.0, 1.0))
            b_gap = float(np.clip(1.0 - level.get("q5_s", q5), 0.0, 1.0))
            shift = shift - b_ov * b_gap * max(p90 - med, 0.0)
        if f.elev >= 10.0:
            shift = max(shift, 0.0)  # c12 (L3): above +10 deg the lift only raises
        return shift

    def _saturate(self, f: _F, level: dict[str, Any], lc: tuple[float, float], w: float) -> float:
        """c12/c13 (L3): in a colour-confirmed overcast a pixel far from the clear reference
        saturates towards 1; from +4 deg where the deck has no clear-coloured gaps."""
        w_s = w * float(np.clip((f.elev - 6.0) / 2.0, 0.0, 1.0))
        b_g = float(np.clip((lc[1] - level.get("q5_s", 0.0)) / (lc[1] - lc[0]), 0.0, 1.0))
        w_s = max(w_s, w * (1.0 - b_g) * float(np.clip((f.elev - 4.0) / 2.0, 0.0, 1.0)))
        wp2 = np.clip((f.a_n - lc[0]) / (lc[1] - lc[0]), 0.0, 1.0)
        f.set(a_n=f.a_n + w_s * wp2 * np.maximum(1.0 - f.a_n, 0.0))
        return w_s

    def overcast_lift(self, f: _F, level: dict[str, Any], ratio: float | None) -> None:
        """c8-c13: an OVERCAST-shaped sky (shape ratio beyond the archive overcast p10 R1, full
        at 2 R1) has its colour opacity shifted towards a frame median of 1, weighted by the
        colour's agreement (c11) and never lifting a pixel within the clear range."""
        if ratio is None:
            return
        tab = self.t.table("c7_shape")
        r1 = float(np.interp(f.elev, tab["elev"], tab["R1"]))
        w_ov = float(np.clip((ratio - r1) / r1, 0.0, 1.0)) * 1.0
        if not (w_ov > 0 and f.valid.sum() > 1000):
            return
        sample = f.a_n[f.valid][::7]
        med = float(np.median(sample))
        q5, p90 = (float(v) for v in np.percentile(sample, [5, 90]))
        shift = self._overflow_shift(f, level, med, q5, p90)
        lc = self._lift_colour(f.elev)
        m_s = level.get("opa_med")
        w_col = 1.0 if m_s is None else float(np.clip((m_s - lc[0]) / (lc[1] - lc[0]), 0.0, 1.0))
        wp = np.clip((f.a_n - lc[0]) / (lc[1] - lc[0]), 0.0, 1.0) if shift > 0 else 1.0
        lift = {
            "w_ov": round(w_ov, 3),
            "w_col": round(w_col, 3),
            "m_s": None if m_s is None else round(m_s, 3),
        }
        f.info["lift"] = lift
        f.set(a_n=f.a_n + w_ov * w_col * wp * shift)
        if f.elev > 4.0:
            lift["sat"] = round(self._saturate(f, level, lc, w_ov * w_col), 3)
        f.info["overcast_level"] = {
            "w": round(w_ov, 3),
            "median_before": round(med, 3),
            "shift": round(w_ov * shift, 3),
            "q5": round(q5, 3),
            "p90": round(p90, 3),
            "q5_s": round(level.get("q5_s", float("nan")), 3),
        }

    def bright_hierarchy(self, f: _F) -> None:
        """c15 (A, R11-5): in a colour-confirmed overcast (from +9 deg, ramped in from +7) the
        brightness keeps its cloud vote but loses its clear vote where it reads less cloud than
        the colour."""
        lift = f.info.get("lift")
        if not isinstance(lift, dict):
            return
        w_ov = (
            float(lift.get("w_ov", 0.0))
            * float(lift.get("w_col", 0.0))
            * float(np.clip((f.elev - 7.0) / 2.0, 0.0, 1.0))
        )
        if w_ov > 0:
            fl = np.where(f.a_l < f.a_n, 1.0 - w_ov * np.clip(f.a_n, 0.0, 1.0), 1.0)
            f.set(wl_clear=f.wl_clear * fl, wl_cloud=f.wl_cloud * fl)

    # ------------------------------------------------------------------ stage: discriminability
    def discriminability(self, f: _F) -> Any:
        """How well clear sky and cloud can be told apart HERE, whatever the verdict, plus
        structure (c7 R6-3) and the absence of the clear sky's glare (c7 R6-4) where the value
        says cloud."""
        p = self.p
        disc = (
            np.maximum(f.con, 0.0) ** 2 / (f.sn_c**2 + np.maximum(f.sno, 1e-3) ** 2) * f.has
            + p.kappa_clear
            * f.dl_ref**2
            / (f.sl**2 + np.maximum(f.slo, 1e-2) ** 2)
            * f.has
            * f.back
            + 0.0
        )
        u_cloud = np.clip((f.alpha - p.alpha_clear) / p.alpha_clear, 0.0, 1.0)
        w_s = (
            p.w_t
            * f.back
            * np.maximum(f.a_t, f.a_m)
            * f.wt
            * f.kconf
            * f.kconf
            * float(np.clip(f.elev / 2.0, 0.0, 1.0))
        )
        disc = disc + w_s * u_cloud
        u_cloud = np.clip((f.alpha - p.alpha_clear) / p.alpha_clear, 0.0, 1.0)
        dark = np.clip((-f.dl - f.tau) / np.maximum(f.span, p.dl_floor), 0.0, 1.0)
        w_day = 1.0 - float(np.clip((7.0 - f.elev) / 2.0, 0.0, 1.0))
        return disc + w_day * f.wl_cloud * dark * u_cloud * f.has

    # ------------------------------------------------------------------ stage: lens
    def lens(self, f: _F, res: FrameResult) -> None:
        """Lens reflections are NOT removed: the kit's ghost detector feeds the flag only."""
        d = (f.n - f.nc - f.lvn).astype(np.float32)
        sun = f.sun
        try:
            lr = lens_flags(
                f.img,
                d,
                f.valid,
                sun.gam,
                self.geo.vec,
                self.geo.centre,
                sun.sun_px,
                sun.visible,
                (f.alpha > 0.15) & f.valid,
            )
        except Exception as e:  # the detector is optional
            f.info["lens_error"] = repr(e)[:120]
            res.lens = np.zeros(f.valid.shape, bool)
            return
        res.lens = lr.ghosts & f.valid
        nz = (lr.rings != 0) & f.valid
        res.rings = nz
        f.info["ring_px_share"] = round(float(nz.sum() / max(f.valid.sum(), 1)), 4)
        if nz.any():
            rv = (np.abs(lr.rings) / f.conf_)[nz][::50]
            f.info["ring_value_pct"] = [round(float(x), 3) for x in np.percentile(rv, [50, 90, 99])]
        f.info["lens"] = {"ghost_px": int((lr.ghosts & f.valid).sum()), "rings": lr.ring_bins}

    # ------------------------------------------------------------------ stage: agreement
    def agreement(self, f: _F, alpha: Any, w_meas: Any) -> tuple[Any, Any]:
        """c1: does a cue say CLOUDIER than the value, beyond its clear-sky uncertainty? The
        confidence is multiplied by the Gaussian likelihood of that disagreement beyond the
        dead zone, smoothed (c13/c14) at 3 deg in daytime, 1 deg at low sun. Returns (agree,
        agree without the brightness term)."""
        p = self.p
        z_n = np.maximum(0.0, np.minimum(f.a_n, 1.0) - alpha) * np.sqrt(f.wn_cloud)
        z_l = (
            np.maximum(0.0, np.minimum(f.a_l, 1.0) - alpha)
            * f.span
            / f.sl
            * f.has
            * np.sqrt(f.back)
        )
        # c8 D (R7-2): the brightness contradicts a value only towards the overcast reference
        w_dir = float(np.clip((10.0 - f.elev) / 2.0, 0.0, 1.0))
        z_l = z_l * (1.0 - w_dir * (f.dl * f.dl_ref < 0))
        z_t = f.a_t * np.clip(1.0 - alpha * np.sqrt(w_meas), 0.0, 1.0) * np.sqrt(p.w_t) * f.wt
        z_m = (
            f.w_low * np.maximum(0.0, np.abs(f.bpm) / f.sd_m - f.k_m) * (f.cov > 0.9)
            * np.sqrt(f.back) * f.kconf * np.clip(1.0 - alpha * np.sqrt(w_meas), 0.0, 1.0)
        )  # fmt: skip
        f.info["mid_share"] = round(float((z_m[f.valid] > 0).mean()), 4) if f.valid.any() else None
        z = np.sqrt(z_n * z_n + z_l * z_l + z_t * z_t + z_m * z_m)
        agree = np.exp(-0.5 * np.maximum(0.0, z - p.tau_l) ** 2)
        vf = f.valid.astype(np.float32)
        s1 = (p.agree_sig if f.elev >= 10.0 else 1.0) * self.geo.px_deg
        agree = (
            cv2.GaussianBlur(agree.astype(np.float32) * vf, (0, 0), s1)
            / np.maximum(cv2.GaussianBlur(vf, (0, 0), s1), 1e-3)
        ).astype(np.float32)
        z_nol = np.sqrt(z_n * z_n + z_t * z_t + z_m * z_m)
        agree_nol = np.exp(-0.5 * np.maximum(0.0, z_nol - p.tau_l) ** 2)
        return agree, agree_nol


def _finalise(f: _F, eng: _Engine, alpha: Any, conf: Any) -> tuple[F32, F32]:
    """No value outside the valid sky; frame statistics."""
    valid, info = f.valid, f.info
    alpha[~valid] = np.nan
    conf = np.where(valid, conf, 0.0).astype(np.float32)
    p = eng.p
    if p.unseen_sky == "fill":
        # where the sensor does not see the sky (the sun's core and the other unmeasured
        # pixels near the sun; the variable static of the day): the measured ring's value
        area = f.area
        near = f.sky & ~valid & (area.angle < p.sun_area_deg)
        spoilt = f.spoilt & eng.geo.inside
        widths = tuple(max(round(d * eng.geo.px_deg), 1) for d in p.fill_ring_deg)
        info["fill"] = {
            **fill_holes(alpha, conf, valid, near | spoilt, widths, p.fill_ring_min_px, area.core),
            "core_px": int((area.core & f.sky).sum()),
            "spoilt_px": int(spoilt.sum()),
        }
    info.update(
        r=float(np.median(f.reff[valid])) if valid.any() else None,
        sigma_g=round(f.sig, 2),
        pedestal=[round(float(x), 1) for x in f.ped],
    )
    v = np.isfinite(alpha)
    info["mean"] = round(float(alpha[v].mean()), 4) if v.any() else None
    info["conf_mean"] = round(float(conf[v].mean()), 4) if v.any() else None
    zen = eng.geo.zen90
    for z0, z1 in BANDS:
        m = v & (zen >= z0) & (zen < z1)
        info[f"z{z0}_{z1}"] = round(float(alpha[m].mean()), 4) if m.any() else None
        info[f"c{z0}_{z1}"] = round(float(conf[m].mean()), 4) if m.any() else None
    info["valid_80_85"] = round(float(valid[(zen >= 80) & (zen < 85)].mean()), 4)
    return alpha, conf


def _shape_rule(f: _F, eng: _Engine, conf: Any, alpha: Any, s_clear: float) -> tuple[Any, Any]:
    """c7 (R6-1): at low sun, in a clear-shaped sky, the share of the value above the
    read-clear level that no structure certifies takes the confidence with it."""
    p = eng.p
    obj = None
    w_low7 = float(np.clip((7.0 - f.elev) / 2.0, 0.0, 1.0))
    if s_clear > 0 and w_low7 > 0:
        obj = cloud_objects(f.lg, alpha, f.valid, p.alpha_clear, p.object_f0, f.info)
        ev = np.maximum(np.maximum(f.a_t, f.a_m), obj)
        d = np.clip((alpha - np.maximum(ev, p.alpha_clear)) / np.maximum(alpha, 1e-3), 0.0, 1.0)
        conf = conf * (1.0 - w_low7 * s_clear * d)
    return conf, obj


def _certify(f: _F, eng: _Engine, alpha: Any, alpha_m: Any, obj: Any, level: dict[str, Any]) -> F32:
    """c9-c16: local structure certification below +10 deg (``_cert/``)."""
    p, geo = eng.p, eng.geo
    if obj is None:
        obj = cloud_objects(f.lg, alpha, f.valid, p.alpha_clear, p.object_f0, f.info)
    a_m_c = lobe_filtered(f.a_m, f.bpm, geo.px_deg)
    stru = np.maximum(np.maximum(f.a_t, a_m_c), obj)
    cert, fld, strong = certification(
        stru, f.a_t, f.a_m, alpha, alpha_m, f.valid, geo.px_deg, p.alpha_clear, p.field_dens
    )
    mz = f.valid & (geo.zen90 < 75)
    f.info["struct_share"] = round(float(strong[mz].mean()) if mz.sum() else 0.0, 5)
    if "joint_s" in level:
        f.info["joint_s"] = round(level["joint_s"], 4)
    f.info["field_share"] = round(float(fld[f.valid].mean()) if f.valid.any() else 0.0, 5)
    return cert


def _confidence(
    f: _F, eng: _Engine, disc: Any, alpha: Any, w_meas: Any, s_clear: float
) -> tuple[Any, Any]:
    """Confidence = D / (D + fill_eps) x the (aureole-adjusted) artefact confidence x the
    cues' agreement; c9 1d: a decisive colour the value agrees with is not taken under 0.30
    by the brightness alone below +10 deg; then the c7 shape rule."""
    p = eng.p
    kconf_c = f.kconf
    if f.elev >= 10.0:
        kconf_c, aur = aureole_confidence(
            AureoleInput(f.kconf, alpha, f.valid, f.sat, f.saz, f.szen),
            eng.geo,
            alpha_clear=p.alpha_clear,
            aur_min=p.aur_min,
            aur_veil=p.aur_veil,
        )
        if aur is not None:
            f.info["aureole"] = aur
    if p.unseen_sky == "fill":
        # the sun-artefact term no longer lowers the confidence near the sun: there the
        # confidence says nothing about the sun (read skysun's zone layer)
        kconf_c = np.where(f.area.angle < p.sun_area_deg, np.float32(1.0), kconf_c)
    disc = disc * f.dratio
    conf = disc / (disc + p.fill_eps) * kconf_c
    agree, agree_nol = eng.agreement(f, alpha, w_meas)
    conf_pre = conf
    conf = conf * agree
    if f.elev < 10.0:
        cond = (f.back <= 0.5) & (np.abs(alpha - np.clip(f.a_n, 0.0, 1.0)) <= 0.1)
        conf = np.where(cond, np.maximum(conf, np.minimum(conf_pre * agree_nol, 0.31)), conf)
    return _shape_rule(f, eng, conf, alpha, s_clear)


def _opacities(eng: _Engine, f: _F, x: FrameInputs) -> float:
    """Cues to single-cue opacities and weights; returns s_clear."""
    level = x.level
    un, ul = 1.0 / (1.0 + level.get("fN", 0.0)), 1.0 / (1.0 + level.get("fL", 0.0))
    known = eng.knowability(f, level, un, ul)
    f.set(known=known)
    eng.spreads(f, level, un, ul)
    eng.colour(f)
    eng.brightness(f, level)
    s_clear, ratio = eng.shape(f, level)
    f.info["shape"] = {
        "ratio": None if ratio is None else round(ratio, 3),
        "s_clear": round(s_clear, 3),
    }
    if "ratio_s" in level:
        f.info["shape"].update(
            {"ratio_s": round(level["ratio_s"], 3), "ratio_g": round(level["ratio_g"], 3)}
        )
    eng.cloud_lean(f, level, s_clear)
    eng.weights(f, level)
    eng.mid_scale(f, x)
    eng.overcast_lift(f, level, ratio)
    eng.bright_hierarchy(f)
    return s_clear


def render_frame(
    img: U8,
    ts: str,
    x: FrameInputs,
    geometry: CameraGeometry,
    tables: ReferenceTables,
    p: CloudMapParams,
) -> FrameResult:
    """The continuous value, the confidence and the layers of one frame."""
    eng = _Engine(geometry, tables, p)
    elev = 90.0 - x.szen
    level = x.level
    info: dict[str, Any] = {
        "ts": ts,
        "elev": round(elev, 3),
        "aN": round(level.get("aN", 0.0), 5),
        "aL": round(level.get("aL", 0.0), 4),
        "beta": [round(float(b), 5) for b in level.get("beta", [])],
    }
    h, wd = geometry.shape
    if elev < p.elev_min:
        info["no_value"] = (
            f"sun below {p.elev_min:.1f} deg: no sky signal above the camera pedestal"
        )
        return FrameResult(
            np.full((h, wd), np.nan, np.float32), np.zeros((h, wd), np.float32), info
        )
    f = _F(img, elev, x.saz, x.szen, info)
    if p.unseen_sky == "fill":
        if x.sun_area is None:
            raise ValueError("sun handling fill needs FrameInputs.sun_area (skycloud.sun.sun_area)")
        f.set(area=x.sun_area)
        f.set(spoilt=np.zeros(x.sky_mask.shape, bool) if x.spoilt is None else x.spoilt)
    eng.layers(f, x.sky_mask)
    eng.smoothed_cues(f, x)
    eng.references(f, level)
    s_clear = _opacities(eng, f, x)
    alpha, w = fuse(f.a_n, f.a_l, f.a_t, f.wn_cloud, f.wn_clear, f.wl_cloud, f.wl_clear, f.w_t)
    f.set(alpha=alpha)
    disc = eng.discriminability(f)
    res = FrameResult(alpha, alpha, info)
    eng.lens(f, res)
    # c1: the artefact confidence discounts the cues' precision in the fill; c16 E: the
    # measured value before the fill is what certification reads
    alpha_m = np.clip(np.nan_to_num(alpha), 0.0, 1.0).astype(np.float32)
    alpha = fill_low_info(alpha, w * f.kconf, f.valid, p.fill_eps)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    conf, obj = _confidence(f, eng, disc, alpha, w, s_clear)
    if elev < 10.0:
        res.cert = _certify(f, eng, alpha, alpha_m, obj, level)
    res.known = f.known
    res.alpha, res.conf = _finalise(f, eng, alpha, conf)
    return res
