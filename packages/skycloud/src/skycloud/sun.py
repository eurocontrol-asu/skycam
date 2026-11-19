"""The sun: its position, its saturated core and the confidence its artefacts leave on the sky.

Everything the cloud map needs to know about the sun goes through this module, so that it is
the one seam where the future ``skysun`` library will plug in (its own sun mask and artefact
confidence replacing these functions behind the same interface).

The functions below are re-implemented from the GVCCS labelling kit
(``skycam-labelling-kit-2026-11-04/tools``): ``sky_geometry.sun_position`` (NOAA ephemeris,
about 0.1 deg), the core part of ``sun_mask.detect`` (the saturated components reaching the
ephemeris position, dilated) and ``sun_artefacts.confidence`` (confidence v2: the diffraction
star, the halo weighted by the sun's visibility, a halo floor). Only what the cloud map reads is
kept: the halo, spike and ghost labels of ``sun_mask.detect`` never reached it. The constants
are the kit's defaults. ``aureole_confidence`` is the cloud map's own (rounds c13 and c15):
where the kit's artefact cut around the sun may be narrowed. All of that is the ``legacy`` sun
handling (version c16, kept bit-identical).

The default handling (``unseen_sky='fill'``) gives a value everywhere within ``sun_area_deg`` of the
sun: ``sun_area`` reads, through ``skysun`` (the one definition shared by both libraries), the
angle of every pixel to the sun and the sun's saturated pixels (saturated pixels connected to a
saturated pixel within 3 deg of the computed sun: its disc, glare and star spikes);
``skycloud.fill.fill_holes`` gives every sky pixel without a measured value near the sun the
median value of the measured ring just outside its hole. skysun's sun position and
skycloud's ``sun_position`` (the kit's) differ by up to 6e-14 deg (not to the last bit), so
the engine keeps the kit's position everywhere (``legacy`` must stay bit-identical) and
skysun's position defines only the sun area and core.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from skycloud.geometry import CameraGeometry, angle_to_sun, to_vectors

__all__ = [
    "SAT",
    "SunArea",
    "SunCore",
    "artefact_confidence",
    "aureole_confidence",
    "detect_core",
    "sun_area",
    "sun_position",
]

F32 = NDArray[np.float32]
B = NDArray[np.bool_]
U8 = NDArray[np.uint8]

SAT = 250  # a channel at or above this is saturated (kit: sun_mask.SAT)
CORE_SEARCH_DEG = 3.0  # saturated components within this of the ephemeris position (sun_mask)
CORE_MARGIN_PX = 4  # the core dilated by this (sun_mask)
CORE_MIN_PX = 20  # the sun is visible when its core has at least this many pixels (sun_mask)
# sun_artefacts.DEFAULTS (confidence v2)
SPIKE_LINES = (7.9, 67.85, 128.05)  # deg, the aperture's 3 ray lines (measured by the kit)
ART = {
    "spike_search_deg": 1.5,
    "spike_z": 3.0,
    "spike_full_z": 8.0,
    "spike_halfwidth_px": 6,
    "spike_r0_px": 3,
    "spike_rmax_px": 1600,
    "spike_gap_px": 40,
    "halo_from": 0.3,
    "halo_full": 1.5,
    "halo_max_deg": 35.0,
    "vis_z0": 5.0,
    "vis_z1": 9.0,
    "vis_spread0": 0.15,
    "vis_spread1": 0.11,
    "halo_hidden_deg": 3.0,
    "halo_floor": 0.35,
}
PTH, PR = 0.2, 2.0  # polar sampling of the star: deg, px


def sun_position(t: datetime, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Sun azimuth (from north, towards east) and zenith angle, degrees (NOAA, ~0.1 deg).
    ``t`` must be timezone-aware UTC (its clock fields are read as UTC)."""
    jd = t.timestamp() / 86400 + 2440587.5
    tc = (jd - 2451545) / 36525
    l0 = (280.46646 + tc * (36000.76983 + tc * 0.0003032)) % 360
    m = np.radians(357.52911 + tc * (35999.05029 - 0.0001537 * tc))
    c = (
        np.sin(m) * (1.914602 - tc * (0.004817 + 0.000014 * tc))
        + np.sin(2 * m) * (0.019993 - 0.000101 * tc)
        + np.sin(3 * m) * 0.000289
    )
    om = np.radians(125.04 - 1934.136 * tc)
    lam = np.radians(l0 + c - 0.00569 - 0.00478 * np.sin(om))
    eps = np.radians(
        23
        + (26 + (21.448 - tc * (46.815 + tc * (0.00059 - tc * 0.001813))) / 60) / 60
        + 0.00256 * np.cos(om)
    )
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    e = 0.016708634 - tc * (0.000042037 + 0.0000001267 * tc)
    y = np.tan(eps / 2) ** 2
    l0r = np.radians(l0)
    eqt = 4 * np.degrees(
        y * np.sin(2 * l0r)
        - 2 * e * np.sin(m)
        + 4 * e * y * np.sin(m) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r)
        - 1.25 * e * e * np.sin(2 * m)
    )
    tst = (t.hour * 60 + t.minute + t.second / 60 + eqt + 4 * longitude_deg) % 1440
    ha = np.radians(tst / 4 - 180)
    la = np.radians(latitude_deg)
    zen = np.degrees(np.arccos(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))
    az = (
        np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(dec) * np.cos(la))) + 180
    ) % 360
    return float(az), float(zen)


@dataclass(frozen=True)
class SunCore:
    """The sun on one frame: its saturated core (dilated, inside the fisheye; empty when the
    sun is hidden), whether it is visible, its pixel and the angle of every pixel to it."""

    core: B
    visible: bool
    sun_px: tuple[int, int]
    gam: F32


def detect_core(img: U8, saz: float, szen: float, geometry: CameraGeometry) -> SunCore:
    """The saturated components reaching the ephemeris position (kit: sun_mask.detect)."""
    s = to_vectors(np.array(saz), np.array(szen)).astype(np.float32)
    gam = np.degrees(np.arccos(np.clip(geometry.vec @ s, -1, 1)))
    gam[~geometry.inside] = 180.0
    sy, sx = np.unravel_index(np.argmin(gam), gam.shape)
    sat = img.max(-1) >= SAT
    _, lab, _, _ = cv2.connectedComponentsWithStats(sat.astype(np.uint8), connectivity=8)
    near = np.unique(lab[(gam < CORE_SEARCH_DEG) & sat])
    core = np.isin(lab, near[near > 0])
    visible = bool(core.sum() >= CORE_MIN_PX)
    if szen > 89 or not visible:
        return SunCore(np.zeros(sat.shape, bool), visible, (int(sx), int(sy)), gam)
    k = CORE_MARGIN_PX
    core = cv2.dilate(core.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    core &= geometry.inside
    return SunCore(core, visible, (int(sx), int(sy)), gam)


def _fma32(a: Any, b: Any, c: Any) -> F32:
    """float32 fused multiply-add a * b + c (the float64 product of two float32 is exact)."""
    return np.asarray(
        (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)
    )


def bilinear_nan_border(y: F32, xs: F32, ys: F32) -> F32:
    """Bilinear sampling of ``y`` at (xs, ys), NaN where a neighbour lies outside: what
    OpenCV 5's ``remap(INTER_LINEAR, BORDER_CONSTANT, NaN)`` computes (float weights, fused
    multiply-adds, x then y). OpenCV 4 quantises the weights to 1/32 of a pixel instead, so
    the kit's star (built with OpenCV 5) is reproduced here independently of the installed
    OpenCV (checked bit-identical on 200,000 random points)."""
    h, w = y.shape
    x0, y0 = np.floor(xs), np.floor(ys)
    fx, fy = (xs - x0).astype(np.float32), (ys - y0).astype(np.float32)
    xi, yi = x0.astype(np.int64), y0.astype(np.int64)

    def at(yy: Any, xx: Any) -> F32:
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        v = np.full(xx.shape, np.nan, np.float32)
        v[ok] = y[yy[ok], xx[ok]]
        return v

    v00, v01, v10, v11 = at(yi, xi), at(yi, xi + 1), at(yi + 1, xi), at(yi + 1, xi + 1)
    v0 = _fma32(fx, v01 - v00, v00)
    v1 = _fma32(fx, v11 - v10, v10)
    return _fma32(fy, v1 - v0, v0)


def _polar(y: F32, cx: float, cy: float, thetas: Any, radii: Any) -> F32:
    t, r = np.meshgrid(np.radians(thetas), radii, indexing="ij")
    xs = (cx + r * np.cos(t)).astype(np.float32)
    ys = (cy - r * np.sin(t)).astype(np.float32)
    return bilinear_nan_border(y, xs, ys)


def _core_centre(core: B) -> tuple[Any, float]:
    n, _, st, cen = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    if n < 2:
        return None, 0.0
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cen[i], float(np.sqrt(st[i, cv2.CC_STAT_AREA] / np.pi))


def _ray_end(above: B, on_sky: B, gap: int) -> int:
    """From the sun outwards: a miss counts on the sky only (behind an obstacle: unknown)."""
    end, miss = 0, 0
    for i in range(len(above)):
        if above[i]:
            end, miss = i + 1, 0
        elif on_sky[i]:
            miss += 1
            if miss > gap:
                break
    return end


def _star_z(img: U8, inside: B, sky: B, cx: float, cy: float, rc: float) -> tuple[Any, ...]:
    """The polar contrast of the log luminance in robust sigmas of its ring."""
    p = ART
    y = np.log(np.maximum(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), 1.0))
    y[~inside] = np.nan
    th = np.arange(0, 360, PTH)
    radii = np.arange(rc + p["spike_r0_px"], p["spike_rmax_px"], PR)
    pol = _polar(y, cx, cy, th, radii)
    on_sky = _polar(sky.astype(np.float32), cx, cy, th, radii) > 0.5
    k = round(4.0 / PTH)
    pad = np.concatenate([pol[-k:], pol, pol[:k]], 0)
    loc = median_filter(np.nan_to_num(pad, nan=0.0), size=(2 * k + 1, 1))[k : k + len(th)]
    c = np.where(np.isfinite(pol), pol - loc, np.nan)
    nominal = [(a + q) % 360 for a in SPIKE_LINES for q in (0.0, 180.0)]
    off = np.ones(len(th), bool)
    for a in nominal:
        off &= np.abs(((th - a) + 180) % 360 - 180) > p["spike_search_deg"] + 1.0
    with np.errstate(all="ignore"):
        med = np.nanmedian(c[off], 0)
        noise = 1.4826 * np.nanmedian(np.abs(c[off] - med), 0) + 1e-4
        z = (c - med) / noise
    return th, radii, on_sky, z, nominal


def _star(img: U8, core: B, inside: B, sky: B) -> tuple[list[dict[str, float]], F32]:
    """The diffraction star (kit: sun_artefacts.star): per ray its angle, contrast near the sun
    and length; and the strength map in [0, 1]."""
    p = ART
    centre, rc = _core_centre(core)
    strength = np.zeros(core.shape, np.float32)
    if centre is None:
        return [], strength
    cx, cy = centre
    th, radii, on_sky, z, nominal = _star_z(img, inside, sky, cx, cy, rc)
    out = []
    hh, ww = core.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    dx, dy = xx - cx, cy - yy  # y up
    gap = max(1, int(p["spike_gap_px"] / PR))
    near = radii <= rc + 150
    for a0 in nominal:
        d = np.abs(((th - a0) + 180) % 360 - 180)
        cand = np.flatnonzero(d <= p["spike_search_deg"])
        with np.errstate(all="ignore"):
            score = np.nanmedian(z[cand][:, near], 1)
        if not np.isfinite(score).any():
            continue
        j = cand[int(np.nanargmax(score))]
        zr = cv2.GaussianBlur(
            np.nan_to_num(z[j], nan=0.0).reshape(1, -1).astype(np.float32), (0, 0), 4
        ).ravel()
        end = _ray_end(zr > p["spike_z"], on_sky[j], gap)
        a = float(th[j])
        length = float(radii[end - 1] - rc) if end else 0.0
        out.append(
            {
                "angle": round(a, 2),
                "z_near": round(float(np.nanmax(score)), 1),
                "length_px": round(length, 0),
            }
        )
        if not end:
            continue
        sv = np.clip((zr - p["spike_z"]) / (p["spike_full_z"] - p["spike_z"]), 0, 1)
        sv[end:] = 0.0
        sv = np.maximum.accumulate(sv[::-1])[::-1]  # the envelope from the sun outwards
        ux, uy = np.cos(np.radians(a)), np.sin(np.radians(a))
        along = dx * ux + dy * uy
        perp = np.abs(-dx * uy + dy * ux)
        band = (perp <= p["spike_halfwidth_px"]) & (along > rc) & (along < radii[end - 1])
        ri = np.clip(np.rint((along[band] - radii[0]) / PR).astype(int), 0, len(radii) - 1)
        w = np.clip(1.0 - perp[band] / (p["spike_halfwidth_px"] + 1.0), 0, 1)
        strength[band] = np.maximum(strength[band], sv[ri] * (0.5 + 0.5 * w))
    return out, strength


def _halo_confidence(img: U8, gam: F32, sky: B, sat: B) -> tuple[F32, float]:
    """1 - relative excess of the glare (ring low percentile over the far sky)."""
    p = ART
    y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    use = sky & ~sat
    edges = np.arange(0, 60.5, 0.5)
    gi = np.digitize(gam[use], edges) - 1
    yv = y[use]
    order = np.argsort(gi)
    gi, yv = gi[order], yv[order]
    starts = np.searchsorted(gi, np.arange(len(edges)))
    prof = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        seg = yv[starts[i] : starts[i + 1]]
        if seg.size >= 50:
            prof[i] = np.percentile(seg, 20)
    mid = (edges[:-1] + edges[1:]) / 2
    far = prof[(mid >= 45) & np.isfinite(prof)]
    if not far.size:
        return np.ones(gam.shape, np.float32), 0.0
    bg = max(float(np.median(far)), 1.0)
    ok = np.isfinite(prof)
    ex = np.interp(gam, mid[ok], np.maximum(prof[ok] - bg, 0) / bg, right=0.0)
    ex[gam > p["halo_max_deg"]] = 0.0
    conf = 1.0 - np.clip((ex - p["halo_from"]) / (p["halo_full"] - p["halo_from"]), 0, 1)
    return conf.astype(np.float32), float(np.nanmax(prof[ok] - bg) / bg)


def _glare_spread(img: U8, gam: F32, sky: B, sat: B, rim: float) -> float:
    """Median over 1-deg rings beyond the core's rim of the 10-90 % spread of log luminance."""
    y = np.log(np.maximum(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), 1.0))
    use = sky & ~sat
    sp = []
    for a in np.arange(1.0, 8.0, 1.0):
        m = use & (gam > rim + a) & (gam <= rim + a + 1)
        if m.sum() > 200:
            q = np.percentile(y[m], [10, 90])
            sp.append(float(q[1] - q[0]))
    return float(np.median(sp)) if sp else -1.0


def _visibility(rays: list[dict[str, float]], spread: float) -> float:
    """How directly the sun's disc reaches the lens: its star or a smooth glare, 0..1."""
    p = ART
    z = max((r["z_near"] for r in rays), default=0.0)
    vz = (z - p["vis_z0"]) / (p["vis_z1"] - p["vis_z0"])
    vs = (p["vis_spread0"] - spread) / (p["vis_spread0"] - p["vis_spread1"]) if spread >= 0 else 0.0
    return float(np.clip(max(vz, vs), 0.0, 1.0))


def artefact_confidence(img: U8, sun: SunCore, inside: B, sky: B) -> F32:
    """The kit's published sun-artefact confidence (float32, 1 trusted .. 0 not at all): per
    pixel the lowest of the diffraction star and the halo weighted by the sun's visibility,
    never below the halo floor. All ones when the sun has no core."""
    p = ART
    core, gam = sun.core, sun.gam
    conf = np.ones(core.shape, np.float32)
    if not core.any():
        return conf
    sat = img.max(-1) >= SAT
    rays, st = _star(img, core, inside, sky)
    rim = float(np.percentile(gam[core], 95))  # the core's angular radius
    spread = round(_glare_spread(img, gam, sky, sat, rim), 3)
    v = _visibility(rays, spread)
    ch, _ = _halo_confidence(img, gam, sky, sat)
    if v < 1.0:
        reach = np.clip(
            (rim + p["halo_hidden_deg"] - gam) / max(p["halo_hidden_deg"] / 2, 0.1), 0.0, 1.0
        )
        hidden = 1.0 - (1.0 - ch) * reach  # the glare within a few degrees of the rim only
        ch = (v * ch + (1.0 - v) * hidden).astype(np.float32)
    ch = np.maximum(ch, p["halo_floor"]).astype(np.float32)
    return np.asarray(np.minimum(ch, 1.0 - st))


@dataclass(frozen=True)
class AureoleInput:
    """What the aureole rule reads (quarter resolution except ``gp``)."""

    kconf: F32
    alpha: F32
    valid: B
    sat: B
    saz: float
    szen: float


def _kit_cut(g4: Any, zen4: Any, kq: F32, vq: B) -> int | None:
    for r0 in range(60):
        m = vq & (g4 >= r0) & (g4 < r0 + 1) & (zen4 < 80)
        if m.sum() > 5 and kq[m].mean() >= 0.9:
            return r0
    return None


def _saturation_radius(g4: Any, zen4: Any, sq: B) -> float:
    rs = 0.0
    for r0 in range(40):
        m = (g4 >= r0) & (g4 < r0 + 1) & (zen4 < 85)
        if m.sum() > 5 and sq[m].mean() > 0.01:
            rs = r0 + 1.0
    return rs


def aureole_confidence(
    a: AureoleInput,
    geometry: CameraGeometry,
    *,
    alpha_clear: float,
    aur_min: float,
    aur_veil: float,
) -> tuple[F32, dict[str, Any] | None]:
    """Rounds c13 (L6) and c15 (C, R11-7), sun at or above +10 deg: where the ring just beyond
    the kit's artefact cut is homogeneous, the cut is narrowed (confidence only).

    The kit's cut is the first 1-deg ring (zenith < 80) whose mean artefact confidence reaches
    0.9. Beyond it, a ring of 2-7 deg: on a CLEAR homogeneous ring (value median below
    alpha_clear, sd below 0.10) the confidence ramps to 1 over 2 deg from max(saturation radius
    + 2, aur_min) deg (eval/c13/aureole_*.md: the unmasked value inside the cut equals the value
    beyond to +0.02 down to 8 deg on clear skies); under a homogeneous veil or overcast (sd below
    0.20, median at least alpha_clear) from aur_veil = 18 deg, where the value becomes continuous
    with the sky beyond (eval/c13/aureole_c12.md). Returns (confidence, info or None)."""
    kconf_c = a.kconf
    info: dict[str, Any] | None = None
    g4 = angle_to_sun(geometry.az4, geometry.zen4, a.saz, a.szen)
    h, wd = a.kconf.shape
    gp = cv2.resize(g4.astype(np.float32), (wd, h), interpolation=cv2.INTER_LINEAR)
    kq, aq, vq, sq = a.kconf[::4, ::4], a.alpha[::4, ::4], a.valid[::4, ::4], a.sat[::4, ::4]
    zen4 = geometry.zen4
    cut = _kit_cut(g4, zen4, kq, vq)
    if cut is None:
        return kconf_c, info
    ring = vq & (g4 >= cut + 2) & (g4 < cut + 7) & (zen4 < 80)
    sd_ring = float(np.std(aq[ring])) if ring.sum() > 20 else 9.0
    hom = sd_ring < 0.10
    clear_ring = hom and float(np.median(aq[ring])) < alpha_clear
    if ring.sum() > 20 and sd_ring < 0.20 and float(np.median(aq[ring])) >= alpha_clear:
        r_new = aur_veil
        if r_new < cut:
            kconf_c = np.maximum(a.kconf, np.clip((gp - r_new) / 2.0, 0.0, 1.0)).astype(np.float32)
            info = {"kit_cut": cut, "new_cut": r_new, "sky": "veil/overcast"}
    if clear_ring:
        rs = _saturation_radius(g4, zen4, sq)
        r_new = max(rs + 2.0, aur_min)
        if r_new < cut:
            kconf_c = np.maximum(a.kconf, np.clip((gp - r_new) / 2.0, 0.0, 1.0)).astype(np.float32)
            info = {"kit_cut": cut, "sat_r": rs, "new_cut": r_new}
    return kconf_c, info


@dataclass(frozen=True)
class SunArea:
    """The sun as skysun defines it on one frame: the angle (deg, float32) of every pixel to the
    sun and the sun's saturated pixels (bool)."""

    angle: F32
    core: B


def sun_area(
    img: U8, when: datetime, latitude_deg: float, longitude_deg: float, geometry: CameraGeometry
) -> SunArea:
    """skysun's angle to the sun and saturated core of a frame (``when``: UTC, aware)."""
    from skysun import Site as SunSite
    from skysun.params import ZoneParams
    from skysun.sun import sun_position as skysun_position
    from skysun.zone import sun_saturation

    sgeo = geometry.skysun
    sun = skysun_position(when, SunSite(latitude_deg=latitude_deg, longitude_deg=longitude_deg))
    angle = sgeo.angle_to(sun)
    return SunArea(angle, sun_saturation(img, angle, sgeo.inside, ZoneParams()))
