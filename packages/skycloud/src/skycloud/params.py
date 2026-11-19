"""Every parameter of the cloud map (version c16), one typed and documented model.

Each value is the one the prototype ran with in round c16; its description says where it comes
from (a measurement, a reference file, a design choice) as the prototype's comments recorded
it. ``eval/...`` and ``review/...`` paths refer to the prototype's working folder (clouds_v3).
Changing a value changes the method: the defaults ARE version c16.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CloudMapParams"]


def _p(default: Any, why: str) -> Any:
    return Field(default=default, description=why)


class CloudMapParams(BaseModel):
    """Parameters of the cloud map, version c16 (defaults)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- where the sensor does not see the sky (2026-11-18, owner's decision; criteria in
    # skycloud-sun/criteria.md and its Addendum 1) -------------------------------------------
    unseen_sky: Literal["fill", "legacy"] = _p(
        "fill",
        "fill (default): no value only where there is no sky (installation, class 1; outside the "
        "domain); a value everywhere else. The aureole is valued as measured, with no cut; the "
        "sun's saturated core (skysun's definition), the other unmeasured pixels within "
        "sun_area_deg of the sun and the variable static of the day (class 2) take the value of "
        "the measured ring just outside them (fill_holes, after every measurement: spoilt pixels "
        "are never read upstream); no sun-artefact term in the confidence within sun_area_deg "
        "(read skysun's zone layer and skystatic's class 2 there). legacy: version c16 as frozen "
        "(no value on the sun core, saturated pixels and class 2; the kit's sun-artefact "
        "confidence and the c13/c15 aureole rule)",
    )
    unknown_state: Literal["layer", "mask"] = _p(
        "layer",
        "layer (default; owner's decision of 2026-11-19: a low confidence must not remove the "
        "value): the delivered value is the always-predicted one, everywhere there is sky; what "
        "the unknown rule would have blanked is written as a side mask (_unknown/<ts>.png, 255 = "
        "the final confidence is under the contract), to be combined by whoever reads the map. "
        "mask: version c16 as frozen (the unknown rule writes 255 into the delivered value)",
    )
    sun_area_deg: float = _p(
        30.0,
        "deg: the area around the sun where unseen_sky=fill fills holes and drops the sun term "
        "(criterion P3's bound)",
    )
    fill_ring_deg: tuple[float, ...] = _p(
        (0.3, 1.0, 3.0),
        "deg: widths of the measured ring a hole is filled from, tried in order: 0.3 deg (3 px "
        "on the GVCCS camera: the sky just around the hole; around the sun's core the measured "
        "value 3-9 px beyond the hole carries a bright filter ring, 0.35 against 0.15 on "
        "2024-01-15 11:00), then 1 and 3 deg when the thinner ring holds too few measured "
        "pixels (behind the skyline, inside a drop field)",
    )
    fill_ring_min_px: int = _p(20, "measured pixels a ring needs before its median is used")

    # --- frame level: the clear-sky level of the frame and its smoothing in time ---------
    s_n: float = _p(
        0.008, "kernel (in N units) of the mode seeking of the level: the adapted clear-sky spread"
    )
    prior: float = _p(0.15, "weight of the level prior at 0, as a share of the directions read")
    b_sd: tuple[float, float, float, float] = _p(
        (0.010, 0.04, 0.03, 0.04),
        "prior sd of the 4 level coefficients (constant, two solar-side terms, horizon "
        "haze): archive clear skies",
    )
    b_lim: float = _p(2.0, "bound of the level coefficients, in prior sd")
    con_min: float = _p(
        0.04, "the level is read where the clear/overcast colour contrast is at least this"
    )
    al_lim: tuple[float, float] = _p(
        (-0.23, 0.20),
        "c1: bounds of the brightness level = 0.5th / 99.5th percentile of aL over the "
        "1,088 archive clear frames",
    )
    t_sig: float = _p(8.0, "min: narrow Gaussian of the temporal level kernel")
    t_far: float = _p(60.0, "min: wide Gaussian of the temporal level kernel (cut at 3 t_far)")
    t_far_w: float = _p(0.02, "weight of the wide Gaussian")
    t_prior: float = _p(0.004, "weak prior at 0 in the temporal mean of the level")
    context_min: float = _p(30.0, "min: context frames read around each requested frame")
    context_step: int = _p(120, "s: the context frames are thinned to one per this")
    f_half_bands: tuple[tuple[float, float], ...] = _p(
        ((3.0, 0.53), (6.0, 0.035), (12.0, 0.00023), (25.0, 0.00051), (90.0, 0.00099)),
        "c2: own-support gate f / (f + f_half), f_half by sun band (upper edge, value) = "
        "95th percentile of the joint "
        "support of the archive OVERCAST frames of that band (c1_level_stats.py, "
        "ref/c1_level_stats_feat_v3.npz)",
    )
    t_shape: float = _p(
        2.0,
        "c10: min, Gaussian smoothing in time of the frame's shape ratio and q5 (below +10 deg)",
    )
    t_amb: float = _p(
        5.0, "c10: min, Gaussian smoothing in time of the joint support and the structure share"
    )
    amb_s: tuple[float, float] = _p(
        (0.005, 0.02),
        "c10: certified-structure share at or below which a frame is structure-less (1), "
        "and at or above which it is "
        "not (0); the development clear dusk 20231101 reads 0.015-0.03",
    )
    amb_j0: float = _p(
        0.39, "c10: joint clear-shape support of the archive overcast frames' 98th percentile"
    )
    # --- pixel level: the cues ------------------------------------------------------------
    dark_lo: float = _p(50.0, "counts: B + R below which the smoothing window is fully tripled")
    dark_hi: float = _p(110.0, "counts: B + R above which it is not tripled at all")
    r0: int = _p(5, "px: guided filter radius at the daytime low gain")
    r_max: int = _p(14, "px: largest guided filter radius")
    eps_k: float = _p(1.5, "guided filter eps = (eps_k sigma_G)^2")
    fine: float = _p(1.0, "px: Gaussian sigma of the fine G the luminance detail is read on")
    sat: int = _p(250, "a channel at or above this is saturated (the kit's value)")
    elev_min: float = _p(
        -3.5, "deg: below this sun elevation there is no sky signal above the pedestal: no value"
    )
    kappa_clear: float = _p(0.5, "weight of the brightness cue's clear verdict")
    kappa: float = _p(
        0.25,
        "weight of the brightness cue relative to its formal precision (an overcast is "
        "not every cloud)",
    )
    con_floor: float = _p(0.02, "floor of the colour contrast Nc - No")
    dl_floor: float = _p(0.15, "floor of the brightness span")
    tau_l: float = _p(
        1.0,
        "dead zone of the brightness departure, in clear-sky spreads (also of the cue agreement)",
    )
    fill_eps: float = _p(
        6.0,
        "precision below which the surroundings take over (also D / (D + fill_eps) of the "
        "confidence)",
    )
    t0: float = _p(0.015, "texture: clear-sky floor (log units)")
    t_noise: float = _p(0.5, "texture: noise term of the floor")
    t_span: float = _p(0.03, "texture: span from the floor to an opacity of 1")
    w_t: float = _p(25.0, "texture: weight")
    shrink: float = _p(
        1.0, "strength of the shrinkage of the colour opacity by its clear-sky uncertainty"
    )
    ref_deg: float = _p(
        3.0, "c1: deg, smoothing scale of the reference tables in angle to the sun (b4_smooth.py)"
    )
    base_q: int = _p(
        25,
        "c1: percentile of the local clear baseline (robust to structure over 3/4 of the window)",
    )
    k_gain: float = _p(
        0.02,
        "c2: colour offset per unit log gain step (30 s pairs with a gain step, sun 0-6 "
        "deg, development days, "
        "eval/c2_index/gain_k.md: 0.023, 0.017); confidence only",
    )
    # --- structure, shape and certification -----------------------------------------------
    alpha_clear: float = _p(
        0.2, "c7: below this value a verdict is 'clear' (the criteria's read-clear threshold)"
    )
    object_f0: float = _p(
        0.5, "c8: share of a cloud body's observable boundary on an edge (eval/c8/objects.md)"
    )
    field_dens: float = _p(
        0.3, "c11: density of lit mid-scale cue in the neighbourhood that certifies a field"
    )
    agree_sig: float = _p(
        3.0, "c13/c14: deg, smoothing of the cue-agreement factor at sun >= +10 deg (1 deg below)"
    )
    aur_min: float = _p(
        8.0, "c13: deg, smallest narrowed aureole cut on a clear homogeneous near-sun sky"
    )
    aur_veil: float = _p(
        18.0,
        "c15: deg, aureole cut under a homogeneous veil / overcast (measured continuity radius)",
    )
    # --- the unknown state (c6_unknown) ------------------------------------------------------
    unknown_conf: float = _p(
        0.30, "c6: no value where the final confidence is below this (criterion H)"
    )
    unknown_sig: float = _p(
        5.0, "c6: cells at 1/8 resolution, smoothing of the confidence = the fill kernel"
    )
    known_thr: float = _p(
        0.45,
        "c9: K level of the diffuse decision: the lowest K at which 90 % of the "
        "clear-valued sky of the development "
        "renders has its own confidence above 0.30 (eval/c9/known_calibration.md)",
    )
    cert_r: int = _p(
        15, "c9: px, a certified pixel needs a +-30 s neighbour's certification within this"
    )
    line_l: int = _p(40, "c10: px, minimum length of a delivered line (the fill kernel's scale)")
    line_w: int = _p(12, "c10: px, maximum mean width of a line")
    line_elong: float = _p(4.0, "c10: length / mean width of a line")
    line_persist: float = _p(
        0.5, "c10: share of a line covered by a +-30 s neighbour's certification"
    )
    line_zen: float = _p(75.0, "c10: deg, a delivered line lies (median) above the rim band")
    line_conf: float = _p(
        0.31, "c10: published confidence floor on a persistent certified line (0.30 plus one code)"
    )
    line_grow: int = _p(6, "c11: growth steps of a line along its certified texture")
    struct_cmax: float = _p(
        0.6, "c12: confidence of certified structure at twice the certification level"
    )
    rim_z: tuple[float, float] = _p((65.0, 75.0), "c12: deg, ramp of the overcast rim rule")
    rim_v: float = _p(
        0.8, "c12: a rim reading below this inside a confirmed deck contradicts it (full at 0.6)"
    )
    island_dv: float = _p(0.10, "c10: max difference of island and ring means: 'the same value'")
    island_ring_unk: float = _p(
        0.02, "c10: share of an island's ring that may touch other unknown pixels"
    )
    island_e: float = _p(10.0, "c10: deg, island / fragment rules only below this sun elevation")
    frag_area: int = _p(4 * 64 * 79, "c12: px, four times the small-island area of the fill kernel")
    wf_sig: float = _p(16.0, "c15: px, smoothing of the diffuse front (1.5 deg)")
    region_sig: float = _p(32.0, "c16: px, scale of the diffuse region's confidence (3 deg)")
    contract_band: float = _p(
        0.037,
        "c16: inside the diffuse region a pixel is unknown only below 0.30 - band: 3 x "
        "the largest frame-to-frame "
        "robust sd of the published confidence on unchanged sky at 0.20-0.40 "
        "(eval/c16/conf_noise.md)",
    )
    deck_on: float = _p(
        0.5, "c16: gap-free deck weight from which no delivered pixel reads in [alpha_clear, 0.5)"
    )
    cert_minarea: int = _p(
        400, "c15/c16: px, a disc of 2 deg: smaller isolated structures are not resolved"
    )
    # --- temporal consistency cap (c4_consistency) -------------------------------------------
    cap: float = _p(
        0.29, "c4: the confidence is capped at this where the value switches on unchanged sky"
    )
    cap_sig: float = _p(2.0, "c4: cells at 1/4 resolution, smoothing of the raw and value changes")
    # --- gain ledger (c5_ledger) -------------------------------------------------------------
    e_period: float = _p(
        12.0,
        "c5: deg, gain steps shift the index measurably only below this (eval/c2_index/gain_k.md)",
    )
    e_day: float = _p(
        10.0,
        "c10: deg, from here up the frame reads the c5-c9 (legacy) ledger, blended over 8-10 deg",
    )
    t_smooth: float = _p(
        8.0, "c5: min, temporal average removed from the cumulated step offsets (= t_sig)"
    )
