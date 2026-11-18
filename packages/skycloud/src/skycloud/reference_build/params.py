"""Every parameter of the reference-table build, one typed and documented model.

The defaults are the values the c16 tables of the GVCCS camera at Brétigny were built with;
each description says which prototype script (``clouds_v3/scripts/``) the value comes from and
what it is: a measurement on that camera, a choice of the method, or a property of the site.
Changing a value changes the tables.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from skycloud.params import CloudMapParams

__all__ = [
    "BinParams",
    "BuildParams",
    "LabelParams",
    "LowCountParams",
    "SmoothParams",
    "ThresholdParams",
]

Sigma3 = tuple[float, float, float]


def _p(default: Any, why: str) -> Any:
    return Field(default=default, description=why)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BinParams(_Frozen):
    """The archive's grid directions, their windows and the bins of the tables."""

    azimuth_step_deg: float = _p(2.0, "a2_features / kit sky_auto.grid_pixels: 180 azimuths")
    zenith_step_deg: float = _p(1.0, "a2_features / kit sky_auto.grid_pixels: 90 zenith angles")
    window_px: int = _p(
        9, "a2_features K: side of the window averaged around each grid pixel (camera scale)"
    )
    window_min_sky: int = _p(
        70,
        "b1_binmeans: a direction is read when at least this many of its window's pixels are sky",
    )
    zenith_max: float = _p(88.5, "b1_binmeans: deg, directions beyond are not read (rim)")
    saturation: int = _p(250, "a2_features: a pixel with a channel at or above this is saturated")
    bright_max: float = _p(
        245.0,
        "b1_binmeans: a direction whose brightest window-mean channel reaches this is not read",
    )
    gamma_bin_deg: float = _p(2.0, "v3_common GB: bin of the angle to the sun")
    n_gamma: int = _p(91, "v3_common NG: bins of the angle to the sun (0-182 deg)")
    zenith_bin_deg: float = _p(1.5, "v3_common ZB: bin of the zenith angle")
    n_zenith: int = _p(60, "v3_common NZ: bins of the zenith angle (0-90 deg)")
    bin_min_dirs: int = _p(2, "b1_binmeans: a frame's bin mean needs at least this many directions")
    pedestal_dark_r: float = _p(
        60.0,
        "c2 study: R corner (counts) from which the corners are dark (high gain); the pedestal "
        "colour G/R, B/R is measured on those frames and compared with the renderer's PED_GR, "
        "PED_BR (a check, nothing is changed)",
    )
    elevation_nodes: tuple[float, ...] = _p(
        (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45,
         50, 55, 60, 65),
        "v3_common ENODES: deg, sun elevations of the tables (dense at low sun, where the sky "
        "changes fastest); a frame belongs to its nearest node",
    )  # fmt: skip


class LabelParams(_Frozen):
    """The label-free clear / overcast rule of the archive frames (b2_labels.py)."""

    node_min_frames: int = _p(40, "a bin's archive percentiles need this many frames of the node")
    q_overcast: float = _p(
        10.0, "percentile of the archive's bin-mean index taken as overcast-like"
    )
    q_clear: float = _p(85.0, "percentile of the archive's bin-mean index taken as clear-like")
    min_gap: float = _p(
        0.03, "a bin is used where the colour cue separates: p85 - p10 above this (index units)"
    )
    min_gamma: float = _p(15.0, "deg: bins this close to the sun are not used (aureole)")
    max_zenith: float = _p(75.0, "deg: bins beyond are not used (horizon haze)")
    node_min_bins: int = _p(50, "a node is labelled only if it has this many usable bins")
    frame_min_share: float = _p(
        0.3, "a frame is labelled only if it reads more than this share of the usable bins"
    )
    clear_q90_max: float = _p(0.35, "clear: 90th percentile of the normalised gap below this")
    clear_q50_max: float = _p(0.15, "clear: median of the normalised gap below this")
    overcast_q10_min: float = _p(0.65, "overcast: 10th percentile of the normalised gap above this")
    episode_below: float = _p(
        6.0,
        "deg: below this sun elevation the colour cue is weak, a frame takes its episode's label",
    )
    episode_window: tuple[float, float] = _p(
        (6.0, 20.0), "deg: the frames of the half-day that decide its episode label"
    )
    episode_share: float = _p(0.8, "share of the episode frames that must agree")
    episode_min_frames: int = _p(4, "an episode needs this many labelled frames in its window")
    half_day_split_utc_hour: int = _p(
        12,
        "UTC hour separating the morning and afternoon episodes: a SITE property (solar noon is "
        "about 11:50 UTC at Brétigny, 2.35 deg E); take the hour of solar noon elsewhere",
    )


class SmoothParams(_Frozen):
    """Smoothing of the tables (b4_smooth.py, b6_spread.py, c1_postfit.py, c14_spread.py).
    Sigmas are in (elevation nodes, angle-to-sun bins, zenith bins)."""

    table_min_frames: int = _p(8, "b3_tables MIN: a raw bin statistic needs this many frames")
    count_cap: float = _p(60.0, "b4_smooth: a bin's weight is its frame count capped at this")
    den_min: float = _p(0.5, "b4_smooth: a smoothed bin needs this much weight around it")
    median_sigma: Sigma3 = _p(
        (0.6, 1.5, 1.0), "b4_smooth: medians (1.5 bins = 3 deg in angle to the sun: ref_deg)"
    )
    spread_sigma: Sigma3 = _p((0.8, 2.5, 2.0), "b4_smooth: spreads and texture")
    final_sigma: Sigma3 = _p((0.0, 0.7, 0.7), "b4_smooth: last light smoothing for the lookup")
    adapted_core_gamma: float = _p(
        12.0, "b6_spread: deg, the frame's own median residual is read beyond this from the sun"
    )
    adapted_core_zenith: float = _p(80.0, "b6_spread: deg, ... and above this zenith angle")
    adapted_sigma: Sigma3 = _p((0.8, 2.5, 2.0), "b6_spread: smoothing of the adapted spreads")
    adapted_den_min: float = _p(0.3, "b6_spread: a smoothed bin needs this share of filled bins")
    postfit_min_elev: float = _p(-3.5, "c1_postfit: deg, clear frames below are not read")
    postfit_fit_gamma: float = _p(12.0, "c1_postfit: deg, the level is fitted beyond this")
    postfit_fit_zenith: float = _p(86.0, "c1_postfit: deg, ... and above this zenith angle")
    postfit_min_dirs: int = _p(400, "c1_postfit: fewer fit directions: the level is 0")
    postfit_bin_gamma: float = _p(5.0, "c1_postfit: deg, residual bins start this far from the sun")
    bias_sigma: Sigma3 = _p((0.6, 1.5, 1.0), "c1_postfit: smoothing of Nc_bias (0 where empty)")
    sdf_sigma: Sigma3 = _p((0.8, 2.0, 1.5), "c1_postfit: smoothing of Nc_sdf")
    rebuild_sigma: Sigma3 = _p(
        (1.5, 4.0, 6.0), "c14_spread: heavy smoothing the spread tables are shrunk towards"
    )
    rebuild_pool: float = _p(
        1.0, "c14_spread POOL: independent frames per bin (neighbouring bins share frames)"
    )


class ThresholdParams(_Frozen):
    """The small threshold tables (c6, c7, c8, c9, c11 JSON files)."""

    shape_bands: tuple[tuple[float, float], ...] = _p(
        ((-1, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 25), (25, 70)),
        "c7_shape_stats: deg, sun bands of the brightness-shape ratio",
    )
    shape_quantiles: tuple[float, float] = _p(
        (90.0, 10.0), "c7_shape_stats: R0 = clear frames' p90, R1 = overcast frames' p10"
    )
    shape_min_frames: int = _p(10, "c7_shape_stats: a band needs this many clear and overcast")
    release_elev: tuple[float, float] = _p(
        (0.0, 12.0), "c8_shape_gate: deg, frames read for the level-release gate"
    )
    lift_bands: tuple[tuple[float, float], ...] = _p(
        ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 12), (12, 25)),
        "c7_lowsun_levels: deg, sun bands of the colour-lift table (c11)",
    )
    lift_quantiles: tuple[float, float] = _p(
        (95.0, 5.0), "c11: clear frames' p95 and overcast frames' p5 of the frame colour opacity"
    )
    lift_decimals: int = _p(2, "c11: the table was transcribed at 2 decimals from the c7 report")
    lift_min_frames: int = _p(5, "c7_lowsun_levels: a band needs this many clear frames")
    frame_min_dirs: int = _p(200, "c7/c8: a frame needs this many directions")
    knowable_elevations: tuple[float, float, float] = _p(
        (0.0, 15.0, 0.5), "c9_knowable: deg, start, stop, step of the sun elevations scanned"
    )
    knowable_sun_azimuth: float = _p(120.0, "c9_knowable: deg, the sun azimuth of the scan")
    knowable_min_gamma: float = _p(25.0, "c9_knowable: deg, sky outside the aureole")
    knowable_max_zenith: float = _p(80.0, "c9_knowable: deg, sky inside the rim")
    knowable_levels: tuple[float, float] = _p(
        (0.30, 0.45), "c9_knowable: K thresholds (the unknown threshold and 1.5 x it)"
    )
    knowable_share: float = _p(5.0, "c9_knowable: percentile of K over the sky (95 % above)")
    midscale_bands: tuple[float, ...] = _p(
        (-4, 0, 2, 4, 6, 9, 12, 20, 35, 70), "c6_midstats: deg, band edges"
    )
    midscale_frames: int = _p(600, "c6_midstats: random archive clear frames read")
    midscale_seed: int = _p(6, "c6_midstats: seed of the frame draw")
    midscale_erode_deg: float = _p(3.0, "c6_midstats: deg, the sky mask is eroded by this")
    midscale_min_gamma: float = _p(15.0, "c6_midstats: deg from the sun")
    midscale_max_zenith: float = _p(80.0, "c6_midstats: deg")
    midscale_min_pixels: int = _p(5000, "c6_midstats: a frame needs this many half-res pixels")


class LowCountParams(_Frozen):
    """The colour index's extra spread across camera gain steps (c3_lowcount_table.py,
    c4_lowcount_ped.py), measured on consecutive 30 s frames of dense days."""

    edges_counts: tuple[float, ...] = _p(
        (0, 30, 45, 60, 80, 100, 125, 150, 200, 260, 350, 520),
        "c3: bins of the window signal S = B + R (raw counts of the camera)",
    )
    edges_pedestal: tuple[float, ...] = _p(
        (0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 11.0, 16.0, 40.0),
        "c4: bins of the signal in pedestal units (B + R) / (pR (1 + PED_BR))",
    )
    step_min: float = _p(0.2, "|d log pedestal| above which a pair has a gain step")
    none_max: float = _p(0.02, "|d log pedestal| below which a pair has none")
    none_keep: float = _p(0.25, "share of the step-free pairs kept (random, seed below)")
    seed: int = _p(1, "seed of the step-free pair draw")
    min_elev: float = _p(-4.0, "deg: pairs whose first frame has the sun below are skipped")
    max_zenith: float = _p(80.0, "deg")
    smooth_max: float = _p(0.04, "a window is smooth when sd(G) / G is below this")
    subsample: int = _p(3, "every n-th direction of a pair is kept")
    quantile: float = _p(90.0, "upper quantile of |dN| per bin")
    min_samples: int = _p(300, "a bin's quantile needs this many samples")


class BuildParams(_Frozen):
    """Parameters of ``build_reference``: the c16 build of the GVCCS camera by default."""

    bins: BinParams = BinParams()
    labels: LabelParams = LabelParams()
    smooth: SmoothParams = SmoothParams()
    thresholds: ThresholdParams = ThresholdParams()
    lowcount: LowCountParams = LowCountParams()
    cloudmap: CloudMapParams = _p(
        CloudMapParams(),
        "the renderer's parameters: the post-fit replays the run-time level fit (s_n, prior, "
        "b_sd, b_lim, al_lim, con_min) and the knowability reads kappa_clear and fill_eps",
    )
    duplicates: Literal["error", "keep"] = _p(
        "error",
        "a timestamp listed twice: 'error' refuses the list; 'keep' reads it twice, as the c16 "
        "build did (264 frames were listed both in the daytime and in the twilight archive)",
    )
