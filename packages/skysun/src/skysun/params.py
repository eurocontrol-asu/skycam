"""Parameters of the method, each with its origin.

Origins cited: the sun-zone prototype accepted by the camera's owner on eight frames on
2026-11-18 (``proto/zone.py`` of the run folder ``contrail/runs/20261118-skysun``); the
measurement of the sun on 400 archive frames (``clouds_v3/eval/c16/sun_spec.md``); skystatic's
ring fit (``optics.json``); the calibration of the reflection table (its metadata). See the
README table.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["ReflectionParams", "SunParams", "ZoneParams"]


class ZoneParams(BaseModel):
    """The sun zone: a radial profile of the glow measured on each frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_deg: float = Field(
        0.5, gt=0, description="Width of the annuli about the sun (prototype accepted by the owner)"
    )
    reach_deg: float = Field(
        45.0,
        gt=0,
        description="Annuli measured up to this angle (prototype); beyond it the zone is 0",
    )
    quantile: float = Field(
        0.25,
        gt=0,
        lt=1,
        description="The darker quarter of each annulus is its sky level, so that clouds near the "
        "sun, brighter than the sky beside them, do not shape the profile (definition agreed "
        "with the owner)",
    )
    min_pixels: int = Field(
        40, ge=1, description="An annulus with fewer measured pixels is interpolated (prototype)"
    )
    reference_deg: tuple[float, float] = Field(
        (30.0, 40.0),
        description="The sky's own level: the profile at these angles from the sun (prototype). "
        "sun_spec.md: on clear frames the brightness excess over the clear-sky reference is "
        "0 +- 0.05 beyond 8 degrees and the cloud value flat beyond 10 degrees",
    )
    fade_deg: tuple[float, float] = Field(
        (30.0, 40.0),
        description="The value fades to 0 across the reference annuli (raised cosine) instead "
        "of being cut at their outer edge: the zone is measured against that level, so nothing "
        "beyond it is attributed to the sun, and no edge is drawn there",
    )
    saturation: int = Field(
        250,
        ge=1,
        le=255,
        description="A channel at or above this is saturated; it is also the top of the headroom "
        "(prototype; the 8-bit JPEG frames of the site camera ring just below 255)",
    )
    smoothing_bins: float = Field(
        1.0, ge=0, description="Gaussian smoothing of the profile, in annuli (prototype)"
    )
    sun_link_deg: float = Field(
        3.0,
        gt=0,
        description="Saturated pixels connected to a saturated pixel within this angle of the "
        "computed sun are the sun's (its disc, glare and star spikes) and read 1. sun_spec.md: "
        "the saturated core reaches 4-5 degrees (p90 5-6) on clear frames at every elevation, "
        "so it always covers this angle when the sun shines",
    )
    core_deg: float = Field(
        2.0,
        gt=0,
        description="The zone's core: the sun's own disc and its first annuli, where whether the "
        "sun shines is read (sun_spec.md: saturated share of the 0-2 degree ring 0.79-0.96 "
        "median on clear frames, 0.00 under overcast)",
    )

    core_sky_min: float = Field(
        0.5,
        gt=0,
        le=1,
        description="The core is read when at least this share of its field pixels is static "
        "sky; below it the sun sits behind the tree line or the installation and the profile's "
        "glow decides (review of 2026-11-18, S1; design choice: a majority)",
    )
    glow_deg: tuple[float, float] = Field(
        (5.0, 10.0),
        description="The profile's glow read for that fallback: mean zone value over these "
        "angles from the sun (review S1: the darker-quarter value at 5-10 deg)",
    )
    core_full: float = Field(
        0.685,
        gt=0,
        le=1,
        description="Without a reflection table only: the sun shines fully from this saturated "
        "share of the core, 0 when no core pixel is saturated. With a table, its calibrated "
        "ramp (one top per elevation bin) is used. Default: the GVCCS archive, 5th percentile "
        "over 2313 clear frames with the sun at or above +10 deg (table v1)",
    )
    glow_ramp: tuple[float, float] = Field(
        (0.195, 0.260),
        description="Without a reflection table only: the fallback's glow ramp (0 at the first, "
        "1 at the second). With a table, its calibrated ramp is used. Default: the embedded "
        "table's values (see its metadata)",
    )
    temporal_window_s: float = Field(
        60.0,
        ge=0,
        description="In a run only: the profile used for a frame is the median of the profiles "
        "of the frames within this many seconds (the frame's own visibility and saturated "
        "pixels are kept). Review of 2026-11-18, S2: clouds passing inside the glow move the "
        "darker quarter of the annuli between 30-s frames; 0 disables",
    )
    temporal_min_visibility: float = Field(
        0.9,
        ge=0,
        le=1,
        description="Only frames whose sun shines at least this much enter that median, and only "
        "when the frame's own sun does: an appearance or a disappearance of the sun is never "
        "smoothed (review S2)",
    )

    @model_validator(mode="after")
    def ordered(self) -> ZoneParams:
        """The reference and fade annuli must lie inside the measured reach."""
        for low, high in (self.reference_deg, self.fade_deg):
            if not 0 <= low < high <= self.reach_deg:
                raise ValueError("reference_deg and fade_deg must be increasing, within reach_deg")
        return self


class ReflectionParams(BaseModel):
    """The rings of the optics: where their light is measured and painted, in native pixels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    band_px: float = Field(
        3.0,
        gt=0,
        description="Half-width of a ring line: skystatic masks +-1.5 px at a width of 1232 "
        "(its measured line width, ~3 px), 3 px at the native width of 2464",
    )
    flank_px: tuple[float, float] = Field(
        (7.0, 13.0),
        description="The sky beside a ring: this far from its line on both sides, so that a "
        "linear brightness gradient cancels (prototype step0.py); pixels in another ring's band "
        "are left out",
    )
    taper_px: float = Field(
        1.0,
        ge=0,
        description="The painted line keeps its value to band_px - taper_px and fades to 0 at "
        "band_px + taper_px (raised cosine): no step at the edge of a band",
    )
    sectors: int = Field(
        36, ge=4, description="Sectors of a ring, 10 degrees each (prototype step0.py)"
    )
    min_share: float = Field(
        0.5,
        gt=0,
        le=1,
        description="A sector is measured when at least this share of its band and of its "
        "flanks is usable sky (not saturated, not static); design choice: a majority",
    )
    eye: bool = Field(
        True,
        description="Also the central eye (skystatic's eye disc): the spots that turn with the "
        "sun near the centre, by the same sector table",
    )
    eye_flank_px: tuple[float, float] = Field(
        (4.0, 10.0),
        description="The sky around the eye disc, this far beyond its radius (design choice: "
        "the same 6-px-wide flank as the rings, starting past the disc's taper)",
    )


class SunParams(BaseModel):
    """All parameters of a sun map."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    zone: ZoneParams = ZoneParams()
    reflections: ReflectionParams = ReflectionParams()
