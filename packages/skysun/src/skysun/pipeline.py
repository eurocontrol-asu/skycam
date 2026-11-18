"""The two layers of one frame: the sun zone and the reflections of the optics.

Each layer is a value in [0, 1] per pixel on its own scale, and they are delivered separately
(decisions of the camera's owner, 2026-11-18): the zone is the brightness the sun's glow adds as
a share of the headroom between the sky's own level and saturation; the reflections layer is the
predicted contrast of a reflection over the reference contrast of this optic's strongest ones.

A frame is first measured (``measure_frame``: sun position, zone profile, saturated pixels,
whether the sun shines), then rendered (``render``). ``sun_map`` does both for one frame alone;
a run may render a frame with the median profile of its neighbours in time
(``products.run_frames``), never a smoothed map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from skystatic import OpticsRings
from skysun.geometry import CameraGeometry
from skysun.params import SunParams
from skysun.reflections import RingLayout
from skysun.sun import Site, SunPosition, sun_position
from skysun.table import ReflectionTable, StructureInfo, VisibilityRamp
from skysun.zone import CoreState, ZoneProfile, core_state, radial_profile, sun_saturation, zone_map

__all__ = [
    "FrameInfo",
    "FrameMeasure",
    "SunCamera",
    "SunMap",
    "measure_frame",
    "render",
    "sun_map",
    "temporal_profile",
]

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
B = NDArray[np.bool_]


class FrameInfo(BaseModel):
    """What a frame's layers say in numbers (written as its JSON)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: datetime
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_pixel_xy: tuple[float, float] | None
    sky_level: float | None  # luminance (0-255) of the sky's own level, the zone's reference
    core_saturation: float  # saturated share of the core's field pixels
    core_sky: float  # share of the core's field pixels that are static sky
    visibility_source: Literal["core", "glow", "night"]
    saturation_radius_deg: float
    visibility: float  # whether the sun shines, 0..1 (multiplies the zone and the reflections)
    profile_frames: int  # frames whose profile made this frame's zone (1: the frame alone)
    zone_radius_deg: dict[str, float]  # where the zone (saturation aside) falls below 0.5/0.2/0.05
    reflections_max: float  # on the reflections layer's scale
    reflections_reference_contrast: float | None  # contrast worth 1 on that scale
    table_sha256: str | None


@dataclass(frozen=True)
class SunMap:
    """Per pixel, float32 in [0, 1]: the two layers, each on its own scale."""

    zone: F32
    reflections: F32
    info: FrameInfo


@dataclass(frozen=True)
class FrameMeasure:
    """What one frame says before it is rendered (no pixels of the frame are kept but the
    saturated ones of the sun)."""

    time: datetime
    sun: SunPosition
    profile: ZoneProfile | None  # None at night
    saturated: B | None
    core: CoreState
    visibility: float
    source: Literal["core", "glow", "night"]
    sun_xy: tuple[float, float] | None


class SunCamera:
    """What does not change from frame to frame: geometry, site, ring layout and table."""

    def __init__(
        self,
        geometry: CameraGeometry,
        site: Site,
        optics: OpticsRings | None,
        table: ReflectionTable | None,
        params: SunParams | None = None,
    ):
        self.geometry = geometry
        self.site = site
        self.params = params or SunParams()
        self.table = table
        self.layout = (
            None
            if optics is None
            else RingLayout(optics, geometry.shape, geometry.inside, self.params.reflections)
        )
        if table is not None:
            if self.layout is None:
                raise ValueError("A reflection table needs the ring circles of skystatic")
            table.check_structures(self.structures())
            if table.metadata.image_size_wh != geometry.size:
                raise ValueError("The reflection table was measured on another image size")

    @property
    def ramp(self) -> VisibilityRamp:
        """The calibrated visibility ramp, or the parameters' defaults without a table."""
        if self.table is not None:
            return self.table.metadata.visibility
        zone = self.params.zone
        return VisibilityRamp(
            elevation_edges=[0.0, 90.0],
            high=[zone.core_full],
            glow_low=zone.glow_ramp[0],
            glow_high=zone.glow_ramp[1],
            origin="ZoneParams defaults (no reflection table)",
        )

    def structures(self) -> list[StructureInfo]:
        """The structures of the ring layout, as a table records them."""
        if self.layout is None:
            return []
        return [
            StructureInfo(name=s.name, centre=s.centre, radius=s.radius, disc=s.disc)
            for s in self.layout.structures
        ]

    def reflections(self, sun: SunPosition, sun_xy: tuple[float, float], visibility: float) -> F32:
        """The predicted reflections layer: the table's contrast at the sun's elevation and
        direction over the reference contrast (clipped to 1), times whether the sun shines."""
        if self.layout is None or self.table is None or visibility <= 0:
            return np.zeros(self.geometry.shape, dtype=np.float32)
        table = self.table
        values = [
            table.layer_value(
                table.lookup(i, sun.elevation_deg, st.paint_angle - st.sun_angle(sun_xy))
            )
            * np.float32(visibility)
            for i, st in enumerate(self.layout.structures)
        ]
        return self.layout.paint(values)


def _visibility(
    profile: ZoneProfile, core: CoreState, sun: SunPosition, camera: SunCamera
) -> tuple[float, Literal["core", "glow"]]:
    """The core's reading when the core is mostly sky, else the profile's glow (review S1)."""
    params = camera.params.zone
    ramp = camera.ramp
    if core.sky >= params.core_sky_min:
        return ramp.from_core(core.saturated, sun.elevation_deg), "core"
    return ramp.from_glow(profile.glow(params.glow_deg)), "glow"


def measure_frame(
    rgb: NDArray[np.uint8], time: datetime, camera: SunCamera, sky: B
) -> FrameMeasure:
    """Measure one frame: (H, W, 3) 8-bit pixels at UTC ``time``; ``sky``: the static mask's
    sky pixels (where the glow is measured)."""
    geometry = camera.geometry
    if rgb.shape[:2] != geometry.shape or sky.shape != geometry.shape:
        raise ValueError(f"Frame {rgb.shape[:2]} / sky {sky.shape} / geometry {geometry.shape}")
    sun = sun_position(time, camera.site)
    if sun.elevation_deg <= 0:
        return FrameMeasure(time, sun, None, None, CoreState(0.0, 0.0), 0.0, "night", None)
    params = camera.params.zone
    angle = geometry.angle_to(sun)
    profile = radial_profile(rgb, angle, sky & geometry.inside, params)
    core = core_state(rgb, angle, geometry.inside, sky, params)
    visibility, source = _visibility(profile, core, sun, camera)
    saturated = sun_saturation(rgb, angle, geometry.inside, params)
    return FrameMeasure(
        time, sun, profile, saturated, core, visibility, source, geometry.sun_pixel(angle)
    )


def temporal_profile(
    measures: Sequence[FrameMeasure], index: int, camera: SunCamera
) -> tuple[F64 | None, int]:
    """The median zone profile of the frames within ``temporal_window_s`` of frame ``index``
    whose sun shines at least ``temporal_min_visibility``, when that frame's own sun does
    (else None: the frame's own profile). Returns (profile or None, frames used)."""
    params = camera.params.zone
    own = measures[index]
    if (
        params.temporal_window_s <= 0
        or own.profile is None
        or own.visibility < params.temporal_min_visibility
    ):
        return None, 1
    values = [
        m.profile.value
        for m in measures
        if m.profile is not None
        and m.visibility >= params.temporal_min_visibility
        and abs((m.time - own.time).total_seconds()) <= params.temporal_window_s
    ]
    if len(values) <= 1:
        return None, 1
    return np.median(np.stack(values), axis=0), len(values)


def _info(m: FrameMeasure, camera: SunCamera, frames: int, reflections: float) -> FrameInfo:
    profile = m.profile
    table = camera.table
    radii = {"0.5": 0.0, "0.2": 0.0, "0.05": 0.0}
    if profile is not None and m.visibility > 0:
        radii = {str(v): profile.radius_at(v / m.visibility) for v in (0.5, 0.2, 0.05)}
    return FrameInfo(
        time=m.time,
        sun_azimuth_deg=m.sun.azimuth_deg,
        sun_elevation_deg=m.sun.elevation_deg,
        sun_pixel_xy=m.sun_xy,
        sky_level=None
        if profile is None or np.isnan(profile.sky_level)
        else float(profile.sky_level),
        core_saturation=m.core.saturated,
        core_sky=m.core.sky,
        visibility_source=m.source,
        saturation_radius_deg=0.0 if profile is None else profile.saturation_radius(),
        visibility=m.visibility,
        profile_frames=frames,
        zone_radius_deg=radii,
        reflections_max=reflections,
        reflections_reference_contrast=None if table is None else table.metadata.reference_contrast,
        table_sha256=None if table is None else table.metadata.sha256,
    )


def render(
    m: FrameMeasure, camera: SunCamera, profile_value: F64 | None = None, frames: int = 1
) -> SunMap:
    """The two layers of a measured frame; ``profile_value`` replaces the frame's own zone
    profile (the temporal median of ``frames`` frames in a run). Every pixel of the field gets
    a value in [0, 1];
    0 everywhere when the sun is below the horizon."""
    geometry = camera.geometry
    if m.profile is None or m.saturated is None or m.sun_xy is None:
        zeros = np.zeros(geometry.shape, dtype=np.float32)
        return SunMap(zeros, zeros.copy(), _info(m, camera, 1, 0.0))
    profile = m.profile if profile_value is None else m.profile.with_value(profile_value)
    angle = geometry.angle_to(m.sun)
    zone = zone_map(profile, angle, m.saturated, geometry.inside, m.visibility)
    reflections = camera.reflections(m.sun, m.sun_xy, m.visibility)
    reflections[~geometry.inside] = 0.0
    used = FrameMeasure(
        m.time, m.sun, profile, m.saturated, m.core, m.visibility, m.source, m.sun_xy
    )
    info = _info(used, camera, frames, float(reflections.max()))
    return SunMap(zone, reflections, info)


def sun_map(rgb: NDArray[np.uint8], time: datetime, camera: SunCamera, sky: B) -> SunMap:
    """The two layers of one frame alone (no temporal median)."""
    return render(measure_frame(rgb, time, camera, sky), camera)
