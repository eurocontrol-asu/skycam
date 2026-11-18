from __future__ import annotations

import numpy as np
import pytest

from skystatic.contrast import (
    SmoothSky,
    band_pass,
    faint_inputs,
    faint_marks,
    fixed_marks,
    local_contrast,
    noise_by_zenith,
    outline_votes,
)

SIZE = 120


def grid() -> tuple[np.ndarray, np.ndarray]:
    return np.indices((SIZE, SIZE)).astype(np.float32)


def rgb(luminance: np.ndarray) -> np.ndarray:
    return np.repeat(luminance[..., None], 3, axis=2).astype(np.float32)


def sky_frame(rng: np.random.Generator) -> np.ndarray:
    """Sky that brightens towards one side (a horizon), with a little noise."""
    y, x = grid()
    smooth = 80 * np.exp(0.8 * ((x - 60) / 60) ** 2 + 0.3 * (y / SIZE))
    return smooth * (1 + 0.003 * rng.standard_normal((SIZE, SIZE)))


def spot(scale: float, centre: tuple[int, int] = (60, 40), radius: float = 4) -> np.ndarray:
    y, x = grid()
    return 1 + scale * np.exp(-((y - centre[0]) ** 2 + (x - centre[1]) ** 2) / (2 * radius**2))


def test_a_quadratic_sky_has_no_contrast_even_next_to_an_object() -> None:
    y, x = grid()
    log_sky = 4 + 0.5 * (x / SIZE) ** 2 - 0.3 * (y / SIZE) + 0.2 * x * y / SIZE**2
    weights = np.ones((SIZE, SIZE), bool)
    weights[:, 100:] = False  # a treeline: no sky there, the fit must not lean on it
    contrast = local_contrast(rgb(np.exp(log_sky)), SmoothSky(weights, sigma=10))
    assert np.abs(contrast[:, :100]).max() < 1e-3


def test_a_spot_has_contrast_over_its_whole_interior_for_either_polarity() -> None:
    weights = np.ones((SIZE, SIZE), bool)
    surface = SmoothSky(weights, sigma=12)
    base = np.full((SIZE, SIZE), 100.0)
    for scale in (0.2, -0.2):
        contrast = local_contrast(rgb(base * spot(scale)), surface)
        centre = contrast[57:64, 37:44]  # the flat centre, where an edge test sees nothing
        assert np.all(np.sign(centre) == np.sign(scale))
        assert np.abs(centre).min() > 0.05


def test_noise_is_measured_per_zenith_band() -> None:
    rng = np.random.default_rng(0)
    zenith = np.tile(np.linspace(0, 89.9, SIZE, dtype=np.float32), (SIZE, 1))
    contrast = (rng.standard_normal((SIZE, SIZE)) * np.where(zenith < 45, 0.01, 0.04)).astype(
        np.float32
    )
    noise = noise_by_zenith(contrast, np.ones((SIZE, SIZE), bool), zenith, band=5.0)
    assert noise[0, 10] == pytest.approx(0.01, rel=0.2)
    assert noise[0, 110] == pytest.approx(0.04, rel=0.2)


def windows_with(spots: list[np.ndarray]) -> np.ndarray:
    """Hourly windows (log mean, mean log luminance) of a sky carrying the given spot factors."""
    rng = np.random.default_rng(1)
    logs = [np.log(sky_frame(rng) * s).astype(np.float32) for s in spots]
    return np.stack([np.stack([v, v]) for v in logs])


def run_marks(windows: np.ndarray) -> np.ndarray:
    domain = np.ones((SIZE, SIZE), bool)
    zenith = np.tile(np.linspace(0, 89.9, SIZE, dtype=np.float32), (SIZE, 1))
    return fixed_marks(windows, domain, zenith, sigma=12, k=3).marks


def test_a_mark_present_all_day_is_found_whole() -> None:
    marks = run_marks(windows_with([spot(-0.1)] * 8))
    assert marks[56:65, 36:45].all()  # interior and rim
    assert marks.sum() < 300  # nothing else: the brightening sky is not a mark


def test_a_mark_that_flips_sign_during_the_day_is_found() -> None:
    # Dirt: dark against the bright morning sky, bright once the sun lights it; one hour
    # passes through zero.
    hours = [spot(-0.08)] * 3 + [np.ones((SIZE, SIZE))] + [spot(0.08)] * 4
    marks = run_marks(windows_with(hours))
    assert marks[58:63, 38:43].all()


def test_a_mark_changes_sign_only_at_comparable_amplitude() -> None:
    from skystatic.contrast import _counted_hits

    # Per window (6 hours) for 2 pixels: a lens ghost, bright for 2 hours then a faint dark
    # trace (25-35 times weaker); dirt that changes sign with the lighting (3-4 times weaker).
    contrast = np.array(
        [
            [0.035, -0.008],
            [0.024, -0.008],
            [-0.001, -0.008],
            [-0.001, 0.026],
            [-0.001, 0.018],
            [-0.0012, 0.0],
        ],
        dtype=np.float32,
    )[:, :, None]
    significance = contrast / np.float32(0.0003)  # every window above 3 noise units but the last
    significance[5, 1] = 0
    counted = _counted_hits(significance, contrast, 3, 0.2)
    assert counted[:, 0, 0].sum() == 2  # the ghost: only its bright hours
    assert counted[:, 1, 0].sum() == 5  # the dirt: both signs
    assert _counted_hits(significance, contrast, 3, 0.0)[:, 0, 0].sum() == 6


def test_a_spot_present_in_a_minority_of_hours_is_not_a_mark() -> None:
    # A cloud parked for a few hours, or the sun's track passing over the pixel.
    flat = np.ones((SIZE, SIZE))
    marks = run_marks(windows_with([spot(0.3)] * 3 + [flat] * 5))
    assert not marks[50:70, 30:50].any()


def test_a_single_window_cannot_make_a_mark() -> None:
    assert not run_marks(windows_with([spot(0.3)])).any()


def test_hours_near_the_sun_do_not_vote() -> None:
    # Significant in 5 of 8 hours, all of them with the sun near the pixel (glare, lit dust): the
    # 3 hours judged do not show it.
    flat = np.ones((SIZE, SIZE))
    windows = windows_with([spot(0.3)] * 5 + [flat] * 3)
    usable = np.ones((8, SIZE, SIZE), bool)
    usable[:5] = False
    domain = np.ones((SIZE, SIZE), bool)
    zenith = np.tile(np.linspace(0, 89.9, SIZE, dtype=np.float32), (SIZE, 1))
    marks = fixed_marks(windows, domain, zenith, sigma=12, k=3, usable=usable).marks
    assert not marks[50:70, 30:50].any()
    assert run_marks(windows)[58:63, 38:43].all()  # without the sun guard it would be a mark


def test_an_outline_that_stays_put_votes_in_every_hour() -> None:
    rng = np.random.default_rng(3)
    zenith = np.tile(np.linspace(0, 89.9, SIZE, dtype=np.float32), (SIZE, 1))
    hours = []
    for sign in (1, -1, 1, -1, 1, -1):  # a wall lit, then shaded: the step changes sign
        image = np.log(100 + rng.normal(0, 0.3, (SIZE, SIZE))).astype(np.float32)
        image[60:, 70:72] += sign * 0.05  # a thin rod
        hours.append(np.stack([image, image]))
    votes = outline_votes(np.stack(hours), np.ones((SIZE, SIZE), bool), zenith, k=3, band=5.0)
    assert (votes[70:110, 69] == 6).all() or (votes[70:110, 72] == 6).all()
    assert (votes[:50] >= 4).mean() < 0.001  # sky texture: no majority of the 6 hours


def faint_windows(hourly: list[np.ndarray]) -> np.ndarray:
    """Hourly windows of a sky with independent noise per hour, carrying the given factors."""
    rng = np.random.default_rng(7)
    logs = [np.log(sky_frame(rng) * f).astype(np.float32) for f in hourly]
    return np.stack([np.stack([v, v]) for v in logs])


def line(scale: float, row: int = 60, start: int = 20, stop: int = 100) -> np.ndarray:
    y, x = grid()
    across = np.exp(-((y - row) ** 2) / 2)
    return 1 + scale * across * ((x >= start) & (x <= stop))


def run_faint(
    windows: np.ndarray,
    seeds: np.ndarray | None = None,
    zenith_deg: float = 10.0,
) -> np.ndarray:
    sky = np.ones((SIZE, SIZE), bool)
    seeds = np.zeros((SIZE, SIZE), bool) if seeds is None else seeds
    zenith = np.tile(np.linspace(zenith_deg, zenith_deg + 40, SIZE, dtype=np.float32), (SIZE, 1))
    inputs = faint_inputs(windows, [10] * len(windows), sky, seeds, sigma=12)
    assert inputs is not None
    return faint_marks(
        inputs,
        zenith,
        scales=(1.2, 4.0),
        weak_level=0.003,
        weak_k=2,
        alone_level=0.005,
        alone_k=4,
        central_zenith=60,
    ).marks


def test_band_pass_keeps_a_drop_and_leaves_no_lobe_beside_an_excluded_object() -> None:
    y, x = grid()
    values = np.where(x < 40, -1.0, 0.0).astype(np.float32)  # a dark wall, left out
    values += 0.01 * np.exp(-((y - 60) ** 2 + (x - 80) ** 2) / 8).astype(np.float32)
    passed = band_pass(values, x >= 40, 1.2, 4.0)
    assert passed[60, 80] > 0.004
    assert np.abs(passed[:, 40:46]).max() < 1e-4  # no halo along the wall


def test_a_faint_drop_fixed_all_day_is_found_on_its_own() -> None:
    marks = run_faint(faint_windows([spot(0.015, radius=2)] * 9))
    assert marks[59:62, 39:42].all()
    assert marks.sum() < 60


def test_faint_marks_on_their_own_only_near_the_zenith() -> None:
    assert not run_faint(faint_windows([spot(0.015, radius=2)] * 9), zenith_deg=49.5).any()


def test_a_faint_spot_present_in_one_part_of_the_day_is_not_a_mark() -> None:
    flat = np.ones((SIZE, SIZE))
    marks = run_faint(faint_windows([spot(0.05, radius=2)] * 3 + [flat] * 6))
    assert not marks[50:70, 30:50].any()


def test_a_faint_line_grows_from_its_seed_and_not_without_one() -> None:
    windows = faint_windows([line(0.01)] * 9)
    assert not run_faint(windows)[58:63, 30:90].any()
    seeds = np.zeros((SIZE, SIZE), bool)
    seeds[58:63, 96:104] = True  # the strong head of the trail
    grown = run_faint(windows, seeds)
    assert grown[59:62, 25:95].any(axis=0).mean() > 0.9


def test_faint_marks_need_parts_of_several_hours() -> None:
    sky = np.ones((SIZE, SIZE), bool)
    windows = faint_windows([spot(0.015, radius=2)] * 8)
    assert faint_inputs(windows, [10] * 8, sky, ~sky, sigma=12) is None


def test_a_faint_drop_present_for_two_thirds_of_the_day_is_found() -> None:
    flat = np.ones((SIZE, SIZE))
    marks = run_faint(faint_windows([flat] * 3 + [spot(0.02, radius=2)] * 6))
    assert marks[59:62, 39:42].all()


def test_a_field_of_faint_drops_leaves_no_dark_mark_between_them() -> None:
    # The band-pass of a bright drop is negative around it; between several drops these lobes add
    # up. Measured against the sky without the drops, the gaps are sky.
    centres = [(50 + 9 * i, 30 + 9 * j) for i in range(3) for j in range(3)]
    field = np.prod([spot(0.03, c, radius=1.5) for c in centres], axis=0)
    marks = run_faint(faint_windows([field] * 9))
    y, x = grid()
    near = np.min([np.hypot(y - cy, x - cx) for cy, cx in centres], axis=0)
    assert all(marks[cy, cx] for cy, cx in centres)
    assert not marks[near > 3.5].any()


def test_a_faint_dark_spot_is_left_to_the_hourly_marks() -> None:
    # A dark pixel of the band-passed mean may be the side lobe of bright drops: no faint dark mark.
    assert not run_faint(faint_windows([spot(-0.015, radius=2)] * 9)).any()
