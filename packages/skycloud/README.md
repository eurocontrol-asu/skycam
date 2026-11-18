# skycloud

Continuous cloud map of a fixed all-sky (fisheye) camera. For every sky pixel of a frame:

- a **value** in [0, 1]: 0 clear sky, 1 thick cloud, graded in between (no classes, no threshold
  baked in: the user chooses one);
- a **confidence** in [0, 1]: how well clear sky and cloud can be told apart there and then;
- an explicit **unknown** state: no value where the final confidence is below 0.30.

No trained model. The method compares each pixel with reference tables of what clear sky and
overcast look like at that place and time, measured once on an archive of the camera's frames.
This version (c16) is a faithful port of the research prototype `clouds_v3` (16 correction
rounds), for the GVCCS site camera at Brétigny-sur-Orge; its reference tables are embedded.
The package sits after `skycam` (geometry) and `skystatic` (static masks) in the chain; the sun's
influence is isolated in `skycloud.sun`, the seam where the future `skysun` library will plug in.

## Method in plain terms

1. **Two cues per pixel.** The colour index N = (B − R)/(B + R) (clouds whiten the blue sky) and
   the brightness L = log((G − pedestal)/pedestal) in units of the camera's dark pedestal (the
   camera changes its gain, not its exposure, and the pedestal follows the gain, so L is
   gain-free). Both are read after an edge-preserving smoothing (guided filter on G): flat sky is
   averaged, thin lines and cloud edges keep their width.
2. **References.** Clear-sky and overcast medians and spreads of N and L, as smooth tables of
   (sun elevation, angle to the sun, zenith angle), interpolated continuously.
3. **The frame's clear-sky level.** The day's aerosols shift the clear colour: a level (a
   constant, two sun-side terms, a horizon-haze term) is found by robust mode seeking around the
   archive's clear sky, jointly on colour and brightness, and smoothed in time over the
   neighbouring frames. It cannot follow a sky far from the archive's clear sky, so an overcast
   is never taken for clear sky; a uniform thin veil close to it is absorbed (one camera cannot
   tell them apart).
4. **Gain ledger.** At twilight the camera steps its gain and the colour index jumps by
   0.01-0.02. Each step is measured on the unchanged sky and removed as a discontinuity (a cache
   per day, computed from the day's frames).
5. **Single-cue opacities and fusion.** Colour, brightness and texture each give an opacity and
   two precisions (of a cloud verdict and of a clear verdict); the value is their weighted mean.
   Brightness and texture speak where the colour cannot exclude a cloud. At low sun, an
   overcast-shaped sky has its colour opacity lifted towards 1, when the colour agrees.
6. **Low-information fill.** Where no cue decides, the value leans on the more informative sky
   around it, and the confidence is low.
7. **Confidence** = D / (D + 6) × the sun-artefact confidence × the agreement of the cues with
   the value, D being how separable clear sky and cloud are there. Near the sun the artefact cut
   is narrowed on homogeneous skies.
8. **Low sun.** Below +6.5° the diffuse sky is decided by a *knowability* K from the tables
   alone, smooth in space and time; cloud **structures** (sharp lines, mid-scale blobs, contour-
   bounded bodies) that persist in a ±30 s neighbour are certified and delivered inside an
   unknown diffuse sky. Below about +4° the map is a structure-only map, by design.
9. **Consistency cap.** Where the raw sky is unchanged 30 s earlier or later but the value
   switched, the confidence is capped at 0.29.
10. **Unknown state.** Pixels whose confidence (read with the ±30 s neighbours) is under 0.30 have
    no value; small fragments and holes are absorbed by their surroundings.

A frame's result depends on its neighbours in time (level smoothing, ledger, consistency, the
unknown decision): rendering a frame inside a different list can change it slightly.

## Inputs

| Input | What |
|---|---|
| frames | `ROOT/YYYY/MM/DD/image_<YYYYmmddHHMMSS>.jp2` (8-bit JPEG content; UTC timestamps) |
| geometry | azimuth and zenith maps (degrees, NaN outside the sky) as `.npy` or arrays of a `.npz` (`--azimuth`, `--zenith`); or `CameraGeometry.from_skycam(calibration)` with the `skycam` extra |
| site | latitude and longitude of the camera (sun position, NOAA ephemeris) |
| static masks | the skystatic daily products (`classes_<day>.png`, read through `skystatic.sky_mask`; class 0 is sky). A day without a mask is an error, never a fallback. Days whose skyline was not visible need `--allow-unreliable-mask` |
| reference tables | embedded for the GVCCS camera (`gvccs_bretigny_c16.npz`, SHA-256 checked at load); another camera's archive with `--tables` / `ReferenceTables.load(path, sha256=...)` |
| ledger cache | a folder; missing days are built on the first render (`skycloud ledger` builds them explicitly) |

## Products (one folder per run)

| File | Encoding |
|---|---|
| `<ts>.png` | value, uint8: value = code / 254 (0 clear .. 254 thick cloud); **255 = unknown** (or not sky) |
| `<ts>_conf.png` | published confidence, code / 255 |
| `<ts>_lens.png` | 255 = lens ghost found by the detector; the value there is left as measured |
| `_unmasked/<ts>.png` | the value before the unknown rule (255 only where there is no value at all) |
| `_known/`, `_cert/`, `_region/` | knowability K × 255, certified structure, diffuse-known region (low sun) |
| `_rings/<ts>.png` | lens-ring field (side layer, not a flag) |
| `_capped/`, `_uncapped/` | confidence after / before the consistency cap |
| `_neighbours/` | the ±30 s neighbours rendered for the temporal passes |
| `states.json`, `frames.json` | frame levels; per frame: sun elevation, level, gain-ledger note, consistency, factors, unknown share, means by zenith band |

## Command line

```bash
uv run skycloud render --list frames.txt --out out/ \
    --images /data/raw --mask-dir /data/skystatic/masks --ledger-dir /data/skycloud/ledgers \
    --azimuth geometry.npz --zenith geometry.npz --latitude 48.600518 --longitude 2.346795 \
    [--allow-unreliable-mask] [--exclude hidden.txt] [--workers 2]
uv run skycloud ledger 20240115 20240116 --images ... --ledger-dir ... --azimuth ... --zenith ... \
    --latitude ... --longitude ...
```

A render process peaks at about 3.5 GB; `--workers N` multiplies it.

## From Python

```python
from pathlib import Path
from skycloud import CameraGeometry, RenderSetup, Site, load_products, render

setup = RenderSetup(
    geometry=CameraGeometry.from_files(Path("geometry.npz"), Path("geometry.npz")),
    site=Site(latitude_deg=48.600518, longitude_deg=2.346795),
    images=Path("/data/raw"), mask_dir=Path("/data/masks"), ledger_dir=Path("/data/ledgers"),
)
report = render(["20240115155700"], setup, Path("out"))
p = load_products(Path("out"), "20240115155700")  # value, unmasked, confidence, lens
```

`render_frame` renders one frame from its inputs (no temporal pass); `CloudMapParams` holds
every parameter with its origin; `ReferenceTables` loads and looks up the tables.

## Building the tables for a camera

`skycloud build-reference` rebuilds the reference archive (`gvccs_bretigny_c16.npz` format)
from an archive of the camera's frames. **It has been run on one camera only, the GVCCS camera
at Brétigny**: there it reproduces the embedded c16 tables bit for bit from the prototype's stored
per-frame measurements, and its per-frame measurements reproduce the stored ones bit for bit
(checked on 3,741 frames of 29 days). Nothing has been built yet for another camera.

### What is needed

| Input | What |
|---|---|
| frames | a folder (`--images`, every `image_*.jp2` below it) or a list (`--list`: paths, or timestamps under `--images`). The fetch from a remote archive is not part of the library. Never an evaluation day: give them with `--exclude-days` |
| how many | the c16 build read 26,028 frames of 203 days (2023-10-05 to 2026-11-11, one day in five of three years): daytime frames every 10 min with the sun at or above +5.5°, twilight frames every 3 min from -7° to +8.5°. 264 of those frames were listed twice (once in each archive) and counted twice; the command refuses a duplicated timestamp unless `--keep-duplicates`. **The minimum number of days has not been measured.** The label-free labels need clear AND overcast skies at every sun elevation: a bin's archive percentiles need 40 frames of its elevation node, a table needs 8 clear (overcast) frames per node; a node without data copies the nearest node's tables |
| geometry | azimuth and zenith maps, as for `render` |
| static sky mask | one a-priori mask for the whole archive (`--sky-mask`, PNG or `.npy`; sky = `--sky-class`, 0 by default). The c16 build used the labelling kit's prior (`clouds/camera/sky_mask/static_mask.png`, class 0) |
| site | latitude and longitude (sun position) |
| carried tables | `c4_consistency` and `c5_ledger_params` are not measured by the build (see below): they are copied verbatim from an existing archive (`--carry ARCHIVE --carry-sha256 HEX`), and so are the two low-count tables unless dense days are given |
| dense days (optional) | `--calibration-list`: frames every 30 s of a few days, for the two low-count tables |

```bash
uv run skycloud build-reference --out gvccs_rebuilt.npz --work /data/skycloud-build \
    --camera "GVCCS site camera, Brétigny" --list archive_frames.txt \
    --sky-mask static_mask.png --azimuth geometry.npz --zenith geometry.npz \
    --latitude 48.600518 --longitude 2.346795 --exclude-days evaluation_days.txt \
    --carry gvccs_bretigny_c16.npz --carry-sha256 a5babedc...b933 \
    [--params overrides.json] [--until labels] [--workers 2] [--keep-duplicates]
```

From Python: `skycloud.reference_build.build_reference(BuildInputs(...), work, out, BuildParams())`.

### Steps

Each step writes into the work folder and records a fingerprint of its parameters and inputs:
an interrupted build resumes (the per-frame step day by day), a changed parameter re-runs only
the steps that depend on it, `--until STEP` stops after a step so it can be checked alone.

| Step | Reads | Writes | Prototype script (`clouds_v3/scripts/`) |
|---|---|---|---|
| `grid` | geometry, sky mask | `grid.npz`: the 16,200 grid directions (azimuth 2° x zenith 1°, nearest pixel) | `a2_features.grid` (kit `sky_auto.grid_pixels`) |
| `features` | frames, grid | `features/<day>.npz`: per frame and direction, 9 x 9 px window means of R, G, B over sky pixels, saturated share, sd of G; corner pedestal; sun position | `a2_features.py` |
| `binmeans` | features | `binmeans/*.npy`: per frame, the means of N, L, texture per (angle to the sun 2°, zenith 1.5°) bin, float16 | `b1_binmeans.py` |
| `labels` | bin means | `labels.npz`, `percentiles.npz`: clear / overcast / neither per frame (label-free rule) | `b2_labels.py` |
| `raw` | bin means, labels | `tables_raw.npz`: medians, robust sds, counts per node and bin | `b3_tables.py` |
| `smooth` | raw tables | `smooth.npz`: smooth complete Nc, No, Lc, Lo and spreads | `b4_smooth.py` |
| `adapted` | bin means, labels, smooth | `adapted.npz`: Nc_sda, Lc_sda (spread once the frame's level is removed) | `b6_spread.py` |
| `postfit` | features, labels, smooth | `postfit.npz`: Nc_bias, Nc_sdf (residual after the run-time joint level fit) | `c1_postfit.py` (replaces `b7_postfit.py`) |
| `spread` | smooth, adapted, postfit | `ref/c1_ref.npz`, `ref/c14_spread.npz`, `ref/grid.npz` | manual copy `v3_ref` -> `c1_ref`, `c14_spread.py` |
| `thresholds` | features, labels, tables | `ref/c7_shape.json`, `c8_release.json`, `c11_lift_colour.json`, `c9_knowable.json` | `c7_shape_stats.py`, `c8_shape_gate.py`, `c7_lowsun_levels.py` (c11 transcribed by hand at 2 decimals), `c9_knowable.py` |
| `midscale` | frames, labels, geometry, mask | `ref/c6_midscale.json` (600 random clear frames, full resolution) | `c6_midstats.py` |
| `lowcount` | dense days, or the carried archive | `ref/c3_lowcount.npz`, `ref/c4_lowcount_ped.npz` | `c3_lowcount_table.py`, `c4_lowcount_ped.py` |
| `carried` | the carried archive | `ref/c4_consistency.json`, `ref/c5_ledger_params.json` | `c4_consistency.py calibrate`, `c5_ledger.py calibrate` + `c5_sigd.py` (not ported) |
| pack | `ref/` | the archive, with its metadata (camera, site, period, frames, days, package version, every parameter, what was carried) | `tables.pack_reference_tables` |

Not measured by the build: `c4_consistency` (R0, D0) was calibrated on renders of the map
itself (the c4 version) on 30 s pairs of development days; `c5_ledger_params` on two dense days,
then `SIG_D` was set by hand from a split-half study. The development days of c16 (2023-10-12,
-21, 11-01, 11-22) are four of the ten evaluation days, so these two tables and the two
low-count tables were measured on evaluation days; another camera needs its own dense
non-evaluation days for them.

Cost on the GVCCS frames (2464 x 2056 px): the per-frame step takes 0.17 s per frame and core
(0.04 s of it decoding) and 45 MB above the process; building the grid peaks at 0.6 GB once.
A full rebuild of the 26,028 frames is estimated at about 50 min with `--workers 2` (85 min on one
core), 23 GB of frames read, 5 GB written in the work folder; the 29-day test build (3,741 frames)
took 11 min with two workers.

### Camera-specific parameters

All are fields of `BuildParams` (JSON overrides with `--params`) unless marked *renderer*.

| Parameter | Value (c16) | How it was obtained here | On another sensor |
|---|---|---|---|
| grid: azimuth / zenith step, window | 2°, 1°, 9 x 9 px | design (the labelling kit's grid); 9 px is about 1° at 9.3 px/deg | keep the steps; scale the window to about 1° |
| `window_min_sky` | 70 of 81 pixels | design | follows the window |
| `saturation`, `bright_max` | 250, 245 counts | the 8-bit JPEG camera | the sensor's saturation |
| pedestal corners | four 60 px squares | the corners outside the fisheye circle (*renderer*: `images.pedestal`) | must lie outside the image circle; fixed in the renderer |
| `PED_GR` (G/R of the pedestal) | 0.545 | *renderer* constant (`cues.PED_GR`): "median of the corners at gains where they are well measured", no script found; the archive corners give 0.537 (R corner >= 60) to 0.550 (all frames) | re-measure; the build reports the archive's G/R and B/R (`binmeans` step) and warns at 0.02 from the constants, but the renderer's constant cannot be changed by the build |
| `PED_BR` | 0.882 | median corner B/R of archive frames with R corner >= 60 (c2 study); used by the low-count table in pedestal units | re-measure |
| `elevation_nodes`, bin sizes | 27 nodes -4..65°, 2° x 1.5° | design | keep |
| `labels.min_gap` | 0.03 | set in `b2_labels.py`: bins where the colour index separates clear from overcast | depends on the sensor's colour response |
| `labels.q_clear`, `q_overcast` | 85th, 10th percentiles | design | depends on the climate of the archive (share of clear and overcast skies) |
| `labels.clear_q90_max`, `clear_q50_max`, `overcast_q10_min` | 0.35, 0.15, 0.65 | set in `b2_labels.py`; no study of them was found | check the labels on the new archive |
| `labels.episode_below`, `episode_window`, `episode_share` | 6°, 6-20°, 80 % | design: the colour cue is weak below 6° | keep |
| `labels.half_day_split_utc_hour` | 12 | a site property: solar noon is about 11:50 UTC at 2.35° E | the UTC hour of solar noon |
| `labels.node_min_frames`, `smooth.table_min_frames`, `smooth.count_cap` | 40, 8, 60 | design | keep; a small archive leaves bins empty (filled from neighbours) |
| smoothing sigmas | see `SmoothParams` | design (`ref_deg` = 3°) | keep |
| level fit (`CloudMapParams`: `s_n`, `prior`, `b_sd`, `b_lim`, `al_lim`, `con_min`) | renderer defaults | *renderer*: `al_lim` and `b_sd` were measured on this archive's clear frames (`c1_level_stats.py`), not by the build | `c1_level_stats.py` is not ported: re-measure by hand |
| other archive-measured renderer parameters | `f_half_bands`, `amb_j0`, `k_gain`, `amb_s` | *renderer*: measured on this archive or on development days | not re-measured by the build |
| threshold tables c6, c7, c8, c9, c11 | in the archive | measured by the build | measured by the build |
| `c4_consistency`, `c5_ledger_params`, low-count tables | in the archive | development days (evaluation days) | dense non-evaluation days; c4/c5 calibrations are not ported |

## Tried and not kept

Switches that were OFF in c16 are not ported (one-line reason from the prototype's comments):

| Experiment | Round | Why it was not kept |
|---|---|---|
| split shrinkage of the colour opacity (`c1_split`) | c1→c2 | caused the c1 high-sun contrail contrast regression |
| detail weighted by its local energy (`c2_energy`) | c2 | experiment, no gain |
| detail shrunk by the formal pixel noise (`c2_s_formal`) | c2 | experiment, only with the split shrinkage |
| mid-value confidence penalty without own support (`c2_midconf`) | c3 | took the confidence of correctly read overcasts (R3-2) |
| thin-band confidence penalty (`c3_midconf`) | c5 | band from the rejected 90 images, iso-line loops (R4-3) |
| low-count spread in pedestal units in the value (`c4_lowcount_ped`) | c4 | made the gain pairs worse (kept in the confidence below +10°) |
| no value where d′ < 1 (`c5_floor`) | c6 | replaced by the confidence-based unknown state |
| post-fit colour bias table (`c7_bias`) | c7 | read the cloudy horizon of 2024-01-15 11:40 as clear (0.19) for a small gain on clear skies |
| frame-anchored level release (`c8_release`) | c8 | released almost none of the target skies (eval/c8/gate.md) |
| shape rule on the dome-scale part only (`c8_diffuse`) | c8 | off after test |
| shape rule kept on beyond the clear range (`c8_persist`) | c8 | 04-12 dawn stretch unchanged from c7 |
| frame ambiguity factor of K (`c9_amb`) | c9 | 80 % of the dome unknown at +4-6°, whole frames flipping |
| hysteresis front of K (`K_FRONT`) | c11 | left the 30 s flips unchanged |
| hole filling of the diffuse region (`WF_FILL`) | c16 | made the largest 30 s flips |
| colour-only level fit, pedestal-corrected index (`joint` off, `cues_c2`) | c1-c2 | joint fit adopted; the corrected index was less separable |

Diagnostic options of the prototype (`--no-adapt`, `--no-context`, `--no-consistency`,
`--states-from`, `--extra`, `--zero-terms`, environment overrides) are not ported either.

## Known limits

See the prototype's review (`clouds_v3/README.md`, "Known limits"): low-sun contrails and cirrus
bands left unknown below +3.5°, diffuse sky unknown below about +4° by design, geometric shapes of
the unknown boundary at +2.5-4.5°, overcast read 0.6-0.8 near the zenith at +5-12° on some days,
lens reflections kept at their measured value (flagged), a uniform thin veil reads as paler clear
sky.
