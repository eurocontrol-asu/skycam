# skysun

Where and how much the sun disturbs each frame of a fixed all-sky (fisheye) camera. For every
pixel of a frame, two values in [0, 1], delivered as two separate layers, each on its own scale:
the **sun zone** (the sun's glow) and the **reflections of the optics** (the rings the sun lights
inside the lens). Always a value in both layers: there is no "unknown" state and no special code,
and there is no combined map (decision of the camera's owner, 2026-11-18). Third mask of the
chain `skycam` (geometry) → `skystatic` (static masks) → **`skysun`** → `skycloud` (cloud map);
made to down-weight training labels where the sun disturbs them.

No trained model. Inputs: the frame, its UTC time, the azimuth and zenith angle of every pixel,
the camera site, and skystatic's products (the sky pixels of the day, the ring circles of the
optics).

## The two layers

1. **The sun zone**, measured on each frame. Its value is the brightness the sun's glow adds
   there as a share of the headroom between the sky's own level and saturation (0 nothing, 1
   saturated). Around the computed position of the sun, in annuli of 0.5 degree, the darker
   quarter of the sky pixels gives the sky level of the annulus (clouds near the sun, brighter
   than the sky beside them, do not shape it). The glow is that level above the sky's own level
   (the annuli 30-40 degrees from the sun) over the headroom to saturation; the saturated share
   of the annulus is taken when larger. The profile is forced non-increasing outward, smoothed
   over one annulus and faded to 0 across the 30-40 degree annuli (no edge). It is multiplied by
   whether the sun shines (below), so it is absent when the sun is hidden. The sun's saturated
   pixels (its disc, glare and star spikes: saturated pixels connected to one within 3 degrees
   of the computed sun) read 1. In a run (`skysun run`), the profile of a frame whose sun shines
   (visibility >= 0.9) is the median of the profiles of the frames within ±60 s whose sun also
   shines: clouds passing inside the glow otherwise make the whole zone swell and shrink between
   30-s frames. The frame's own visibility and saturated pixels are always used, and the map
   itself is never smoothed, so an appearance or disappearance of the sun is not blurred.
   `sun_map` (one frame alone) has no temporal median.
2. **The reflections of the optics**, predicted. Its value is the predicted contrast of the
   reflection (its light over the sky beside it) over the contrast of this optic's strongest
   reflections: value × reference = contrast, clipped at 1. Reference for the site camera:
   7.47 %, the 99th percentile of single-frame sector contrasts over the clear calibration
   frames where the sun fully shines, the seven rings pooled, the eye left out (its sectors
   reach 90 % when a ghost spot lands on it); stored in the table. skystatic fits seven ring
   circles (each with its own centre) and a small central "eye" once; their position never
   changes, their light depends on where the sun is and whether it shines. Each is cut into 36
   sectors about its own centre. A table, calibrated once on clear archive frames, gives the
   median contrast of a sector against its direction relative to the sun's direction (both seen
   from the ring's centre in the image) and the sun's elevation. The prediction is the table
   times whether the sun shines, painted along each line (±3 px, tapered over ±1 px). A cloud
   edge crossing a ring cannot light it in the prediction. A predicted contrast below 0.3 % is
   negligible (0.04 on the layer's scale).

**Whether the sun shines** (0..1, the visibility, multiplies both layers) is read from the zone's
core: the pixels of the field within 2 degrees of the computed sun, static or not (a mast in
front of the disc does not hide the saturated pixels around it). 0 when none is saturated (the
sun's disc does not reach the lens), 1 from the saturated share that 95 % of clear archive
frames of the same sun elevation reach (0.33 at +5-10 degrees, 0.50 at +10-15, 0.66 at +15-20,
0.71-0.84 higher: a low sun saturates less of its core), linear between. When less than half of
the core is static sky (the sun behind the tree line or the installation), the core says
nothing; then the glow of the profile over the sky beside decides: its mean value 5-10 degrees
from the sun, 0 at 0.195 or below, 1 at 0.260 or above (measured below +20 degrees of sun on
hidden-sun overcast frames and on clear frames with a shining sun).

When the sun is below the horizon (geometric elevation <= 0) every pixel of both layers is 0.

## Use

```bash
uv run skysun run FRAMES_OR_FOLDERS... --geometry geometry.npz --static SKYSTATIC_OUT \
    --latitude 48.6005 --longitude 2.3468 --output out/
```

- `--geometry`: one `.npz` with `azimuth_deg` (from north through east) and `zenith_deg` per
  native pixel, NaN where a pixel sees no sky.
- `--static`: a skystatic run folder (one daily mask per date and `optics.json`; the daily mask
  of the frame's date is required, as in `skystatic.sky_mask`) or a frozen installation NPZ
  (its classes for every date of its validity, the circles inside).
- `--table`: a reflection table NPZ; default the one embedded for the GVCCS site camera.
  `--no-reflections` gives the zone only (a camera without a table).
- Frame names hold a `YYYYmmddHHMMSS` UTC timestamp. Frames are read one at a time in time
  order; the measurements (not the pixels) of the frames within ±60 s are held for the temporal
  median. A frame whose JSON exists is skipped, so an interrupted run resumes (the frames around
  the resume point are measured again for their neighbours). Unreadable files are reported and
  skipped.

Written per frame under `out/YYYY-MM-DD/`:

| File | Content |
|---|---|
| `zone_<stamp>.png` | the sun zone: 8-bit grayscale, code = round(255 × value), value = code / 255 |
| `reflections_<stamp>.png` | the reflections on their own scale, same encoding; value × `reflections_reference_contrast` = contrast |
| `sun_<stamp>.json` | the frame's numbers: sun azimuth and elevation, the sun's pixel, the sky's own level, the core's saturated share and sky share, which reading decided the visibility (`core`, `glow`, `night`), the visibility, the number of frames whose median profile made the zone, the saturation radius, where the zone falls below 0.5 / 0.2 / 0.05, the strongest reflection and the reference contrast, the table's checksum |

From Python, the same functions (`sun_map` = `render(measure_frame(...))` for one frame alone;
`products.run_frames` adds the temporal median of the profiles):

```python
from pathlib import Path
from skysun import CameraGeometry, Site, SunCamera, packaged_table, sun_map
from skysun.frames import frame_time, read_rgb
from skysun.static import StaticSource

geometry = CameraGeometry.from_file(Path("geometry_gvccs.npz"))
static = StaticSource(Path("skystatic_out"))
camera = SunCamera(geometry, Site(latitude_deg=48.6005, longitude_deg=2.3468),
                   static.optics(geometry.size), packaged_table())
path = Path("image_20240115110000.jp2")
result = sun_map(read_rgb(path), frame_time(path), camera, static.sky(frame_time(path).date(), geometry.size))
result.zone, result.reflections, result.info
```

### Calibrating the reflection table (once per camera)

```bash
uv run skysun measure ARCHIVE/YYYY/MM/DD ... --geometry geometry.npz --static installation.npz \
    --latitude LAT --longitude LON --labels labels.npz --output measure/ --workers 4 \
    --exclude-day 2023-10-12 ...            # evaluation days: refused before anything is read
uv run skysun build-table measure/ table.npz --camera-id CAMERA --width 2464 --height 2056
```

`labels.npz` holds `ts` (YYYYmmddHHMMSS) and `lab` (1 clear, 2 overcast, 0 neither) of archive
frames; only labelled frames with the sun above +3 degrees are measured, one file per day
(resumable). The build sets the visibility ramp (tops per elevation bin, glow fallback), keeps
clear frames where the sun fully shines, takes per ring, 5-degree elevation bin and relative
sector the median contrast, fills an elevation bin with fewer than 10 frames from the nearest
filled one, takes the reference contrast (99th percentile of ring sector contrasts, eye left
out), and seals the table with a SHA-256 over its arrays and metadata (circles, ramp, reference,
frames, days, period, selection, command).

## Parameters and their origin

| Parameter | Value | Origin |
|---|---|---|
| annuli | 0.5 deg, up to 45 deg | prototype `proto/zone.py`, accepted by the owner on eight frames (2026-11-18) |
| sky level of an annulus | 25th percentile | the definition agreed with the owner: the darker quarter, so that clouds do not shape the profile |
| annulus measured | >= 40 pixels, else interpolated | prototype |
| sky's own level | median of the 30-40 deg annuli (>= 5 of them, else of all) | prototype; `sun_spec.md`: on clear frames the excess over the clear-sky reference is 0 ± 0.05 beyond 8 deg |
| fade | raised cosine from 30 to 40 deg | the zone is measured against the 30-40 deg level: nothing there is the sun's, and the prototype's cut at 40 deg left an edge |
| saturation | a channel >= 250 | prototype; top of the headroom |
| smoothing | Gaussian, 1 annulus | prototype |
| sun's saturated pixels | connected to a saturated pixel within 3 deg of the computed sun | `sun_spec.md`: the saturated core reaches 4-5 deg (p90 5-6) on clear frames |
| core | 2 deg | `sun_spec.md`: saturated share of the 0-2 deg ring 0.79-0.96 (median) on clear frames, 0.00 under overcast |
| core read on | every field pixel within 2 deg | review S1: a mast in front of the disc |
| core mostly not sky | < 50 % static sky | design choice: a majority (review S1) |
| visibility ramp (core) | 0 at no saturated core pixel, 1 from a top per 5-deg elevation bin (0.33 ... 0.84) | archive (table metadata): 5th percentile of the core's saturated share over clear frames whose core is mostly sky and holds a saturated pixel; bins with < 30 such frames take the nearest (review S4) |
| visibility ramp (glow) | mean zone value 5-10 deg: 0 at 0.195, 1 at 0.260 | archive, sun below +20: 95th percentile over 954 hidden-sun overcast frames, 5th over the clear frames with a shining sun (the core is mostly not sky only at +6.3 to +11.4, p10-p90) |
| temporal median | profiles within ±60 s, visibility >= 0.9 | review S2; measured on four development days: a single-frame reference (pooled 30-40 or 30-45 deg, lowest annulus) or a ±60 s median of the reference level alone left the share of failing sun-stable pairs at 5.6-6.0 % (proxy), the profile median brought it to 1.6 % |
| reflections reference | 7.47 % contrast = 1 | archive (table metadata): 99th percentile of 550 000+ ring sector contrasts on clear frames with a shining sun, eye left out (Addendum 2 of the criteria; the review measured 7.4 %) |
| ring band | ±3 px (native) | skystatic: ±1.5 px at width 1232, the measured line width |
| ring flanks | 7-13 px from the line, other rings' bands left out | prototype `proto/step0.py` (a linear gradient cancels) |
| painted taper | full to 2 px, 0 at 4 px | no step at the band's edge |
| sectors | 36 (10 deg) | prototype `proto/step0.py` |
| sector measured | >= 50 % of its band and flanks usable | design choice: a majority |
| eye | disc of skystatic's eye radius, flanks 4-10 px beyond | the same 6-px flank as the rings |
| elevation bins | 5 deg, 0-70 | the site's highest sun is +65 |
| table cell | median contrast over the frames | robust to a cloud crossing a ring on some frames |
| sparse elevation bin | < 10 frames: nearest filled bin | design choice |
| negligible | predicted contrast < 0.3 % (0.04 on the reflections layer) | skystatic: the clean optics' own texture (0.29-0.31 %) and its ring level (0.31 %) |

## What it was checked on

2026-11-18, GVCCS site camera, against the criteria fixed before the work and the review of the
first build (run folder `contrail/runs/20261118-skysun`, `kcheck_v2/`; frames chosen from the raw
contact sheets before any map):

- **Calibration**: 7130 labelled archive frames measured (203 days, 2023-10 .. 2026-11, none of
  the ten evaluation days); table from 2192 clear frames where the sun fully shines, on 109
  days; 8 structures × 13 elevation bins × 36 sectors; reflections reference 7.47 %.
- **Zone** (22 clear or cirrus frames, sun +12 to +59): 0.5 at 4.75-8.75 deg (median 5.75), 0.2
  at 10.75-21.25 (median 12.75), 0.05 at 23.75-31.25 (median 28.25). Sun hidden (16 overcast
  frames): at most 0.008 outside saturated pixels (visibility <= 0.013). Largest step between
  neighbouring pixels outside the sun's saturated pixels: 0.035.
- **Sun behind the tree line or on a mast** (review S1): 2024-05-10 05:00-05:30, 61 frames,
  visibility below 0.5 on 45 before, 0 after (zone at 10 deg: median 0.27); 2024-01-15
  15:05-15:15, 21 frames, 9 before, 0 after; 2024-03-18 06:35-07:00, 51 frames, 44 before, 29
  after (from 06:42 the disc is behind broken cloud as well as the trees: the core, mostly sky
  again from 06:53, is not saturated).
- **Reflections**: 1465 ring sectors measured at >= 1 % contrast on those frames: 88.4 % carry a
  predicted contrast >= 0.3 %, Spearman 0.58 pooled. Sun hidden: at most 0.012 on the layer.
- **Always a value**: 10 765 readable frames of the six development days, two layers each, codes
  in 0..255, night frames 0; one corrupt file reported and skipped.
- **Time** (pairs 30 s apart, visibility changing by < 0.05, share of the static sky domain
  changing by > 0.1): zone 22 of 6520 pairs over 1 % (0.34 %; 0, 1.4, 0, 0, 0.44, 0.17 % on
  2023-09-27, 2023-10-21, 2023-11-01, 2024-01-15, 2024-03-18, 2024-05-10), worst 3.5 % of the
  sky; before the temporal median 109 of 6595 (1.65 %), worst 9.3 %. Reflections: 0 pairs.
- **Cost**: 0.43 GB peak per process; 6 days in parallel at 0.45-0.75 s per frame each.

## Limits

- **Reflections above +35 degrees of sun**: on clear calibration frames the rank agreement
  between a frame's own sector contrasts and the table (per-frame Spearman, median) is 0.68 at
  +10-35, 0.67 at +35-50 and 0.53 at +50-65 (review, in-sample). On the one high-sun development
  day (cirrus) it falls to about 0.1-0.25, which the review attributes to cirrus texture (1-1.5 %
  noise in a single frame's sector measurement), not to the model.
- **Lens ghosts**: the chain of small bright spots on the line from the sun through the image
  centre (small discs at low sun, many aperture-shaped spots at high sun, one larger ellipse
  opposite) is not modelled, except where it falls on the eye disc: median contrast about 9 %,
  at most about 0.05 % of the sky per sunny frame, about half of it outside the zone (review).
- **Sun below +6 degrees**: not assessed. The glow lies on the horizon and the installation;
  both layers stay in [0, 1].
- **Sun behind the tree line**: the visibility then comes from the glow over the sky beside;
  the reflections are multiplied by it too, although the disc itself is behind the trees
  (whether the rings are lit then is not verified).
- **Clouds inside the glow of a shining sun** still move the zone between 30-s frames after the
  temporal median: 0.34 % of sun-stable pairs, 1.4 % on the broken-cloud day 2023-10-21.
- **The glow fallback cannot tell a low sun shining behind the trees from a low sun behind the
  trees and a cloud**: on the development days it decides 1032 frames (sun 0-12 deg), 404 of
  them at visibility 1; some (2023-11-01 07:00-07:50, 2023-10-21 06:24-07:42) are broken or
  overcast mornings where the frame does not show whether the disc shines.
- Dust lit by the sun for a few hours is not modelled.
- The reflection table is for the GVCCS site camera and its seven rings; another camera needs its
  own skystatic circles and its own calibration.

## Development

This package lives in `packages/skysun` of the skyobs workspace (Python 3.12+, development 3.14).
Tests live in `tests_skysun/{unit,integration,e2e}` on a synthetic 400 × 400 camera.

```bash
uv run --directory packages/skysun pytest
uv run --directory packages/skysun ruff check . && uv run --directory packages/skysun ruff format --check .
uv run --directory packages/skysun mypy src
```
