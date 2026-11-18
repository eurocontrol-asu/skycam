# skystatic

Static parts of a fixed all-sky camera, found from averaged images: the **installation**
(trees, buildings, masts, antennas) and the **variable static** of each day (dirt or drops on
the optics, anything that stays put for a day but not for months).

No trained model. Inputs: the frames of several days and the zenith angle of every pixel
(camera calibration).

## Method

1. **Daytime frames only.** A frame is daytime when its fisheye circle is visible: the sky
   around the zenith is at least twice as bright as the part of the frame beyond the horizon,
   in at least one colour channel.
   A flat frame has ratio 1 (the GVCCS site camera writes a flat magenta frame at night,
   brighter in its centre than the day, so brightness cannot tell day from night).
2. **Average the day, frame by frame**, in three consecutive parts (colour, for the edge test)
   and in clock hours (luminance, for the marks). What moves (clouds, the sun) is smoothed out;
   what is fixed in the image keeps its place.
3. **Static by edges.** On the log image, an edge is a local excess of gradient over the median
   gradient of its surroundings. The sky is the smooth region connected to the zenith; every
   other pixel of the domain is static. A pixel must be static in all three parts.
4. **Static by local contrast (marks): dirt, drops, thin rods.** In each hourly average the local
   contrast is log(luminance) minus a smooth sky (a quadratic surface fitted on sky pixels under
   a Gaussian window of 12 px), in units of that hour's sky noise at the pixel's zenith angle. A
   mark exceeds 3 noise units, whatever its sign, in more than half of the day's hours: the same
   dirt is dark in the morning and bright once the sun lights it, but a cloud, the sun's track or
   its glare does not come back to one pixel hour after hour. An outline (gradient above 3 noise
   units) that stays put in most hours and touches what is already static counts too (thin
   masts). With the camera's azimuth map and site, an hour in which the sun comes within 20
   degrees of a pixel cannot vote for it, and a pixel judged in fewer than half of the day's
   hours is left undecided. Marks are grown to half their mean significance. A mark that hugs the
   skyline (touches the installation beyond the core zenith angle) and changes sign with the
   lighting over most of its pixels (significant hours of both signs, the other sign at a fifth
   of the strongest amplitude or more) is the glow of a lit roof or wall, not a mark.
5. **Rings of the optics: each its own circle, fitted once on the reference days.** The
   reflections inside the camera are thin bright circles, not exactly concentric. On the mean
   contrast of the reference days the common centre is the point that makes the radial profile
   sharpest; a ring is a circle along which the mean contrast peaks, over centre (within 4 px of
   the common one) and radius, above twice the profile's noise; it is then measured on its own
   (the radius of its maximum in each of 24 azimuth sectors, sectors crossed by a ring already
   found left out, a least-squares circle through those maxima) and kept when a third of the
   sectors show it. A line of the common radial profile (maximum above twice its noise, sign in
   more than half of 8 sectors) that cannot be measured on its own keeps the common centre. Every
   ring is masked +-1.5 px about its own centre as installation, every day; two rings whose
   bands come closer than the masking width (3 px) anywhere are one band, from the inner edge of
   the inner one to the outer edge of the outer one (no sliver of sky lit by both). There is no ring
   "of the day": measured on each review day, every reference circle holds its ring (the
   maximum of 53-100 % of the usable sectors within +-1.5 px of it, on clean and dirty days
   alike). The rings are left out of the faint marks, and within the band-pass's reach of each
   ring (its width plus 4 px) the ring's own light is removed first (median of each arc of
   radius and azimuth sector about the ring's own centre), so that a mark crossing a ring is
   measured up to the line and nothing is removed elsewhere.
6. **Faint bright marks by hysteresis (two thresholds and connectivity).** Dust, drops and a
   water trail of 0.5-1.5 % are under 3 noise units in most single hours, but they stay put. The
   day's mean and the means of its three parts (consecutive groups of at least three hours) are
   taken as local contrast and band-passed at the scale of a drop or a thin line (difference of
   Gaussian means of 1.2 and 4 px, over the sky only). Their noise is temporal: the spread of the
   differences between parts, in which everything fixed cancels and what moves remains. A *weak*
   pixel is brighter than its surroundings by at least 0.3 % and 2 temporal noise units in the
   day and in all three parts (or the same on two parts, each at 2 units); it is kept when
   connected, through weak pixels, to a bright mark (a trail grows from its head, a drop's faint
   skirt from its core). A faint mark with no seed reaches 0.5 % and 4 noise units, with 2 units
   in every part, over the whole day or over two of its parts, the latter only where the sun
   stayed 20 degrees away from the pixel in both. The daily mask holds a mark all day, even if it
   was seen for two thirds of it. Only BRIGHT faint marks: the band-pass of a bright drop is
   negative all around it (its side lobe; between the drops of a field the lobes add up), so a
   dark pixel of the band-passed mean is no evidence of a dark mark; dark dirt must be a mark
   hour by hour (step 4). Faint marks are looked for within the core zenith angle (60 degrees).
7. **Installation = static on at least 60 % of the reference days**, grown by hysteresis through
   what is static or weak (either sign) on at least 60 % of them and attached to it (the thin top
   of a mast, faint on most single days, joins its base and is class 1 every day). The
   references come from the period of the processed days (median date within a year of every
   processed day).
8. **Variable static of a day = its static pixels minus the installation**, with 2 px of slack
   (a mark mostly beyond the slack is kept whole). Sky that is not
   connected to the open sky (across the ring lines) takes the class that surrounds it. A day's
   classes depend on that day and on the reference days only, never on the other days of a run.

So a pixel enters class 2 by one of: the edge test (step 3), the hourly marks (step 4, minus the
skyline glow), a faint
bright mark (step 6: a stand-alone mark over the whole day or over two sun-free parts, or a weak
pixel attached to a bright mark), or enclosure (step 8). The rings of the optics (step 5) are
part of the installation (class 1) on every day.

### Parameters and their origin

| Parameter | Value | Origin |
|---|---|---|
| `daytime_contrast` | 2 | flat frame = 1 in every colour channel; daytime frames of the site camera 2.2-4 (5th-50th percentile); taken in the channel where it is largest, because the camera paints the surround magenta on some daylight frames |
| `daytime_relative` | 0.25 x p95 | only for cameras with no pixel beyond the horizon (fallback); twilight is several times dimmer than daylight |
| `parts` | 3 | design choice: the sun moves ~45 degrees of azimuth per third, so its track does not overlap between parts |
| `core_zenith` | 60 degrees | site camera: nothing fixed stands above 30 degrees of elevation but one antenna tip (zenith 53-58) |
| `edge` | 0.05 | measured: sky around the zenith has a 99th percentile of 0.01-0.02 (overcast parts), the installation border a median of 0.06-0.11 |
| `background` | 36 px (>= 15) | design choice: several edge widths, small against the horizon brightening |
| floor / sigma of the edge test | 0.1 / 1 px | design choice: below a tenth of the sky level a pixel is dark object; a 1 px blur removes pixel noise without widening edges |
| `window_minutes` | 60 | design choice: the sun moves ~15 degrees per hour; a day holds 6-15 hours |
| windows kept | >= half the median frame count | dawn and dusk stubs hold too few frames to be averages |
| `mark_k` | 3 | the window's own noise: Gaussian noise passes one window with probability 0.003, a majority of 6 or more windows with less than 1e-12 |
| majority | more than half of the judged windows | clouds and the sun do not come back to one pixel in most separate hours |
| judged windows | >= half the day's, >= 3 | the sun near a pixel for most of the day leaves too little to decide on |
| `faint_scales` | 1.2 / 4 px | band-pass of the faint marks: the smallest drop and dust discs are 5-7 px across (measured), a trail 1-3 px wide; the outer scale removes the smooth sky and slow cloud residue |
| `faint_weak` | 0.3 % | weak level: the largest fixed texture of the clean optics in the band-passed day mean is 0.29-0.31 % within 60 degrees on the quiet reference days (2024-09-05, 09-25, 10-15) |
| `faint_weak_k` | 2 | weak level in temporal noise units (also the level every part must reach for a stand-alone faint mark); safety comes from connectivity to a mark |
| `faint_alone` | 0.5 % | a faint mark with no seed stands above the clean optics' texture (0.3 %) with margin; 0 disables faint marks |
| `faint_alone_k` | 4 | the same in temporal noise units |
| faint zone | `core_zenith` (60 degrees) | where the sky is certainly open (the core's own origin); no separate parameter |
| hours per part | >= 3 | a part must span the ~45 degrees of sun azimuth for which the three parts are designed (`parts`; the sun moves ~15 degrees per hour) |
| spans of two parts | sun >= `sun_clearance` all of both parts | the same guard as the hourly marks: a mark judged on part of the day must have been seen with the sun far. A single sun-free part is not enough: on the 10 reference days (clean optics), one part at the stand-alone level with its sign in 2/3 of its hours admits 35-1 733 px on 7 of the 10 days |
| faint marks: bright only | sign of the band-passed day | the band-pass (1.2 / 4 px) of a bright disc of 2-4 px is negative from r 5 to 9 px around it (its side lobe), and the lobes of a field of drops add up. On the 10 reference days the dark faint marks (rules of round c7) lie within 9 px of a bright mark for 87 % of their 14 001 px, against 7.5 % of the open sky |
| `mark_opposite` | 0.2 | what "changes sign" means for the skyline glow: on the 10 reference days the hourly marks that change sign keep 0.22 of their strongest amplitude in the other sign at the 5th percentile (median 0.49, 42 marks) |
| skyline glow | touches the installation (slack + 1 px, rings excluded), median zenith beyond `core_zenith`, both signs on half of its pixels | no new level: the core zenith angle (where nothing fixed stands but dirt), the mark's own sign changes (the same counted hours as the marks), a majority of its pixels |
| merged ring bands | closest approach of two circles < 2 x masking width | geometric: the bands then leave less than one masking width of sky between them |
| ring removal | median per arc (0.5 px of radius, 16 sectors) about each ring's own centre, within `ring_halfwidth` + the outer scale (4 px) of each line, nowhere else | a ring is the same along its circle; sectors absorb a centre error of a pixel; a trail or drop fills a small part of an arc; near the centre the arcs are too short to hold a ring's median (it removed drops there in round c6) |
| ring lines left out | `ring_halfwidth` (the measured line width) | the line itself is masked; nothing is decided under it |
| own centre of a ring | within 4 px of the common centre; maxima in 24 sectors, kept when 8 show it | measured on the reference days: the rings' own centres lie 0.5-1.8 px from the common one (one ring, r 109.5, at (+1.2, +0.7)); a ring that merges with a neighbour over part of its circle (that ring over about a third) still shows its own maximum in the other sectors, a spot in one or two |
| installation growth | 60 % of the references | the same majority as the installation vote, on "static or weak"; 50 % added 1 337 px of edge halo in 545 pieces (measured) |
| `mark_sigma` | 12 px | larger than the largest mark on the optics (15 px), small enough for a quadratic to follow the horizon |
| `mark_band` | 5 degrees | design choice: the sky noise grows towards the horizon |
| `sun_clearance` | 20 degrees | measured on the 10 reference days: within 20 degrees of the sun the share of significant sky pixels in an hourly mean is an order of magnitude above the far field (26x at 0-5, 9x at 15-20, 7x at 20-25, 2x at 40-45, 1x at 60 degrees); at 45 degrees the fixed dirt spot D1 (in the installation) is judged in too few hours, i.e. undecided, on 5 of the 10 reference days (0 at 20 degrees) |
| `tolerance` slack | whole marks | a static piece mostly beyond the 2 px slack is kept whole (no sky ring inside a mark touching class 1); a piece mostly inside it is edge jitter |
| half-peak growth | 1/2, never below `mark_k` | the blurred rim of an out-of-focus spot |
| `ring_halfwidth` | 1.5 px | measured width of a ring line (~3 px) |
| ring line level | 2 x profile noise | the profile's own noise (robust standard deviation over radii); the eye: the central disc while its mean profile stays above that level |
| enclosed sky | none | sky not connected to the open sky is not sky (one rule, no threshold) |
| `ring_level` | 0.31 % (level mode) | user's choice of depth: 0.31 % ~ 4 % of the sky, 0.23 % ~ 12 % |
| ring sectors | sign in > 4 of 8 sectors | a ring is a circle; a spot on one radius is not |
| `min_days` | 0.6 | design choice: a clear majority of >= 10 reference days |
| `reference_window` | 365 days | design choice: one seasonal cycle; references describe the period of the days they serve |
| `tolerance` | 2 px | design choice: an edge moves by a pixel or two between days |
| `outlier_ratio` | 0.5 | usable days have static fractions of 11.5-14 %; a lost skyline halves it |

Memory: frames are decoded by a bounded pool and accumulated one by one (parts, hourly windows);
one day of the site camera peaks at about 2-3 GB (it was 17 GB before frames were bounded); a run with
`--jobs 2` peaked at 2.8-4.4 GB (3.2-3.5 GB with the faint marks, cache reused).

## Use

```bash
uv sync
uv run skystatic run DAY [DAY ...] --zenith zenith.npz --output out/ \
    [--reference REF_DAY ...] [--cache cache/] [--pattern '*_160.jpg'] [--jobs 2] \
    [--azimuth azimuth.npz --latitude LAT --longitude LON]
```

A day is a folder of images or a zip archive of images; file names must hold a
`YYYYMMDDhhmmss` timestamp. The day label is the ISO date of those timestamps,
regardless of the directory or archive name (for example `2023/10/12` becomes
`2023-10-12`). Empty sources, sources spanning several dates, and duplicate day
labels or dates fail before any cache lookup or output write. The same source
may be both a processed day and a reference. Explicit Python labels must also
be unique. A run describes one camera.

Caches identify the source path, selected images, file revisions and aggregation
parameters. Old caches keyed only by a folder name are not reused. Removing or
modifying images invalidates their cache; a cache is not an archive of the inputs.

The zenith map is a `.npy`, or one array of a `.npz`
(`--zenith-key`, `--zenith-unit deg|rad`), at the native size of the frames or any size with the
same aspect ratio.

Written to `--output`:

| File | Content |
|---|---|
| `installation.png` | 0 sky, 1 installation, 3 beyond the zenith limit |
| `reference_votes.png` | number of reference days on which each pixel is static |
| `classes_<day>.png` | same, plus 2 = variable static of that day |
| `overlay_<day>.jpg` | averaged image of the day with the classes drawn, for inspection |
| `report.json` | parameters, frames used, fractions and component counts per day |

From Python:

```python
from pathlib import Path
from skystatic import SkyGeometry, StaticConfig, open_source, run

report = run(
    days=[open_source(Path("raw/2023-10-06"))],
    reference=[open_source(Path(p)) for p in reference_folders],
    geometry=SkyGeometry.from_file(Path("zenith.npz"), key="zenith_deg"),
    config=StaticConfig(),
    output=Path("out"),
    cache=Path("cache"),
)
```

`skystatic.evaluate.agreement` compares a mask with a reference mask of the same camera.
The [evaluation protocol](docs/evaluation.md) explains how to measure missed dirt
and false detections without selecting thresholds on the final test cases.

## Consume a daily mask

```python
from datetime import datetime
from pathlib import Path
from skystatic import load_day_mask, sky_mask

sky = sky_mask(Path("masks"), datetime(2024, 5, 10, 12, 30), size=(2464, 2464))
# bool array of shape (height, width): True only for class 0 (sky).
product = load_day_mask(Path("masks/classes_2024-05-10.png"))
print(product.metadata.skyline_visible, product.metadata.variable_reliable)
```

Dates follow the clock/calendar in the image filenames; there is no implicit
UTC or local-time conversion. Selection requires the exact date: there is no
nearest-day or installation-only fallback. Two products for the same date are
an error. Resizing uses nearest neighbours and requires the same aspect ratio.
Classes 1 (installation), 2 (daily static) and 3 (outside domain) are all excluded.

Each daily PNG embeds JSON in its `skystatic` text field: schema version, date,
label, `skyline_visible`, `variable_reliable` and frozen `installation_id`, if
used. Quality is available without `report.json`. A missing skyline sets both
flags false; `sky_mask` raises unless `allow_unreliable=True` is explicit.
This is a conservative quality gate, not a measured confidence score or a claim
that every faint spot has been found. The class pixels themselves are unchanged.
Legacy PNGs without metadata are refused and must be regenerated in a fresh
output directory. Previously overwritten products cannot be recovered by this fix.

## Freeze and reuse an installation

Choose and review reference days for one camera, then freeze that run explicitly:

```bash
uv run skystatic freeze /data/path/reference-run /data/path/camera-v1.npz \
    --camera-id CAMERA --version v1 \
    --valid-from 2023-01-01 --valid-until 2027-01-01
uv run skystatic run DAY --zenith zenith.npz --output /data/path/day-run \
    --installation /data/path/camera-v1.npz
```

The dates above are examples; choose the validity period from the camera's
installation history. `valid_from` is inclusive and `valid_until` exclusive.
Freezing reads an existing run without computing new masks. The NPZ contains the
installation classes, original reference votes, reference-day provenance,
calibration identity, parameters, camera ID, version, validity period and a
SHA-256 over provenance and arrays. It is loaded without pickle and its checksum
is verified. Creation is atomic and refuses to overwrite an existing file.
Keep this artifact in durable storage, outside Git, with the reviewed reference
selection and its evaluation; use a new version for a new baseline.

A run with `--installation` does not vote again. It checks the calibration,
segmentation parameters, image dimensions and date validity, and refuses
`--reference` at the same time. Its `reference_votes.png` reproduces the original
votes. Reports and daily PNGs carry the complete installation identifier
(camera/version/content hash). Worker counts, voting thresholds and daily slack
do not change the frozen pixels. A new camera geometry or work resolution
requires a new reviewed artifact. No real-camera baseline or validity period is
automatically chosen by this software change.

Python offers the same operations through `freeze_installation(...)`,
`FrozenInstallation.from_file(...)` and `run(..., installation=artifact)`.

## What it was checked on

2026-11-13 (previous method): twelve consecutive days per camera, at half the native resolution.
The numbers below were measured with the brightness-based day test and the edge test only and
are kept for history; the GVCCS numbers are superseded.

| Camera | Installation | Static per day (min / median / max) | Variable per day (median) |
|---|---|---|---|
| GVCCS site camera, 2023-10-01..12, 5 reference days over 2023-2026 | 11.6 % | 4.0 / 11.7 / 12.0 % | 0.24 %, 29 components |
| Eye2Sky OLUOL, 2022-04-14..25, reference = the same days | 8.6 % | 7.7 / 8.7 / 9.0 % | 0.05 %, 46 components |
| Eye2Sky BARSE, 2022-04-01..12, 4 reference days in April and June | 13.0 % | 8.9 / 11.6 / 12.6 % | 0.03 %, 24 components |

2026-11-17 (this method), GVCCS site camera, twelve review days (2023-09-27 .. 2024-03-18),
ten reference days of 2024-03 .. 2024-10 (after a cleaning that ended a Sep 2023 - Feb 2024
dirt episode, before a dirt spot that appeared in 2024-11), azimuth map and site given:
installation 16.8 % of the domain, of which 3.8 % are the seven rings of the optics; static per
day 11.3-13.9 %, variable static 0.02-5.0 % per day (blue-sky days carry a field of drops: 3.7-5.0 %). Every
day has a visible skyline. The Eye2Sky cameras were not re-run.

## Limits

- **Fog or a white-out day has no visible skyline**: the sky leaks into the installation of
  that day. Such a reference day is dropped from the vote (`rejected_reference_days`), such a
  processed day is flagged (`skyline_visible: false`, `variable_reliable: false`). The pixel classes are
  preserved for inspection, but automatic consumers refuse the daily mask unless
  explicitly opted in. (With daytime frames recognised by structure, none of the twelve
  review days of the site camera is flagged; 2023-10-03 and 2024-01-15 were flagged only
  because night frames diluted their averages.)
- **The reference period decides what "installation" means.** Dirt that lasts longer than
  60 % of the reference days becomes installation; trees that lose their leaves change the
  installation with the season.
- **Faint or thin objects come and go**: a faint cable or a thin mast at the horizon that
  passes the vote on too few reference days is sky on some days and variable static on
  others.
- **Reflections of the sun inside the optics** move during the day and are averaged out; they
  are not handled here. Dust lit only while the sun passes nearby (a few hours) is left as sky:
  it is static only for part of the day, and a daily mask cannot say when.
- Short-exposure cameras (Eye2Sky) need `--pattern` to select the daytime exposure.

## Skycam calibration (optional)

Install the extra in an application using skycam calibrations:

```bash
uv add 'skystatic[skycam]'
```

```python
from pathlib import Path
from skycam.adapters import JP2CalibrationLoader
from skystatic import SkyGeometry

calibration = JP2CalibrationLoader(Path("calibration")).load("visible")
geometry = SkyGeometry.from_skycam(calibration)
```

The constructor converts the native zenith map from radians to degrees,
preserves pixel orientation and excludes non-finite directions. The map must
match `calibration.image_size`. It does not project or resample the image.
`from_array` and `.npy`/`.npz` loading, including the CLI, remain usable without
skycam, OpenCV, Numba, Shapely or Matplotlib installed. This extra adds a Python
constructor; the CLI continues to accept zenith files as before.

## Development

This package lives in `packages/skystatic` of the skyobs workspace. It requires
Python 3.12+; development uses Python 3.14. Tests live in
`tests_skystatic/{unit,integration,e2e}`. Run `uv sync` at the workspace root.

```bash
uv run --directory packages/skystatic pytest  # unit, integration (filesystem) and e2e (CLI) on synthetic scenes
make lint  # from the skyobs workspace root
```
