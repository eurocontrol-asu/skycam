# skystatic

Static parts of a fixed all-sky camera, found from averaged images: the **installation**
(trees, buildings, masts, antennas) and the **variable static** of each day (dirt or drops on
the optics, anything that stays put for a day but not for months).

No trained model. Inputs: the frames of several days and the zenith angle of every pixel
(camera calibration).

## Method

1. **Average a day.** Summing the daytime frames and dividing by their number erases what
   moves (clouds, sun): the sky becomes smooth and everything fixed keeps sharp edges. The
   default average is the geometric mean (product of the frames scaled to 0..1, to the power
   1/N), which is pulled down by the dark moments of a pixel and so strengthens dark objects;
   the plain mean is available (`--aggregate mean`).
2. **Sky = the smooth region around the zenith.** Edges are measured on the log image, after
   dividing by the sky level, so the threshold does not depend on exposure, and as a local
   excess over the median gradient of the surroundings: the steady brightening of the sky
   towards the horizon is a gradient, not an edge. Everything below
   the zenith limit that is not connected to the zenith through smooth pixels is static: a
   uniform roof enclosed by its own outline is static as well.
3. **A day is cut into parts that must agree** (default 3). A slow cloud bank can survive one
   average; it does not sit at the same place in all of them.
4. **Installation = static on nearly every reference day** (default 60 %). The reference days
   must span longer than the life of anything on the optics, otherwise long-lived dirt is
   counted as installation.
5. **Variable static of a day = its mask minus the installation**, with a small slack for the
   pixel or two by which an edge moves between days.

## Use

```bash
uv sync
uv run skystatic run DAY [DAY ...] --zenith zenith.npz --output out/ \
    [--reference REF_DAY ...] [--cache cache/] [--pattern '*_160.jpg'] [--jobs 6]
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

## What it was checked on (2026-11-13)

Twelve consecutive days per camera, at half the native resolution, default parameters.

| Camera | Installation | Static per day (min / median / max) | Variable per day (median) |
|---|---|---|---|
| GVCCS site camera, 2023-10-01..12, 5 reference days over 2023-2026 | 11.6 % | 4.0 / 11.7 / 12.0 % | 0.24 %, 29 components |
| Eye2Sky OLUOL, 2022-04-14..25, reference = the same days | 8.6 % | 7.7 / 8.7 / 9.0 % | 0.05 %, 46 components |
| Eye2Sky BARSE, 2022-04-01..12, 4 reference days in April and June | 13.0 % | 8.9 / 11.6 / 12.6 % | 0.03 %, 24 components |

Against the obstruction masks shipped by DLR with Eye2Sky (which include a safety margin):
every pixel of our installation lies inside the DLR mask on both stations; we recover 73 %
(OLUOL) and 65 % (BARSE) of the DLR mask, 89 % and 84 % within 4 pixels. What is left is the
DLR margin and, on BARSE, sky seen through bare branches.

On the site camera the known dirt spots of October 2023 come out as variable static, and the
drops of 2023-10-01/02 as well.

## Limits

- **Fog or a white-out day has no visible skyline**: the sky leaks into the installation of
  that day. Such a reference day is dropped from the vote (`rejected_reference_days`), such a
  processed day is flagged (`skyline_visible: false`, `variable_reliable: false`). The pixel classes are
  preserved for inspection, but automatic consumers refuse the daily mask unless
  explicitly opted in (site camera: 2023-10-03 and the reference day 2024-02-02).
- **The reference period decides what "installation" means.** Dirt that lasts longer than
  60 % of the reference days becomes installation; trees that lose their leaves change the
  installation with the season.
- **Faint or thin objects come and go**: the thin end of an antenna or a faint dirt spot is
  found on clear days and missed on hazy ones, so it can show up as variable static.
- **Reflections of the sun inside the optics** move during the day and are averaged out; they
  are not handled here.
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
