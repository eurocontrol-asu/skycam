# skywind

Cruise-altitude wind (FL250-FL410) above the GVCCS ground camera (48.6005 N, 2.3468 E, domain about +-1.5 deg),
reconstructed for any day by correcting ERA5 with winds derived from Mode S Comm-B replies (BDS 5,0 track and turn,
BDS 6,0 heading and speed) collected by OpenSky. The corrected field and its uncertainty are exported as a NetCDF that
pycontrails opens as a `MetDataset` in place of ERA5, and used to advect contrail emission points for 0-40 min.

Milestone 1 of the library: it ports the prototype study (`contrail-llm-labelling/out/layer-motion/multiday`), replaces
its kernel-weighted ridge estimator by a proper Gaussian-process analysis (the ridge is kept as a selectable baseline),
fits the hyperparameters on aircraft data only, and ships two evaluations (aircraft leave-one-out; contrail advection
against GVCCS labels).

## Method in one page

**Observations** (`decode`, `identify`, `observations`). Raw DF20/21 replies are decoded under both registers
(vectorised); a reply is BDS 5,0 if its ground speed / track agree with the ADS-B state vector of the same aircraft
(<= 12 kt, <= 6 deg), BDS 6,0 if IAS and Mach agree at the barometric altitude (<= 12 kt) and the heading is within 30 deg
of the track; ambiguous replies are dropped. Receivers whose replies disagree with the state vectors (clock errors,
2024-05-10) are scored and distrusted. One observation per (flight, 30 s bin) with >= 2 replies of each register,
wings level (|roll| <= 3 deg), level (|vertical rate| <= 1000 ft/min), steady heading / track (circular sd <= 0.5 deg)
and speeds (sd <= 4 kt): wind = ground vector - air vector (TAS + magnetic heading + IGRF declination computed at run
time with `ppigrf`). The residual against ERA5 (linear in time, pressure from barometric altitude, lat, lon; the same
convention as the pycontrails advection) is split along / to the right of the heading.

**Field model** (`field.gp`). The residual r(x, y, z, t) = (du, dv) has independent zero-mean Gaussian-process
components, variance `sigma_field^2`, separable correlation in horizontal distance, pressure altitude and time
(Matern 3/2 by default, Gaussian available; flow-following by default). Each observation is a scalar:

    y_i = h_i . r(x_i) + b_aircraft(i) + e_i,    b ~ N(mean_bias, sigma_bias^2),   e ~ N(0, sigma_noise^2)

with h_i the heading unit vector. The per-aircraft random effect (airspeed bias) makes the 30 s observations of one
aircraft correlated, which keeps the posterior variance honest. The posterior mean and covariance of (du, dv) at any
point are the exact conditional Gaussians, solved locally (tiles of 15 min x one altitude band, observations within 3
length scales, at most 6000) so that a whole day scales. Far from data the residual returns to 0 (ERA5) and its
standard deviation to `sigma_field`: the exported uncertainty says where the correction is informed.
With crossing traffic both components are determined; with one-directional traffic the orthogonal component stays at
the prior (tested).

Options (all evaluated, see results): no per-aircraft bias; *flow-following* observations (coordinates displaced with
the ERA5 wind to the tile time, so residual structures move with the flow); *cross-heading* observations after a
per-airframe heading-offset calibration (EMADDC-style: offset = circular mean of reference-heading minus IGRF heading,
reference = ERA5 + the along-only field of the other aircraft; persistent table accumulating per (airframe, day)
sufficient statistics, versioned, with n and day span; along-only fallback for airframes without enough data), with their
own noise and per-aircraft effect.

Baseline (`field.ridge`): the prototype's kernel-weighted ridge least squares, same interface.

**Hyperparameters** (`field.hyper`) are estimated from aircraft data only: maximum composite marginal likelihood
(30 min blocks, exact likelihood and analytic gradient, L-BFGS-B) for the GP; leave-one-aircraft-out grid search for
the ridge. Never from contrail annotations. Transfer is shown by leave-one-day-out (each day evaluated with values
fitted on the other days). Checks: Hollingsworth-Lonnberg binned covariances of cross-aircraft innovation pairs and
Desroziers ratios (`field.diagnostics`).

**Outputs** (`windfield`, `export`, `advect`). `CorrectedWind.at(t, p, lat, lon)` / `.at_altitude(...)` returns ERA5,
residual, corrected wind, residual sd and an information index. `export` writes the ERA5 horizontal grid refined x2
(0.125 deg, to resolve the ~25 km residual scale), ERA5 levels plus every 5 hPa (about 140 m at 250 hPa, below the
~200 m vertical scale), 10 min steps: `eastward_wind` / `northward_wind` (corrected), `era5_*`, `residual_*`,
`residual_*_std`, `residual_wind_covariance`, `residual_information`, plus ERA5 t, q, w, z. Advection is pycontrails
`DryAdvection` (pointwise, u / v / omega, 30 s steps) on that dataset, identical to the prototype's ERA5 advection
when the residual is zero (checked to the metre).

## Layout

    src/skywind/
      config.py          pydantic settings (paths, site, decoding, observation, GP, ridge, export, sources)
      days.py            days and windows (days.csv: dev / holdout sets)
      decode.py          BDS 5,0 / 6,0 decoding         identify.py   register identification, receiver check
      statevectors.py    loading, sanitising, track     declination.py IGRF (ppigrf)
      era5.py            ERA5 interpolation             observations.py 30 s winds and residuals
      field/             data, kernels, gp, ridge, hyper (fits), heading (offset table), diagnostics
      windfield.py       CorrectedWind                  export.py     NetCDF / MetDataset
      advect.py          DryAdvection                   pipeline.py   per-day steps     workflow.py  end-to-end
      sources/, acquire.py   new-day acquisition (ssh copy from a data host, OpenSky Trino)
      evaluate/          aircraft (A), contrails (B, generic), gvccs_calib (truth source), runner, campaign
      cli.py             thin cyclopts commands

## Data

No absolute data path in code. `skywind.toml` (committed) points at two gitignored symlinks of the repository:
`data -> /data/gjarry/contrail` (working data: `cache/` and `runs/` under the campaign `20261113-skywind`) and
`store -> /datasave/gjarry/contrail` (durable inputs `inputs/20261113-skywind/{commb,opensky,era5}` and `days.csv`).
`prototype` and `gvccs` symlinks give the calibration artefacts and the GVCCS train split (the test split is never
read). Override per machine with `skywind.local.toml` or `SKYWIND_DATA`, `SKYWIND_STORE`, `SKYWIND_PROTOTYPE`,
`SKYWIND_GVCCS`. Every run directory has a `README.md` whose first line is `status: ... — purpose`.

## Running

    uv sync --extra opensky          # pyopensky only needed to acquire new days
    uv run skywind days --set-name dev

New day, end to end (fetches state vectors and ERA5 from the configured host, Comm-B from OpenSky Trino with the
credentials of `~/.config/pyopensky/settings.conf` -- it fails instead of opening a browser -- then decodes, builds the
observations, exports the corrected field and advects the cruise traffic of the box with ERA5 and corrected winds):

    uv run skywind run 2024-06-01 --windows "05:00-07:00"

Step by step: `skywind acquire DAY WINDOWS`, `decode DAY...`, `observe DAY...`, `calibrate DAY...` (heading table, for
the cross-heading option), `fit DAY... --out params.json`, `export DAY [--params params.json] [--method ridge]`,
`advect DAY [--points emissions.parquet] [--winds era5|corrected]`.

Evaluations (dev days by default; holdout days are never used for tuning):

    uv run skywind evaluate aircraft gp gp_nobias gp_flow gp_matern gp_cross gp_cross_strict
    uv run skywind evaluate contrails gp gp_flow gp_cross ridge --aircraft-runs data/runs/20261113-skywind/<aircraft run>

Library use:

```python
from skywind.config import load_settings
from skywind.pipeline import fitted_field, gp_params

s = load_settings()
wind = fitted_field(s, "20240318", gp_params(s))          # ERA5 + GP residual from that day's Mode S
table = wind.at(t_epoch, p_hpa, lat, lon)                  # u, v, du, dv, std_u, std_v, information...
```

Tests: `uv run pytest` (unit; integration and e2e need the stores: `uv run pytest -m "integration or e2e"`),
`uv run ruff check src tests`, `uv run mypy`.

## Results (milestone 1, six dev days)

Runs (in `data/runs/20261113-skywind/`, status `reference`): `20261113-eval-aircraft-3` (A, GP variants),
`20261113-eval-aircraft-2` (A, ridge baseline, transfer table), `20261113-eval-contrails-merged` (B),
`20261113-diagnostics`. Reproduce with `skywind evaluate aircraft ...`, `skywind evaluate contrails ...`,
`skywind evaluate merge-contrails ...`, `skywind evaluate report --aircraft-run ... --contrail-run ...`.

### A. Aircraft leave-one-aircraft-out, leave-one-day-out hyperparameters

Along-heading residual of each aircraft predicted from the OTHER aircraft only; hyperparameters fitted without the
evaluated day. Skill = 1 - MSE / MSE(ERA5). RMSE in m/s.

| day | obs | aircraft | RMSE ERA5 | GP Matern+flow | GP Matern | ridge (LOAO-tuned) | skill Matern+flow | skill Matern | skill ridge |
|---|---|---|---|---|---|---|---|---|---|
| 20230927 | 11674 | 361 | 1.90 | 1.38 | 1.48 | 1.58 | 0.48 | 0.40 | 0.31 |
| 20240115 | 6354 | 189 | 1.66 | 1.38 | 1.33 | 1.36 | 0.31 | 0.37 | 0.33 |
| 20240206 | 6946 | 220 | 1.78 | 1.39 | 1.37 | 1.43 | 0.39 | 0.41 | 0.35 |
| 20240318 | 11419 | 324 | 2.54 | 1.62 | 1.65 | 1.72 | 0.59 | 0.58 | 0.54 |
| 20240412 | 15283 | 414 | 2.61 | 1.66 | 1.70 | 1.72 | 0.60 | 0.58 | 0.56 |
| 20240510 | 7263 | 301 | 1.48 | 1.16 | 1.17 | 1.21 | 0.38 | 0.38 | 0.33 |
| pooled | 58939 | 1453* | 2.16 | 1.48 | 1.51 | 1.57 | **0.53** | 0.51 | 0.47 |

\* aircraft-days. All variants, pooled (z = normalised residual, ideal sd 1, coverage 0.68 / 0.95):

| variant | RMSE | skill | z sd | cover 68 | cover 95 |
|---|---|---|---|---|---|
| GP Gaussian | 1.60 | 0.450 | 1.02 | 0.71 | 0.95 |
| GP Gaussian, no per-aircraft bias | 1.66 | 0.408 | 1.21 | 0.63 | 0.90 |
| GP Gaussian, flow-following | 1.58 | 0.467 | 1.01 | 0.72 | 0.95 |
| GP Gaussian, cross-heading (offset table, all days) | 1.64 | 0.415 | 1.04 | 0.71 | 0.94 |
| GP Gaussian, cross-heading (offset table without the day) | 1.61 | 0.442 | 1.02 | 0.72 | 0.95 |
| GP Matern 3/2 | 1.51 | 0.510 | 1.03 | 0.71 | 0.95 |
| GP Matern, day-specific variances | 1.51 | 0.511 | 0.99 | 0.71 | 0.95 |
| **GP Matern, flow-following (retained)** | **1.48** | **0.530** | 1.02 | 0.71 | 0.95 |
| ridge, LOAO-tuned (20 km / 150 m / 20 min) | 1.57 | 0.473 | 0.88 | 0.78 | 0.97 |
| ridge, prototype kernel (75 km / 500 m / 20 min) | 1.94 | 0.194 | 1.27 | 0.61 | 0.89 |

Retained hyperparameters (Matern 3/2, flow-following, all six days; shipped as `GPConfig` defaults): horizontal
43.4 km, vertical 230 m, time 72.6 min, field sd 1.77 m/s, per-aircraft airspeed bias N(-0.19, 0.78^2) m/s, noise
0.59 m/s. Per fold (fitted without the held-out day): 39-46 km, 211-261 m, 64-90 min, field sd 1.60-1.85, bias sd
0.75-0.81, mean bias -0.19 to -0.21, noise 0.57-0.60: the values transfer between days. For the Gaussian kernel the
transfer table (`eval-aircraft-2/transfer.csv`) gives per-day skill with the other days' values within a few
hundredths of the skill with the day's own values. The per-day calibration is better when the three variances are
re-estimated on the day's own aircraft data (z sd per day 0.93-1.08 instead of 0.78-1.19), so `export` / `run` do
that by default (length scales unchanged).

Diagnostics (Gaussian defaults, `20261113-diagnostics`): Hollingsworth-Lonnberg cross-aircraft covariances give a
vertical scale of 195 m (marginal likelihood: 196 m), but a slow horizontal decay (3.9 at 5 km, 2.5 at 40 km, 1.5 at
100 km; single-Gaussian fit 96 km) -- a mixture of scales, which is why the heavier-tailed Matern kernel fits better --
and a long time scale (78 min). Desroziers ratios vary strongly by day (background variance x0.5 to x1.8: 2024-03-18
and 2024-04-12 are the high-variance days), consistent with the day-specific variances above.

### B. Contrail advection against GVCCS labels (six days, 87 flights with a drift fit)

Cross-track error e = (annotated centre - advected position) . right normal, per point and frame; per flight a line
e = offset + drift x age is fitted to per-minute medians (ages 0.5-40 min, >= 8 distinct minutes). Drift is the wind
effect (m/min), the offset at emission is not. Hyperparameters of each day fitted without that day (evaluation A);
nothing from GVCCS enters any fit. pycontrails DryAdvection, 30 s steps, on the exported fields.

| day | flights | ERA5 median / p90 | GP Matern median / p90 | improved | GP Matern+flow median / p90 | improved | ridge median / p90 | improved |
|---|---|---|---|---|---|---|---|---|
| 20230927 | 5 | 60.6 / 98.6 | 21.1 / 53.0 | 0.80 | 29.5 / 42.6 | 0.80 | 29.7 / 58.5 | 0.80 |
| 20240115 | 6 | 25.7 / 71.2 | 13.1 / 53.5 | 1.00 | 19.2 / 68.5 | 0.83 | 30.9 / 63.3 | 0.50 |
| 20240206 | 12 | 45.5 / 119.2 | 28.1 / 105.3 | 0.67 | 22.7 / 116.3 | 0.67 | 25.1 / 72.6 | 0.83 |
| 20240318 | 19 | 62.3 / 169.4 | 39.4 / 81.8 | 0.79 | 39.9 / 103.8 | 0.63 | 40.5 / 116.3 | 0.63 |
| 20240412 | 20 | 84.3 / 114.6 | 44.3 / 89.1 | 0.80 | 37.0 / 97.4 | 0.80 | 43.8 / 89.7 | 0.65 |
| 20240510 | 25 | 36.2 / 108.4 | 18.0 / 85.4 | 0.64 | 25.3 / 88.1 | 0.56 | 32.8 / 84.7 | 0.48 |
| pooled | 87 | **58.6 / 114.9** | **29.8 / 95.7** | **0.75** | **28.2 / 101.1** | 0.68 | 33.3 / 89.9 | 0.62 |

(|drift| in m/min; "improved" = share of flights whose |drift| is smaller than with ERA5.)

Pooled |cross-track distance| (m) by age, median (p90); "offset removed" subtracts each flight's own offset:

| age (min) | ERA5 | GP Matern | GP Matern+flow | ridge | ERA5, offset removed | Matern | Matern+flow | ridge |
|---|---|---|---|---|---|---|---|---|
| 0.5-5 | 174 (514) | 131 (390) | 143 (402) | 143 (404) | 186 | 139 | 148 | 150 |
| 5-10 | 425 (993) | 249 (726) | 272 (783) | 304 (733) | 408 | 248 | 245 | 293 |
| 10-20 | 664 (1679) | 411 (1278) | 411 (1389) | 525 (1323) | 634 | 383 | 361 | 464 |
| 20-40 | 1375 (2964) | 843 (2273) | 736 (2355) | 1073 (2561) | 1455 | 811 | 712 | 1058 |

Offset at emission, median |offset| (m): ERA5 110, GP Matern 110, GP Matern+flow 113, ridge 124 (signed median -33
for ERA5): the correction does not change it, as expected for a non-wind effect.

Uncertainty check (per flight drift / (60 x posterior sd of the residual along the normal), ideal sd 1): GP Matern sd
0.78 (95 % coverage 0.97), GP Matern+flow 0.86 (0.94), ridge 1.63 (0.75): the GP standard deviation is usable (slightly
conservative), the ridge one is not.

### Not done / limits

- Evaluation B for the day-specific-variance and cross-heading variants on the Matern base, and the Matern
  cross-heading variant in A, were stopped unfinished (priority to B). The LOAO-tuned ridge grid was not re-run on the
  wider grid (its fold values sit at the old grid's lower edges, 20 km / 150 m); the transfer table and the
  diagnostics were computed for the Gaussian kernel only.
- The local GP keeps at most 6000 neighbours per tile: with 3000 the busy days lost about 0.05 of skill (now default
  6000; exporting one day takes a few minutes).
- Six dev days only; the ten holdout days of `days.csv` were not touched.
