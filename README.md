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
(Gaussian by default, Matern 3/2 available). Each observation is a scalar:

    y_i = h_i . r(x_i) + b_aircraft(i) + e_i,    b ~ N(mean_bias, sigma_bias^2),   e ~ N(0, sigma_noise^2)

with h_i the heading unit vector. The per-aircraft random effect (airspeed bias) makes the 30 s observations of one
aircraft correlated, which keeps the posterior variance honest. The posterior mean and covariance of (du, dv) at any
point are the exact conditional Gaussians, solved locally (tiles of 15 min x one altitude band, observations within 3
length scales, at most 3000) so that a whole day scales. Far from data the residual returns to 0 (ERA5) and its
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
    uv run skywind evaluate contrails data/runs/20261113-skywind/<aircraft run> gp gp_flow gp_cross

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

RESULTS_PLACEHOLDER
