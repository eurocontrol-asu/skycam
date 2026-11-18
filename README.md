# skyobs

A uv workspace for ground-based sky observation, with five independent Python packages.

| Package | Purpose | Documentation |
|---|---|---|
| [skycam](packages/skycam/) | Calibrated fisheye geometry, image projection and dense sampling maps | [Online documentation](https://eurocontrol-asu.github.io/skycam/) |
| [skycloud](packages/skycloud/) | Continuous cloud map: value, confidence and an unknown state per sky pixel | [Method, inputs, products and limits](packages/skycloud/README.md) |
| [skystatic](packages/skystatic/) | Installation and daily static masks from averaged all-sky images | [Method, usage and limitations](packages/skystatic/README.md) |
| [skysun](packages/skysun/) | Where and how much the sun disturbs each frame: its glow and the reflections of the optics, a value in [0, 1] per pixel | [Method, products, calibration and limits](packages/skysun/README.md) |
| [skywind](packages/skywind/) | Mode S observations, ERA5 wind correction and contrail advection | [Method, configuration and frozen evaluations](packages/skywind/README.md) |

All five packages require **Python 3.12+**. Development uses **Python 3.14**;
CI tests all members on Python 3.12, 3.13 and 3.14. Python 3.15 remains
unverified, including the availability of stable dependency wheels.
The virtual root project is named `skyobs-workspace`; it is not published.

## Development

From the workspace root:

```bash
uv sync
uv run pytest
make lint
uv build --package skycam
uv build --package skycloud
uv build --package skystatic
uv build --package skysun
uv build --package skywind
uv run --group docs mkdocs serve -f packages/skycam/mkdocs.yml
```

`uv sync` installs all five members and the common development tools. Use
`uv sync --all-packages --all-groups --all-extras` for all tooling and extras.
There is one `uv.lock`. Each member keeps its own ruff, mypy and pytest
configuration. Root pytest uses importlib and collects all configured test
roots; skycloud uses `tests_skycloud/`, skystatic uses `tests_skystatic/`, skysun uses `tests_skysun/` and skywind uses `tests_skywind/`
to avoid collisions with skycam's existing `tests/` package. CI discovers members from `[tool.uv.workspace]`.

The source packages keep their public import names: `skycam`, `skycloud`, `skystatic`, `skysun` and `skywind`.
Skystatic identifies days from image timestamps and provides `sky_mask(...)`
for exact-date, size-aware consumption, with quality metadata inside each PNG.
Its installation can be frozen and reused per camera and validity period; see
[consumer and installation usage](packages/skystatic/README.md#consume-a-daily-mask).
Installing `skystatic` alone does not install skycam. Install
`skystatic[skycam]` to use `SkyGeometry.from_skycam(calibration)`; see the
[example](packages/skystatic/README.md#skycam-calibration-optional).
Skysun depends on skystatic (sky pixels and ring circles, `skystatic.load_optics`), never on
skycloud; its reflection table for the GVCCS site camera is embedded and checksummed.

## Skywind configuration and tests

Run `uv run skywind ...` from the workspace root, where `skywind.toml` lives.
Configuration retains its original precedence: defaults, `skywind.toml`,
`skywind.local.toml` in the current directory, an explicit Python config file,
then `SKYWIND_DATA`, `SKYWIND_STORE`, `SKYWIND_PROTOTYPE` and `SKYWIND_GVCCS`.
The ignored `data`, `store`, `prototype` and `gvccs` links identify local input
and output locations; they are not included in Git or distributions.

OpenSky acquisition remains optional (`uv sync --package skywind --extra opensky` or install
`skywind[opensky]`). The tested `pyopensky>=2.16,<2.17` constraint is retained.
Installing skywind alone does not install skycam or skystatic.

Default tests include synthetic unit, temporary-file integration and CLI cases.
Six skywind tests tagged `requires_store` are skipped even when local data links
exist. To opt into those tests explicitly, configure the stores and run:

```bash
SKYWIND_TEST_STORES=1 uv run pytest -c packages/skywind/pyproject.toml packages/skywind/tests_skywind -m requires_store
```

Importing the package does not rerun frozen evaluations or tune their parameters.

## Files and storage

Keep image collections, calibrations for real cameras, results, masks, caches
and environments outside the repository. Local ignored `data` and `store`
symlinks can point to working and durable storage. The existing small skycam
fixtures under `packages/skycam/tests/fixtures/` are regression references;
projection caches are generated in pytest's temporary directory. Synthetic
skystatic scenes are generated in memory or pytest temporary directories.
For verification without temporary files in the checkout, pass
`--basetemp=/path/to/external/scratch` to pytest and `--out-dir` to `uv build`.

## Versions and publication

Skycam remains the same PyPI distribution. Hatch-vcs reads the repository
root through `root = "../.."`, retaining the existing `v<version>` release
convention. Skystatic and skywind have independent `skystatic/v<version>` and
`skywind/v<version>` namespaces. No publishing workflow is enabled for either.
Without matching local tags,
versions are development versions derived from Git, not release numbers.
Skywind's complete source history is retained, including the original commits
referenced by `frozen-v1` and `frozen-v1.1`. Their IDs and historical lockfiles
remain accessible through Git; the original source checkout and tags are kept
intact. Frozen research configurations are not release-version tags.
Full Git history and release tags must be available in release checkouts;
CI checks out with `fetch-depth: 0`.

The PyPI workflow builds and publishes **only skycam**, from
`packages/skycam`, through the existing `pypi` environment and OIDC Trusted
Publishing. The PyPI project's trusted publisher must authorize this repository,
workflow and environment before a release. Local builds never publish.

Skycam's documentation workflow builds `packages/skycam/mkdocs.yml` and keeps
the existing website structure. Repository renaming, the public documentation
address, pushing branches and commits, PyPI releases and downstream editable
path changes remain operator decisions.
