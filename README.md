# skyobs

A uv workspace for ground-based sky observation, with two independent Python packages.

| Package | Purpose | Documentation |
|---|---|---|
| [skycam](packages/skycam/) | Calibrated fisheye geometry, image projection and dense sampling maps | [Online documentation](https://eurocontrol-asu.github.io/skycam/) |
| [skystatic](packages/skystatic/) | Installation and daily static masks from averaged all-sky images | [Method, usage and limitations](packages/skystatic/README.md) |

Both packages require **Python 3.12+**. Development uses **Python 3.14**;
CI tests both members on Python 3.12, 3.13 and 3.14. Python 3.15 remains
unverified, including the availability of stable dependency wheels.
The virtual root project is named `skyobs-workspace`; it is not published.

## Development

From the workspace root:

```bash
uv sync
uv run pytest
make lint
uv build --package skycam
uv build --package skystatic
uv run --group docs mkdocs serve -f packages/skycam/mkdocs.yml
```

`uv sync` installs both members and the common development tools. Use
`uv sync --all-packages --all-groups --all-extras` for all tooling and extras.
There is one `uv.lock`. Each member keeps its own ruff, mypy and pytest
configuration. Root pytest uses importlib and collects both configured test
roots; skystatic uses `tests_skystatic/` to avoid collisions with skycam's
existing `tests/` package. CI discovers members from `[tool.uv.workspace]`.

The source packages keep their public import names, `skycam` and `skystatic`.
Installing `skystatic` alone does not install skycam. Install
`skystatic[skycam]` to use `SkyGeometry.from_skycam(calibration)`; see the
[example](packages/skystatic/README.md#skycam-calibration-optional).

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
convention. Skystatic has an independent `skystatic/v<version>` namespace.
No skystatic publishing workflow is enabled. Without matching local tags,
versions are development versions derived from Git, not release numbers.
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
