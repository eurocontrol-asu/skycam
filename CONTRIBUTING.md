# Contributing to skyobs

## Development Setup

```bash
git clone https://github.com/eurocontrol-asu/skyobs.git
cd skycam
make install
```

The three members live under `packages/skycam`, `packages/skystatic` and
`packages/skywind`.
Use Python 3.14 for development; all packages require Python 3.12+.
Run commands from the workspace root. The root pytest configuration covers
all members and uses importlib to avoid test import collisions.

```bash
uv run --directory packages/skycam pytest
uv run --directory packages/skystatic pytest
uv run --directory packages/skywind pytest
uv build --package skycam
uv build --package skystatic
uv build --package skywind
```

Keep data, masks, results, build outputs and runtime caches out of Git.
Skywind tests needing research stores are skipped unless
`SKYWIND_TEST_STORES=1` is explicitly set. Never rerun frozen campaigns or
holdout evaluations as part of routine package checks.

## Code Quality

Before submitting a PR, run all checks:

```bash
make check  # Runs lint + audit + test
```

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix  
- `docs:` Documentation
- `test:` Tests
- `chore:` Maintenance
- `refactor:` Code refactoring

## Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit changes following convention
4. Push and create a PR
