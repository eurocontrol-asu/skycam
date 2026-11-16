# Contributing to skyobs

## Development Setup

```bash
git clone https://github.com/eurocontrol-asu/skycam.git
cd skycam
make install
```

The two members live under `packages/skycam` and `packages/skystatic`.
Use Python 3.14 for development; both packages require Python 3.12+.
Run commands from the workspace root. The root pytest configuration covers
both members and uses importlib to avoid test import collisions.

```bash
uv run --directory packages/skycam pytest
uv run --directory packages/skystatic pytest
uv build --package skycam
uv build --package skystatic
```

Keep data, masks, results, build outputs and runtime caches out of Git.

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
