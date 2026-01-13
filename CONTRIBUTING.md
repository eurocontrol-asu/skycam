# Contributing to skycam

## Development Setup

```bash
git clone https://github.com/eurocontrol-asu/skycam.git
cd skycam
make install
```

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
