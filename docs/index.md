---
hide:
  - navigation
  - toc
---

# skycam

<p align="center">
  <strong>Camera-agnostic fisheye image projection library for ground-based sky observation</strong>
</p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/skycam/actions/workflows/ci.yml">
    <img src="https://github.com/eurocontrol-asu/skycam/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <a href="https://coveralls.io/github/eurocontrol-asu/skycam?branch=main">
    <img src="https://coveralls.io/repos/github/eurocontrol-asu/skycam/badge.svg?branch=main" alt="Coverage" />
  </a>
  <a href="https://pypi.org/project/skycam/">
    <img src="https://img.shields.io/pypi/v/skycam" alt="PyPI" />
  </a>
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+" />
  <img src="https://img.shields.io/badge/typed-strict-blue.svg" alt="Typed" />
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" />
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv" />
  </a>
</p>

---

## Documentation

| | |
|---|---|
| 🚀 **[Tutorial](tutorial.md)** | Project your first image in 5 minutes |
| 🛠️ **[How-To Guides](guides.md)** | Solve specific problems: batch processing, env config |
| 🧠 **[Concepts](concepts.md)** | Understand the architecture and algorithm |
| 📚 **[Reference](reference.md)** | Complete API documentation |

---

## Quick Install

```bash
uv add skycam
```

## Minimal Example

```python
from pathlib import Path

from skycam.adapters import JP2CalibrationLoader, load_jp2
from skycam.domain.models import ProjectionSettings
from skycam.domain.projection import ProjectionService

# Load calibration and create projector
loader = JP2CalibrationLoader(Path("calibration"))
calibration = loader.load("visible")
projector = ProjectionService(
    calibration=calibration,
    settings=ProjectionSettings(),
)

# Project fisheye → regular grid
projected = projector.project(load_jp2(Path("input.jp2")))
```

## Features

| Feature | Description |
|---------|-------------|
| ⚡ **Numba JIT** | ~100x speedup via compiled kernels |
| 💾 **Coordinate Cache** | Sub-100ms init after first load |
| ✅ **Pydantic v2** | Validated settings with env support |
| 🏛️ **Hexagonal Architecture** | Clean domain/adapters/config layers |

---

<p align="center">
  <a href="tutorial/"><strong>Get Started →</strong></a>
</p>
