# skycam

<p align="center">
  <strong>Camera-agnostic fisheye image projection library for ground-based sky observation</strong>
</p>

<!-- Canonical Badge Row: Status → Stack → Tooling → Info -->
<p align="center">
  <!-- Status -->
  <a href="https://github.com/eurocontrol-asu/skycam/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/skycam/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://coveralls.io/github/eurocontrol-asu/skycam?branch=main"><img src="https://coveralls.io/repos/github/eurocontrol-asu/skycam/badge.svg?branch=main" alt="Coverage"></a>
  <!-- Stack -->
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/pydantic-v2-E92063" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/typed-strict-blue" alt="Typed">
  <!-- Tooling -->
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <!-- Info -->
  <a href="https://pypi.org/project/skycam/"><img src="https://img.shields.io/pypi/v/skycam" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="License">
  <a href="https://eurocontrol-asu.github.io/skycam/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| ⚡ **Numba JIT** | ~100x faster projection via compiled bilinear interpolation |
| 💾 **Coordinate Cache** | Sub-100ms init after first calibration load |
| 🏛️ **Hexagonal Architecture** | Clean separation of domain, adapters, and config |
| ✅ **Pydantic v2** | Validated settings with environment variable support |
| 🔬 **Scientific Accuracy** | WGS84 geodesic calculations via GeographicLib |

## 🚀 Installation

```bash
uv add skycam
```

Or with pip:

```bash
pip install skycam
```

## 📖 Quick Start

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

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Adapters["Adapters"]
        CAL[JP2 Calibration]
        IMG[Image I/O]
    end
    
    subgraph Domain["Domain"]
        PROJ[ProjectionService]
        INTERP[Numba Bilinear]
    end
    
    subgraph Config["Config"]
        SETTINGS[SkycamSettings]
    end
    
    CAL --> PROJ
    IMG --> PROJ
    SETTINGS --> PROJ
    PROJ --> INTERP
```

## ⚙️ Configuration

Environment variables (prefix: `SKYCAM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SKYCAM_CALIBRATION_DIR` | `calibration` | JP2 calibration files directory |
| `SKYCAM_CATEGORY` | `visible` | Camera category (`visible`, `infrarouge`) |
| `SKYCAM_DATA_DIR` | `data` | Input/output data directory |

## 🛠️ Development

```bash
git clone https://github.com/eurocontrol-asu/skycam.git
cd skycam
make install  # uv sync --all-groups
make check    # lint + audit + test
```

## 📚 Documentation

**[eurocontrol-asu.github.io/skycam](https://eurocontrol-asu.github.io/skycam/)**

- [🚀 Tutorial](https://eurocontrol-asu.github.io/skycam/tutorial/) — Project your first image in 5 minutes
- [🛠️ Guides](https://eurocontrol-asu.github.io/skycam/guides/) — How-to recipes for common tasks
- [🧠 Concepts](https://eurocontrol-asu.github.io/skycam/concepts/) — Architecture and algorithm deep-dive
- [📚 Reference](https://eurocontrol-asu.github.io/skycam/reference/) — Complete API documentation

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

EUPL-1.2 — See [LICENSE](LICENSE) for details.
