# skycam

<p align="center">
  <a href="https://github.com/eurocontrol-asu/skycam/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/skycam/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://coveralls.io/github/eurocontrol-asu/skycam?branch=main"><img src="https://coveralls.io/repos/github/eurocontrol-asu/skycam/badge.svg?branch=main" alt="Coverage"></a>
  <a href="https://pypi.org/project/skycam/"><img src="https://img.shields.io/pypi/v/skycam" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/typed-strict-blue" alt="Typed">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://eurocontrol-asu.github.io/skycam/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
</p>

> Camera-agnostic fisheye image projection library for ground-based sky observation

## Installation

```bash
uv add skycam
```

## Quick Start

```python
from pathlib import Path

from skycam.adapters import JP2CalibrationLoader, load_jp2
from skycam.domain.models import ProjectionSettings
from skycam.domain.projection import ProjectionService

# Load calibration data
loader = JP2CalibrationLoader(Path("calibration"))
calibration = loader.load("visible")

# Create projection service
settings = ProjectionSettings(resolution=1024, cloud_height=10000.0)
projector = ProjectionService(calibration=calibration, settings=settings)

# Project a fisheye image
image = load_jp2(Path("input.jp2"))
projected = projector.project(image)
```

## Development

```bash
git clone https://github.com/eurocontrol-asu/skycam.git
cd skycam
make install
make check
```

## License

EUPL-1.2 - See [LICENSE](LICENSE) for details.
