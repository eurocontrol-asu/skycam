# 🛠️ How-To Guides

Practical recipes for common tasks. Each guide solves a specific problem.

---

## ⚡ How to Optimize Batch Processing

**Problem:** Processing many images is slow because interpolators rebuild each time.

**Solution:** Use `lazy_init=True` and `calibration_path` for caching.

```python
from pathlib import Path

from skycam.adapters import JP2CalibrationLoader, load_jp2, save_image
from skycam.domain.models import ProjectionSettings
from skycam.domain.projection import ProjectionService

calibration_dir = Path("calibration")
loader = JP2CalibrationLoader(calibration_dir)
calibration = loader.load("visible")

# Enable coordinate caching for ~100x faster init after first run
projector = ProjectionService(
    calibration=calibration,
    settings=ProjectionSettings(),
    calibration_path=calibration_dir,  # Enables disk caching
    lazy_init=True,  # Defer init until first project()
)

# Process all images with a single interpolator
for image_path in Path("images").glob("*.jp2"):
    image = load_jp2(image_path)
    projected = projector.project(image)
    save_image(projected, Path("output") / f"{image_path.stem}.jpg")
```

The cache is stored in `calibration/.cache/pixel_coords_*.npy`.

---

## 🔧 How to Configure via Environment Variables

**Problem:** You need different settings per deployment without code changes.

**Solution:** Use `SKYCAM_` prefixed environment variables.

```bash
export SKYCAM_CALIBRATION_DIR=/mnt/data/calibration
export SKYCAM_CATEGORY=infrarouge
export SKYCAM_DATA_DIR=/mnt/output
```

```python
from skycam.config import SkycamSettings

# Automatically loads from environment
settings = SkycamSettings()
print(settings.calibration_dir)  # /mnt/data/calibration
print(settings.category)         # infrarouge
```

Or use a `.env` file:

```ini
# .env
SKYCAM_CALIBRATION_DIR=/data/calibration
SKYCAM_CATEGORY=visible
```

---

## 📍 How to Calculate Geographic Coordinates

**Problem:** You need to find the lat/lon of an aircraft visible in the image.

**Solution:** Use `calculate_latitude_longitude()` with known azimuth/zenith.

```python
from skycam.domain.projection import ProjectionService
from skycam.domain.models import ProjectionSettings, CalibrationData
import numpy as np

# Create a minimal projector (calibration not needed for geo calculations)
calibration = CalibrationData(
    azimuth_array=np.zeros((10, 10)),
    zenith_array=np.zeros((10, 10)),
    image_size=(10, 10),
)
projector = ProjectionService(
    calibration=calibration,
    settings=ProjectionSettings(),
    lazy_init=True,  # Skip interpolator build
)

# Observer position (camera location)
observer_lat = 48.6005
observer_lon = 2.3468
observer_alt = 90.0

# Calculate target position from viewing angles
lat, lon = projector.calculate_latitude_longitude(
    azimuth=45.0,        # degrees from north
    zenith=30.0,         # degrees from vertical
    target_altitude=10000.0,  # aircraft altitude in meters
    observer_lat=observer_lat,
    observer_lon=observer_lon,
    observer_alt=observer_alt,
)

print(f"Aircraft at: {lat:.4f}°N, {lon:.4f}°E")
```

---

## 🎯 How to Change Output Resolution

**Problem:** You need higher or lower resolution output.

**Solution:** Configure `ProjectionSettings.resolution`.

```python
from skycam.domain.models import ProjectionSettings

# Low resolution (fast, 256x256 output)
settings_fast = ProjectionSettings(resolution=256)

# High resolution (slow, 4096x4096 output)
settings_hq = ProjectionSettings(resolution=4096)

# Default is 1024x1024
settings_default = ProjectionSettings()
```

!!! warning "Resolution affects cache"
    Each resolution creates a separate cache file. Switching resolutions will trigger a new ~10s interpolator build.

---

## 🌡️ How to Switch Camera Categories

**Problem:** You have multiple camera types (visible, infrared).

**Solution:** Pass the category to `JP2CalibrationLoader.load()`.

```python
from skycam.adapters import JP2CalibrationLoader
from pathlib import Path

loader = JP2CalibrationLoader(Path("calibration"))

# Load visible camera calibration
visible_cal = loader.load("visible")

# Load infrared camera calibration
infrared_cal = loader.load("infrarouge")
```

Required files:
```
calibration/
├── azimuth_visible.jp2
├── zenith_visible.jp2
├── azimuth_infrarouge.jp2
└── zenith_infrarouge.jp2
```

---

## 🧪 How to Run Tests

**Problem:** You want to verify the installation works correctly.

**Solution:** Use the Makefile targets.

```bash
# Clone and install
git clone https://github.com/eurocontrol-asu/skycam.git
cd skycam
make install

# Run all checks (lint + security + tests)
make check

# Run only tests
make test

# Run with coverage
uv run pytest --cov=src --cov-report=html
```
