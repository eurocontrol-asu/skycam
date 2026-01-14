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

## 📍 How to Convert Pixels to Geographic Coordinates

**Problem:** You need to find the lat/lon of an object at a specific pixel location in the projected image.

**Solution:** Use `AircraftProjector.pixels_to_lonlat()` with the target altitude.

```python
import numpy as np
from skycam.domain import AircraftProjector, Position

# Camera position
pos = Position()  # Uses ECTL Bretigny defaults, or pass custom values
projector = AircraftProjector(
    camera_lat=pos.latitude,
    camera_lon=pos.longitude,
    camera_alt=pos.altitude,
)

# Pixel coordinates of interest (e.g., from click or detection)
px = np.array([512.0])  # X pixel
py = np.array([400.0])  # Y pixel
alt = np.array([10000.0])  # Assumed altitude in meters

# Convert to geographic coordinates
lon, lat = projector.pixels_to_lonlat(px, py, alt)

print(f"Object at: {lat[0]:.4f}°N, {lon[0]:.4f}°E")
```

---

## 🛫 How to Overlay Aircraft Positions

**Problem:** You have aircraft positions (ADS-B data) and want to plot them on a projected image.

**Solution:** Use `AircraftProjector` for vectorized lon/lat → pixel conversion.

```python
import numpy as np
from skycam.domain import AircraftProjector, Position

# Use default camera position (ECTL Bretigny)
pos = Position()
projector = AircraftProjector(
    camera_lat=pos.latitude,
    camera_lon=pos.longitude,
    camera_alt=pos.altitude,
)

# Aircraft positions from ADS-B feed
lon = np.array([2.30, 2.35, 2.40, 2.45])
lat = np.array([48.55, 48.60, 48.65, 48.70])
alt = np.array([10000.0, 10500.0, 11000.0, 10800.0])  # meters

# Convert to pixel coordinates
px, py = projector.lonlat_to_pixels(lon, lat, alt)

# Plot on your projected image
import matplotlib.pyplot as plt
plt.scatter(px, py, c='red', s=20, label='Aircraft')
```

---

## ✈️ How to Project Flight Paths (Shapely)

**Problem:** You want to draw flight paths or airspace boundaries on a projected image.

**Solution:** Use `project_geometry()` with 3D Shapely geometries.

```python
from shapely.geometry import LineString, Polygon
from skycam.domain import AircraftProjector, Position

pos = Position()
projector = AircraftProjector(
    camera_lat=pos.latitude,
    camera_lon=pos.longitude,
    camera_alt=pos.altitude,
)

# Flight path as 3D LineString (lon, lat, alt_meters)
flight_path = LineString([
    (2.30, 48.55, 10000),
    (2.35, 48.58, 10200),
    (2.40, 48.62, 10500),
    (2.45, 48.65, 10300),
])

# Project to pixel coordinates
path_pixels = projector.project_geometry(flight_path)

# Draw with matplotlib
import matplotlib.pyplot as plt
coords = np.array(path_pixels.coords)
plt.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2, label='Flight path')

# Airspace sector as 3D Polygon
sector = Polygon([
    (2.25, 48.50, 9000),
    (2.50, 48.50, 9000),
    (2.50, 48.70, 9000),
    (2.25, 48.70, 9000),
])
sector_pixels = projector.project_geometry(sector)
```

!!! note "Altitude is required"
    All geometries must have Z coordinates (altitude in meters). 2D geometries will raise `ValueError`.

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
