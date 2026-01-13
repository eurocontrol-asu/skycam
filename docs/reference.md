# 📚 API Reference

Complete reference for all public classes, functions, and configuration options.

---

## 🔧 Configuration

### Environment Variables

All settings support environment variable overrides via `SKYCAM_` prefix:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SKYCAM_CALIBRATION_DIR` | `Path` | `calibration` | Directory containing JP2 calibration files |
| `SKYCAM_DATA_DIR` | `Path` | `data` | Directory for input/output data |
| `SKYCAM_CATEGORY` | `str` | `visible` | Camera category (`visible`, `infrarouge`) |

### `SkycamSettings`

::: skycam.config.SkycamSettings
    options:
      show_source: false
      members: false

```python
from skycam.config import SkycamSettings

settings = SkycamSettings(
    calibration_dir=Path("/data/calibration"),
    category="infrarouge",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `calibration_dir` | `Path` | `calibration` | JP2 calibration files directory |
| `data_dir` | `Path` | `data` | Input/output data directory |
| `category` | `str` | `visible` | Camera category |
| `position` | `Position` | ECTL Bretigny | Camera geographic position |
| `camera` | `CameraConfig` | Default | Camera metadata |
| `projection` | `ProjectionSettings` | Default | Projection parameters |

---

## 🎯 Domain Layer

### `ProjectionService`

Main projection engine. Converts fisheye images to regular grids.

```python
from skycam.domain.projection import ProjectionService
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calibration` | `CalibrationData` | *required* | Loaded calibration arrays |
| `settings` | `ProjectionSettings` | *required* | Projection configuration |
| `calibration_path` | `Path \| None` | `None` | Path for coordinate caching |
| `lazy_init` | `bool` | `False` | Defer interpolator build |

#### Methods

##### `project(image, as_uint8=True)`

Project a fisheye image to the output grid.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `NDArray[np.uint8]` | Input image `(H, W, C)` |
| `as_uint8` | `bool` | Return uint8 (True) or float64 (False) |

**Returns:** `NDArray` with shape `(resolution, resolution, C)`

**Raises:** `ProjectionError` if projection fails

##### `ensure_initialized()`

Explicitly build interpolators. Called automatically by `project()`.

##### `calculate_azimuth_zenith(...)`

Calculate viewing angles from observer to target.

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_lat` | `float` | Target latitude (degrees) |
| `target_lon` | `float` | Target longitude (degrees) |
| `target_alt` | `float` | Target altitude (meters) |
| `observer_lat` | `float` | Observer latitude (degrees) |
| `observer_lon` | `float` | Observer longitude (degrees) |
| `observer_alt` | `float` | Observer altitude (meters) |

**Returns:** `tuple[float, float]` — (azimuth_degrees, zenith_degrees)

##### `calculate_latitude_longitude(...)`

Calculate target position from viewing angles.

| Parameter | Type | Description |
|-----------|------|-------------|
| `azimuth` | `float` | Azimuth angle (degrees) |
| `zenith` | `float` | Zenith angle (degrees) |
| `target_altitude` | `float` | Target altitude (meters) |
| `observer_lat` | `float` | Observer latitude (degrees) |
| `observer_lon` | `float` | Observer longitude (degrees) |
| `observer_alt` | `float` | Observer altitude (meters) |

**Returns:** `tuple[float, float]` — (latitude, longitude) in degrees

---

### `ProjectionSettings`

Immutable configuration for the projection algorithm.

```python
from skycam.domain.models import ProjectionSettings
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `resolution` | `int` | `1024` | `64 ≤ x ≤ 8192` | Output grid size (pixels) |
| `cloud_height` | `float` | `10000.0` | `≥ 100` | Assumed cloud height (meters) |
| `square_size` | `float` | `75000.0` | `≥ 1000` | Physical grid size (meters) |
| `max_zenith_angle` | `float` | `80.0` | `0 ≤ x ≤ 90` | Maximum valid zenith (degrees) |

---

### `CalibrationData`

Container for loaded calibration arrays.

```python
from skycam.domain.models import CalibrationData
```

| Field | Type | Description |
|-------|------|-------------|
| `azimuth_array` | `NDArray` | Azimuth map in radians `[-π, π]` |
| `zenith_array` | `NDArray` | Zenith map in radians `[0, π/2]` |
| `image_size` | `tuple[int, int]` | Array dimensions `(height, width)` |

---

### `Position`

Geographic position (WGS84).

```python
from skycam.domain.models import Position
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `longitude` | `float` | `2.3468` | Longitude (degrees) |
| `latitude` | `float` | `48.6005` | Latitude (degrees) |
| `altitude` | `float` | `90.0` | Altitude (meters) |

---

### `CameraConfig`

Camera metadata.

```python
from skycam.domain.models import CameraConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"Bretigny..."` | Camera name |
| `location` | `str` | `"ECTL Bretigny"` | Installation location |
| `camera_type` | `Literal[...]` | `"hemispherical"` | Lens type |
| `installation_date` | `date \| None` | `None` | Installation date |

---

### Exceptions

```python
from skycam.domain import SkycamError, CalibrationError, ProjectionError, ConfigurationError
```

| Exception | Description |
|-----------|-------------|
| `SkycamError` | Base exception for all skycam errors |
| `CalibrationError` | Calibration data cannot be loaded or is invalid |
| `ProjectionError` | Projection calculation failed |
| `ConfigurationError` | Configuration is invalid or missing |

---

## 🔌 Adapters Layer

### `JP2CalibrationLoader`

Loads JP2 calibration files and converts to radians.

```python
from skycam.adapters import JP2CalibrationLoader
```

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `calibration_dir` | `Path` | Directory with `azimuth_*.jp2` and `zenith_*.jp2` files |

#### Methods

##### `load(category="visible")`

Load calibration for specified camera category.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | `str` | `"visible"` | Camera category |

**Returns:** `CalibrationData`

**Raises:** `CalibrationError` if files not found or shapes mismatch

---

### Image I/O Functions

```python
from skycam.adapters import load_jp2, load_jpg, load_image, save_image
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_jp2` | `(path: Path) → NDArray[uint8]` | Load JP2 as RGB array |
| `load_jpg` | `(path: Path) → NDArray[uint8]` | Load JPEG as RGB array |
| `load_image` | `(path: Path) → NDArray[uint8]` | Load any OpenCV format |
| `save_image` | `(image, path, format_hint?, quality?)` | Save with auto-detection |

---

## 🛠️ Makefile Commands

```bash
make install    # uv sync --all-groups
make check      # lint + audit + test
make lint       # ruff check + ruff format --check + mypy
make format     # ruff format + ruff check --fix
make test       # pytest
make audit      # pip-audit
make docs-serve # mkdocs serve
make clean      # Remove build artifacts
```
