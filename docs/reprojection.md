# Calibrated reprojection

`CalibratedReprojection` converts calibrated native fisheye pixels to angular
directions and back. `SamplingMap` stores a dense **pull map**: for each target
pixel it records the corresponding source pixel. Build once, then reuse for RGB,
opacity, labels and separate instance masks. These APIs are independent of
`ProjectionService`, `JP2CalibrationLoader` and `AircraftProjector`; their existing
behavior is unchanged.

```python
import numpy as np
from skycam.domain import CalibratedReprojection, SamplingMap

# calibration is a CalibrationData returned by JP2CalibrationLoader.load().
source = CalibratedReprojection.from_calibration(calibration)
xy = source.directions_to_pixels([[0.2, 0.4]])
angles = source.pixels_to_directions(xy)

# Exact externally supplied (H, W, 2) azimuth/zenith grid, in radians.
# Use the historical GVCCS grid with the default legacy azimuth convention.
# For the independently registered 1026 grid -> 1024 frame use explicit crop:
mapping = source.sampling_map(
    gvccs_angle_grid, crop=(1, 1, 1, 1), target_shape=(1024, 1024)
)
rgb_projected = mapping.sample_image(native_rgb)  # float64; linear by default
opacity_projected = mapping.sample_image(native_opacity)
labels_projected = mapping.sample_labels(native_labels)  # bool/integer unchanged
instances_projected = mapping.sample_instances(native_instances)  # (N, H, W)
mapping.save("native_to_gvccs.npz")
mapping = SamplingMap.load("native_to_gvccs.npz")
```

For arbitrary camera-to-camera reprojection, construct a second calibrated object
in the **same angular convention** and call
`source.sampling_map(target.directions, target_shape=target.image_shape)`.
The constructor also accepts separate native `azimuth` and `zenith` arrays.

- Arrays use `(height, width, ...)`, indexed `[row, column]`. Point arrays have
  shape `(..., 2)` and contain **`(x=column, y=row)`**. Integer coordinates denote
  pixel centers. RGB channels follow the source image's existing order.
- Direction arrays contain **`(azimuth, zenith)` in radians**. The loader's
  azimuth is `2*pi*raw/64000 - pi`, whereas the kit's geographic azimuth is
  `2*pi*raw/64000`. Use `azimuth_offset=np.pi` when constructing from loader data
  for geographic directions. Default `0` preserves the legacy convention used
  by historical GVCCS grids. Outputs wrap azimuth to `[-pi, pi)`; this conversion
  does not change the loader or rotate existing projection outputs.
- The sky plane is `(zenith*sin(azimuth), zenith*cos(azimuth))`. Interpolation
  crosses the azimuth seam continuously and has one point at zenith. Forward
  conversion uses bilinear interpolation in that plane; inverse conversion uses
  triangular interpolation. Point round trips have interpolation error on
  nonlinear calibrations. Returned azimuth is zero at zenith.
- Upper-hemisphere zenith `[0, pi/2]`, finite azimuth and calibrated support are
  required. NaN/inf, outside-image pixels, outside-hull directions and triangles
  bridging nonadjacent native pixels yield NaN points. Invalid contributing
  forward calibration corners also yield NaN; zero-weight corners do not.
  Invalid map positions have `valid=False` and sample to `fill_value` (default
  zero). Use `mapping.valid` to distinguish outside support from background.
- `source_shape`, `target_shape`, `source_xy` and `valid` are exposed on each map.
  Sampling checks the actual source dimensions. `crop=(top,bottom,left,right)`
  removes explicit pixel borders from the supplied target grid, without angular
  interpolation. `target_shape` asserts the resulting size and never infers a
  crop or resizes a grid. Without a crop the supplied grid is used exactly.
- `sample_labels` accepts `(H,W,...)` boolean/integer arrays and preserves dtype
  and all IDs, including integers beyond floating point precision.
  `sample_instances` accepts `(N,H,W)` boolean/integer masks and samples each
  independently, preserving overlapping instances. Both always use nearest
  neighbours; half-pixel ties go to the larger index. A dense raster mask is the
  evaluation path: projecting polygon vertices alone misses nonlinear curves
  and thin trails. Nearest sampling can still lose subpixel trails when a target
  grid is coarser; it does not perform area coverage or antialiasing.
- `sample_image` accepts real numeric `(H,W,...)` arrays, defaults to bilinear
  interpolation and always returns float64, without clipping. It also supports
  `interpolation="nearest"`. Keep discrete labels on the label/instance methods.
  Last-row/column pixel centers are included. NaNs in images propagate from
  nonzero-weight corners. `fill_value=np.nan` is useful for continuous data.
- Each calibration object caches its inverse triangulation in memory, building
  it lazily on the first valid inverse query. Reuse the object and saved maps to
  avoid repeated triangulations. Calibration must be locally one-to-one;
  degenerate calibrations raise `ValueError`. NPZ maps store source shape and
  coordinates without pickle. Callers must track calibration, angular convention
  and target-grid provenance; a matching shape alone does not verify registration.
