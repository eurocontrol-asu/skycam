"""Calibrated direction/pixel conversion and reusable dense pull sampling maps.

Angles are (azimuth, zenith) in radians; points are (x=column, y=row).
Arrays always use (height, width, ...). No legacy projection behavior is changed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay, QhullError

from skycam.domain.models import CalibrationData

_Scalar = TypeVar("_Scalar", bound=np.generic)
FloatArray = NDArray[np.float64]


def _points(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 2:
        raise ValueError("Points must have shape (..., 2)")
    return result


def _shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or any(
        not isinstance(n, (int, np.integer)) or n < 1 for n in shape
    ):
        raise ValueError("Image shape must be positive (height, width)")
    return int(shape[0]), int(shape[1])


def _bounded_xy(xy: FloatArray, shape: tuple[int, int]) -> FloatArray:
    """Snap numerical noise at pixel centers; reject outside the native extent."""
    height, width = shape
    result = xy.copy()
    finite = np.isfinite(result)
    nearest = np.rint(result[finite])
    result[finite] = np.where(
        np.abs(result[finite] - nearest) < 1e-8, nearest, result[finite]
    )
    valid = (
        np.isfinite(result).all(axis=-1)
        & (result[..., 0] >= 0)
        & (result[..., 0] <= width - 1)
        & (result[..., 1] >= 0)
        & (result[..., 1] <= height - 1)
    )
    result[~valid] = np.nan
    return result


def _sky_plane(directions: FloatArray) -> FloatArray:
    azimuth, zenith = directions[..., 0], directions[..., 1]
    valid = np.isfinite(directions).all(axis=-1) & (zenith >= 0) & (zenith <= np.pi / 2)
    # Equidistant sky plane: continuous across +/-pi and unique at zenith.
    safe_azimuth = np.where(valid, azimuth, 0.0)
    safe_zenith = np.where(valid, zenith, 0.0)
    plane = np.stack(
        [safe_zenith * np.sin(safe_azimuth), safe_zenith * np.cos(safe_azimuth)],
        axis=-1,
    )
    plane[~valid] = np.nan
    return plane


def _linear_sample(image: FloatArray, xy: FloatArray) -> FloatArray:
    """Bilinear sampling including edge centers and only nonzero-weight corners."""
    tail = image.shape[2:]
    flat = xy.reshape(-1, 2)
    valid = np.asarray(np.isfinite(flat).all(axis=1))
    safe = np.where(valid[:, None], flat, 0)
    x0, y0 = np.floor(safe).astype(np.intp).T
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    dx, dy = (safe - np.stack([x0, y0], axis=-1)).T
    result = np.zeros((len(flat), *tail), dtype=np.float64)
    for x, y, weight in (
        (x0, y0, (1 - dx) * (1 - dy)),
        (x1, y0, dx * (1 - dy)),
        (x0, y1, (1 - dx) * dy),
        (x1, y1, dx * dy),
    ):
        active = valid & (weight > 0)
        result[active] += image[y[active], x[active]] * weight[active].reshape(
            (-1, *([1] * len(tail)))
        )
    result[~valid] = np.nan
    return result.reshape((*xy.shape[:-1], *tail))


@dataclass(frozen=True)
class SamplingMap:
    """Reusable pull map from a target grid to a source image.

    ``source_xy[target_row, target_col]`` is a source (x, y) pixel center.
    Invalid coordinates become NaN and sample to ``fill_value``. Source and
    target shapes are explicit; no margins, resizing, or axis swaps are inferred.
    Maps own read-only arrays and may be shared by callers after construction.
    """

    source_shape: tuple[int, int]
    source_xy: FloatArray
    valid: NDArray[np.bool_] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shape = _shape(self.source_shape)
        xy = _points(self.source_xy)
        if xy.ndim != 3 or min(xy.shape[:2]) < 1:
            raise ValueError("Sampling coordinates must have shape (H, W, 2)")
        xy = _bounded_xy(xy, shape)
        valid = np.asarray(np.isfinite(xy).all(axis=-1))
        xy.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "source_shape", shape)
        object.__setattr__(self, "source_xy", xy)
        object.__setattr__(self, "valid", valid)

    @property
    def target_shape(self) -> tuple[int, int]:
        """Target (height, width)."""
        return self.source_xy.shape[0], self.source_xy.shape[1]

    def _check_image(self, image: NDArray[_Scalar]) -> None:
        if image.ndim < 2 or image.shape[:2] != self.source_shape:
            raise ValueError(f"Expected source image shape {self.source_shape}, ...")

    def _nearest(
        self, image: NDArray[_Scalar], fill_value: int | float | bool
    ) -> NDArray[_Scalar]:
        self._check_image(image)
        xy = self.source_xy[self.valid]
        # Half-pixel ties go to the larger index, independent of dtype.
        x, y = np.floor(xy + 0.5).astype(np.intp).T
        result = np.full(
            (*self.target_shape, *image.shape[2:]), fill_value, dtype=image.dtype
        )
        result[self.valid] = image[y, x]
        return result

    def sample_labels(
        self, labels: NDArray[_Scalar], *, fill_value: int | bool = 0
    ) -> NDArray[_Scalar]:
        """Nearest sampling of (H, W, ...) integer/boolean labels, dtype intact."""
        if labels.dtype.kind not in "biu":
            raise TypeError("Labels must have boolean or integer dtype")
        if not isinstance(fill_value, (bool, int, np.integer)):
            raise TypeError("Label fill_value must be an integer or boolean")
        if labels.dtype.kind == "b":
            if fill_value not in (0, 1):
                raise ValueError("Boolean fill_value must be 0 or 1")
        else:
            limits = np.iinfo(labels.dtype.str)
            if not limits.min <= fill_value <= limits.max:
                raise ValueError("fill_value is outside the label dtype range")
        return self._nearest(labels, fill_value)

    def sample_instances(
        self, masks: NDArray[_Scalar], *, fill_value: int | bool = 0
    ) -> NDArray[_Scalar]:
        """Sample independent masks (N, H, W), preserving overlap and dtype."""
        if masks.ndim != 3:
            raise ValueError("Instance masks must have shape (N, H, W)")
        sampled = self.sample_labels(np.moveaxis(masks, 0, -1), fill_value=fill_value)
        return np.moveaxis(sampled, -1, 0)

    def sample_image(
        self,
        image: NDArray[np.generic],
        *,
        interpolation: Literal["linear", "nearest"] = "linear",
        fill_value: float = 0.0,
    ) -> FloatArray:
        """Sample continuous (H, W, ...) images/opacity, returning float64.

        Use sample_labels/sample_instances for discrete data. NaN image values
        propagate only from corners with nonzero interpolation weight.
        """
        self._check_image(image)
        if image.dtype.kind not in "biuf":
            raise TypeError("Images must have real numeric dtype")
        continuous = np.asarray(image, dtype=np.float64)
        if interpolation == "nearest":
            return self._nearest(continuous, fill_value)
        if interpolation != "linear":
            raise ValueError("interpolation must be 'linear' or 'nearest'")
        result = _linear_sample(continuous, self.source_xy)
        result[~self.valid] = fill_value
        return result

    def save(self, path: str | Path) -> None:
        """Save coordinates and source shape as NPZ to exactly the given path."""
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream, source_shape=self.source_shape, source_xy=self.source_xy
            )

    @classmethod
    def load(cls, path: str | Path) -> "SamplingMap":
        """Load an NPZ map without pickle; callers own calibration provenance."""
        with np.load(path, allow_pickle=False) as data:
            shape = data["source_shape"]
            if shape.shape != (2,) or shape.dtype.kind not in "iu":
                raise ValueError("Invalid saved source shape")
            return cls((int(shape[0]), int(shape[1])), data["source_xy"])


class CalibratedReprojection:
    """Bidirectional native pixel/direction interpolation for one calibration.

    Native azimuth and zenith arrays have shape (H, W), in radians. The output
    convention is the input azimuth plus explicit ``azimuth_offset``. In
    particular JP2CalibrationLoader returns 2*pi*raw/64000-pi; add offset=pi
    to obtain the kit geographic convention 2*pi*raw/64000 (modulo 2*pi).
    Default offset=0 keeps the loader/historical GVCCS convention intact.

    Interpolation uses (zenith*sin(azimuth), zenith*cos(azimuth)), so both the
    seam and zenith are regular. Only the upper hemisphere [0, pi/2] is valid.
    Calibration must describe a locally one-to-one camera. Inverse Delaunay
    triangulation is built once, lazily, per instance; triangles joining
    nonadjacent native pixels are excluded to avoid bridging calibration holes.
    Forward bilinear and inverse triangular interpolation have finite precision.
    """

    def __init__(
        self,
        azimuth: ArrayLike,
        zenith: ArrayLike,
        *,
        azimuth_offset: float = 0.0,
    ) -> None:
        az = np.asarray(azimuth, dtype=np.float64)
        ze = np.asarray(zenith, dtype=np.float64)
        if az.ndim != 2 or az.shape != ze.shape or min(az.shape) < 2:
            raise ValueError("Calibration maps must have matching (H, W), H,W >= 2")
        if not np.isfinite(azimuth_offset):
            raise ValueError("azimuth_offset must be finite")
        self._plane = _sky_plane(np.stack([az + azimuth_offset, ze], axis=-1))
        self._plane.setflags(write=False)
        self._inverse: tuple[Delaunay, FloatArray] | None = None

    @classmethod
    def from_calibration(
        cls, calibration: CalibrationData, *, azimuth_offset: float = 0.0
    ) -> "CalibratedReprojection":
        """Use a CalibrationData object, validating its declared image shape."""
        result = cls(
            np.asarray(calibration.azimuth_array, dtype=np.float64),
            np.asarray(calibration.zenith_array, dtype=np.float64),
            azimuth_offset=azimuth_offset,
        )
        if result.image_shape != calibration.image_size:
            raise ValueError("Calibration image_size disagrees with map shape")
        return result

    @property
    def image_shape(self) -> tuple[int, int]:
        """Native image (height, width)."""
        return self._plane.shape[0], self._plane.shape[1]

    @property
    def directions(self) -> FloatArray:
        """Native (H, W, 2) angular grid, suitable as another camera's target."""
        y, x = np.indices(self.image_shape)
        return self.pixels_to_directions(np.stack([x, y], axis=-1))

    def pixels_to_directions(self, xy: ArrayLike) -> FloatArray:
        """Convert (..., 2) native (x,y) centers to (azimuth,zenith) radians.

        Invalid pixels or contributing invalid calibration corners return NaN.
        Azimuth is canonical [-pi, pi), and exactly zero at zenith.
        """
        points = _bounded_xy(_points(xy), self.image_shape)
        plane = _linear_sample(self._plane, points)
        zenith = np.hypot(plane[..., 0], plane[..., 1])
        azimuth = (np.arctan2(plane[..., 0], plane[..., 1]) + np.pi) % (
            2 * np.pi
        ) - np.pi
        azimuth = np.where(zenith < 1e-14, 0.0, azimuth)
        return np.stack([azimuth, zenith], axis=-1)

    def _ensure_inverse(self) -> tuple[Delaunay, FloatArray]:
        if self._inverse is None:
            valid = np.isfinite(self._plane).all(axis=-1)
            y, x = np.indices(self.image_shape)
            pixels = np.stack([x[valid], y[valid]], axis=-1).astype(np.float64)
            try:
                triangulation = Delaunay(self._plane[valid])
            except QhullError as exc:
                raise ValueError(
                    "Calibration needs nondegenerate valid directions"
                ) from exc
            self._inverse = triangulation, pixels
        return self._inverse

    def directions_to_pixels(self, directions: ArrayLike) -> FloatArray:
        """Convert (..., 2) (azimuth,zenith) radians to native (x,y) centers.

        Azimuth is periodic. NaN/inf, negative/below-horizon zenith, directions
        outside the calibrated hull and across calibration holes return NaN.
        """
        plane = _sky_plane(_points(directions))
        flat = plane.reshape(-1, 2)
        result = np.full_like(flat, np.nan)
        valid = np.isfinite(flat).all(axis=-1)
        if not np.any(valid):
            return result.reshape(plane.shape)
        triangulation, pixels = self._ensure_inverse()
        indices = np.flatnonzero(valid)
        simplex = triangulation.find_simplex(flat[indices])
        inside = simplex >= 0
        indices, simplex = indices[inside], simplex[inside]
        transforms = triangulation.transform[simplex]
        bary = np.einsum(
            "nij,nj->ni", transforms[:, :2], flat[indices] - transforms[:, 2]
        )
        weights = np.column_stack([bary, 1 - bary.sum(axis=-1)])
        vertices = pixels[triangulation.simplices[simplex]]
        active = weights > 1e-8
        span = np.max(np.where(active[..., None], vertices, -np.inf), axis=1) - np.min(
            np.where(active[..., None], vertices, np.inf), axis=1
        )
        local = (span <= 1).all(axis=-1)
        result[indices[local]] = np.einsum(
            "ni,nij->nj", weights[local], vertices[local]
        )
        return _bounded_xy(result.reshape(plane.shape), self.image_shape)

    def sampling_map(
        self,
        target_directions: ArrayLike,
        *,
        crop: tuple[int, int, int, int] | None = None,
        target_shape: tuple[int, int] | None = None,
    ) -> SamplingMap:
        """Build a reusable dense source-to-target pull map using an exact grid.

        target_directions is (H, W, 2) in this object's angular convention.
        crop=(top,bottom,left,right) explicitly removes grid borders, in pixels.
        target_shape asserts the final size; it never resizes/trims a grid.
        For historical 1026 GVCCS grids and 1024 frames specify crop=(1,1,1,1)
        only after establishing that registration for the supplied grid.
        """
        grid = _points(target_directions)
        if grid.ndim != 3 or min(grid.shape[:2]) < 1:
            raise ValueError("Target directions must have shape (H, W, 2)")
        if crop is not None:
            if len(crop) != 4 or any(
                not isinstance(n, (int, np.integer)) or n < 0 for n in crop
            ):
                raise ValueError("crop must be nonnegative (top,bottom,left,right)")
            top, bottom, left, right = crop
            height, width = grid.shape[:2]
            if top + bottom >= height or left + right >= width:
                raise ValueError("crop removes the entire target grid")
            grid = grid[top : height - bottom, left : width - right]
        if target_shape is not None and grid.shape[:2] != _shape(target_shape):
            raise ValueError("Target grid shape mismatch; provide an explicit crop")
        return SamplingMap(self.image_shape, self.directions_to_pixels(grid))
