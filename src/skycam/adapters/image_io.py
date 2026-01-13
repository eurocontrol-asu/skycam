"""Image I/O adapter for loading and saving images."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from skycam.domain.exceptions import SkycamError


def load_jp2(path: Path | str) -> NDArray[np.uint8]:
    """Load a JP2 image file.

    Args:
        path: Path to JP2 file

    Returns:
        Image as RGB uint8 numpy array (H, W, C)

    Raises:
        SkycamError: If file cannot be loaded
    """
    return load_image(Path(path))


def load_jpg(path: Path | str) -> NDArray[np.uint8]:
    """Load a JPEG image file.

    Args:
        path: Path to JPEG file

    Returns:
        Image as RGB uint8 numpy array (H, W, C)

    Raises:
        SkycamError: If file cannot be loaded
    """
    return load_image(Path(path))


def load_image(path: Path | str) -> NDArray[np.uint8]:
    """Load an image file (any format supported by OpenCV).

    Args:
        path: Path to image file

    Returns:
        Image as RGB uint8 numpy array (H, W, C)

    Raises:
        SkycamError: If file cannot be loaded
    """
    path = Path(path)
    if not path.exists():
        msg = f"Image file not found: {path}"
        raise SkycamError(msg)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        msg = f"Failed to load image: {path}"
        raise SkycamError(msg)

    # OpenCV loads as BGR, convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def save_image(
    image: NDArray[np.uint8],
    path: Path | str,
    format_hint: Literal["jpg", "png", "jp2"] | None = None,
    quality: int = 95,
) -> None:
    """Save an image to file.

    Args:
        image: RGB uint8 numpy array (H, W, C)
        path: Output path
        format_hint: Optional format hint (uses extension if not provided)
        quality: JPEG quality (0-100)

    Raises:
        SkycamError: If file cannot be saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    # Set encoding parameters
    params: list[int] = []
    ext = format_hint or path.suffix.lstrip(".").lower()
    if ext in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    try:
        success = cv2.imwrite(str(path), bgr, params)
        if not success:
            msg = f"Failed to save image: {path}"
            raise SkycamError(msg)
    except cv2.error as e:
        msg = f"Failed to save image: {path} ({e})"
        raise SkycamError(msg) from e
