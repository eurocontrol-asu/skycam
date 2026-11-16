#!/usr/bin/env python3
# ruff: noqa: T201  # CLI utility script - print statements are intentional
"""Generate projected outputs from the refactored implementation.

This script loads the test input images and generates new projected outputs
using the refactored ProjectionService, saving them for visual comparison
with the legacy golden outputs.
"""

from pathlib import Path

from skycam.adapters.calibration import JP2CalibrationLoader
from skycam.adapters.image_io import load_jp2, save_image
from skycam.domain.models import ProjectionSettings
from skycam.domain.projection import ProjectionService


def main() -> None:
    """Generate projection outputs for test images."""
    # Paths
    fixtures_path = Path(__file__).parent / "fixtures"
    calibration_path = fixtures_path / "calibration"
    gold_inputs_path = fixtures_path / "gold_inputs"
    new_outputs_path = fixtures_path / "new_outputs"

    # Create output directory
    new_outputs_path.mkdir(exist_ok=True)

    # Load calibration data
    print("Loading calibration data...")
    loader = JP2CalibrationLoader(calibration_path)
    calibration = loader.load("visible")

    # Create projection service with default settings (matching legacy)
    settings = ProjectionSettings()
    print("Using settings:")
    print(f"  - resolution: {settings.resolution}")
    print(f"  - cloud_height: {settings.cloud_height}")
    print(f"  - square_size: {settings.square_size}")
    print(f"  - max_zenith_angle: {settings.max_zenith_angle}")

    projector = ProjectionService(calibration=calibration, settings=settings)

    # Process each test image
    test_images = [
        "image_20250215080830",
        "image_20250706152130",
    ]

    for image_stem in test_images:
        input_path = gold_inputs_path / f"{image_stem}.jp2"
        output_path = new_outputs_path / f"{image_stem}.jpg"

        print(f"\nProcessing {image_stem}...")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")

        # Load input image
        input_image = load_jp2(input_path)
        print(f"  Input shape: {input_image.shape}")

        # Project image
        projected = projector.project(input_image)
        print(f"  Projected shape: {projected.shape}")

        # Save output
        save_image(projected, output_path, quality=95)
        print(f"  Saved to: {output_path}")

    print("\n✅ Done! Compare new outputs with gold_outputs/")


if __name__ == "__main__":
    main()
