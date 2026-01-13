"""Tests for image I/O adapter."""

from pathlib import Path

import numpy as np
import pytest

from skycam.adapters.image_io import load_image, load_jp2, load_jpg
from skycam.domain.exceptions import SkycamError


class TestLoadJP2:
    """Tests for load_jp2 function."""

    def test_load_jp2_returns_ndarray(
        self,
        gold_inputs_path: Path,
    ) -> None:
        """load_jp2 returns numpy array."""
        path = gold_inputs_path / "image_20250215080830.jp2"
        img = load_jp2(path)
        assert isinstance(img, np.ndarray)

    def test_load_jp2_dtype(
        self,
        gold_inputs_path: Path,
    ) -> None:
        """load_jp2 returns uint8."""
        path = gold_inputs_path / "image_20250215080830.jp2"
        img = load_jp2(path)
        assert img.dtype == np.uint8

    def test_load_jp2_is_rgb(
        self,
        gold_inputs_path: Path,
    ) -> None:
        """load_jp2 returns 3-channel RGB."""
        path = gold_inputs_path / "image_20250215080830.jp2"
        img = load_jp2(path)
        assert len(img.shape) == 3
        assert img.shape[2] == 3  # RGB


class TestLoadJPG:
    """Tests for load_jpg function."""

    def test_load_jpg_returns_ndarray(
        self,
        gold_outputs_path: Path,
    ) -> None:
        """load_jpg returns numpy array."""
        path = gold_outputs_path / "image_20250215080830.jpg"
        img = load_jpg(path)
        assert isinstance(img, np.ndarray)

    def test_load_jpg_dtype(
        self,
        gold_outputs_path: Path,
    ) -> None:
        """load_jpg returns uint8."""
        path = gold_outputs_path / "image_20250215080830.jpg"
        img = load_jpg(path)
        assert img.dtype == np.uint8


class TestLoadImage:
    """Tests for load_image function."""

    def test_missing_file_raises_error(
        self,
        fixtures_path: Path,
    ) -> None:
        """Missing file raises SkycamError."""
        with pytest.raises(SkycamError, match="not found"):
            load_image(fixtures_path / "nonexistent.jpg")
