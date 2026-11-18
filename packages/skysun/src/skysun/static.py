"""What skysun takes from skystatic: the sky pixels (the skyline) and the ring circles."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skystatic import FrozenInstallation, OpticsRings, load_optics, sky_mask

__all__ = ["StaticSource"]


class StaticSource:
    """A skystatic product: a run folder (one daily mask per date and ``optics.json``) or a
    frozen installation NPZ (one mask for every date of its validity, circles inside)."""

    def __init__(self, path: Path, *, allow_unreliable: bool = False):
        self.path = path
        self.allow_unreliable = allow_unreliable
        self._frozen = FrozenInstallation.from_file(path) if path.suffix == ".npz" else None
        if self._frozen is None and not path.is_dir():
            raise FileNotFoundError(f"{path}: neither a skystatic run folder nor a frozen NPZ")
        self._cache: dict[tuple[date, tuple[int, int]], NDArray[np.bool_]] = {}

    def optics(self, size: tuple[int, int]) -> OpticsRings | None:
        """The ring circles at ``size`` (width, height); None when skystatic fitted none."""
        if self._frozen is not None:
            product = self._frozen.optics
            return None if product is None else product.scaled(size)
        return load_optics(self.path, size)

    def sky(self, day: date, size: tuple[int, int]) -> NDArray[np.bool_]:
        """Class-0 pixels for ``day`` at ``size``. A frozen installation must be valid then."""
        key = (day, size)
        if key not in self._cache:
            self._cache.clear()  # one day at a time: frames come in time order
            self._cache[key] = self._load(day, size)
        return self._cache[key]

    def _load(self, day: date, size: tuple[int, int]) -> NDArray[np.bool_]:
        if self._frozen is None:
            moment = datetime(day.year, day.month, day.day)  # the file names' clock (skystatic)
            return sky_mask(self.path, moment, size, allow_unreliable=self.allow_unreliable)
        meta = self._frozen.metadata
        if not meta.valid_from <= day < meta.valid_until:
            raise ValueError(f"{day}: outside the frozen installation's validity period")
        classes = Image.fromarray(np.asarray(self._frozen.classes))
        return np.asarray(np.asarray(classes.resize(size, Image.Resampling.NEAREST)) == 0)
