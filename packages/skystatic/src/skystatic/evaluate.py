"""Agreement between a computed mask and a reference mask of the same camera."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.ndimage import binary_dilation

__all__ = ["Agreement", "agreement"]


class Agreement(BaseModel):
    """Overlap of two masks inside a domain; fractions are of the domain."""

    computed_fraction: float
    reference_fraction: float
    iou: float
    precision: float
    recall: float
    precision_within_tolerance: float
    recall_within_tolerance: float
    tolerance: int


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def agreement(
    computed: NDArray[np.bool_],
    reference: NDArray[np.bool_],
    domain: NDArray[np.bool_],
    tolerance: int = 3,
) -> Agreement:
    """Compare ``computed`` to ``reference`` inside ``domain``.

    ``precision`` is the share of computed pixels that the reference also marks, ``recall`` the
    share of reference pixels that are computed. The tolerant variants accept a counterpart
    within ``tolerance`` pixels, which separates a shifted border from a real disagreement.
    """
    a = computed & domain
    b = reference & domain
    near_a = binary_dilation(a, iterations=tolerance) if tolerance else a
    near_b = binary_dilation(b, iterations=tolerance) if tolerance else b
    size = int(domain.sum())
    return Agreement(
        computed_fraction=_ratio(int(a.sum()), size),
        reference_fraction=_ratio(int(b.sum()), size),
        iou=_ratio(int((a & b).sum()), int((a | b).sum())),
        precision=_ratio(int((a & b).sum()), int(a.sum())),
        recall=_ratio(int((a & b).sum()), int(b.sum())),
        precision_within_tolerance=_ratio(int((a & near_b).sum()), int(a.sum())),
        recall_within_tolerance=_ratio(int((b & near_a).sum()), int(b.sum())),
        tolerance=tolerance,
    )
