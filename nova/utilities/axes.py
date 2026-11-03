"""Validation of uniformly sampled coordinate axes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def uniform_axis(
    values: ArrayLike,
    name: str,
    *,
    minimum_size: int = 2,
    tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> tuple[NDArray[np.float64], float]:
    """Return an endpoint-preserving axis and step after a precision-aware check."""

    stored = np.asarray(values)
    if stored.ndim != 1 or stored.size < minimum_size:
        raise ValueError(
            f"{name} must be one dimensional with at least {minimum_size} nodes"
        )
    axis = np.asarray(stored, dtype=np.float64)
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    precision = stored.dtype if np.issubdtype(stored.dtype, np.floating) else axis.dtype
    reconstructed = np.linspace(axis[0], axis[-1], axis.size)
    stored_reconstruction = np.linspace(
        stored[0], stored[-1], stored.size, dtype=precision
    )
    precision_floor = 16.0 * np.finfo(precision).eps * max(1.0, abs(float(axis[-1])))
    allowed = max(
        float(tolerance),
        float(relative_tolerance) * max(1.0, abs(float(axis[-1]))),
        precision_floor,
    )
    deviation = float(np.max(np.abs(stored - stored_reconstruction)))
    if deviation > allowed:
        raise ValueError(f"{name} must be uniformly spaced")
    return reconstructed, float((axis[-1] - axis[0]) / (axis.size - 1))
