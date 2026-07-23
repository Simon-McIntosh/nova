"""Tolerance classes for the characterization gate.

The gate is tolerance-based, not byte-identity. Positional, gap and deviation
outputs expressed in millimetres must agree to 1 micron -- three decimal places
on a millimetre, well below the metrology noise floor. Every other kind of
output carries its own class, set at least two orders of magnitude below the
physical noise of that quantity.

A class is an absolute + relative tolerance pair consumed by
:func:`compare`, which mirrors ``numpy.allclose`` semantics
(``|a - b| <= atol + rtol * |b|``) but also returns the realized deviations so
a failure reports how far out it landed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    """An absolute/relative tolerance pair with a human-readable rationale."""

    atol: float
    rtol: float
    note: str


# Millimetre lengths: 1 micron absolute (three decimal places on mm). This is
# the headline gate for positional, gap and deviation outputs.
LENGTH_MM = Tolerance(atol=1e-3, rtol=0.0, note="1 micron on a millimetre")

# Metre lengths: 1 micron absolute expressed in metres.
LENGTH_M = Tolerance(atol=1e-6, rtol=0.0, note="1 micron in metres")

# Angles in radians: two orders below a ~1e-4 rad alignment noise floor.
ANGLE_RAD = Tolerance(atol=1e-6, rtol=1e-9, note="1e-6 rad, below alignment noise")

# Angles in degrees: the radian floor carried through 180/pi.
ANGLE_DEG = Tolerance(atol=5.7e-5, rtol=1e-9, note="1e-6 rad expressed in degrees")

# Dimensionless spectral / Fourier coefficients and normalized quantities.
COEFFICIENT = Tolerance(atol=1e-9, rtol=1e-9, note="tight numeric agreement")

# Fitted hyperparameters (GP lengthscales, nuggets) -- looser, optimizer-bound.
HYPERPARAMETER = Tolerance(atol=1e-6, rtol=1e-6, note="optimizer-reproducible")

# Fallback for anything not explicitly classified: tight, since the pinned
# environment is usually byte-stable.
DEFAULT = Tolerance(atol=1e-9, rtol=1e-9, note="pinned-environment numeric agreement")


CLASSES: dict[str, Tolerance] = {
    "length_mm": LENGTH_MM,
    "length_m": LENGTH_M,
    "angle_rad": ANGLE_RAD,
    "angle_deg": ANGLE_DEG,
    "coefficient": COEFFICIENT,
    "hyperparameter": HYPERPARAMETER,
    "default": DEFAULT,
}


def tolerance_for(name: str) -> Tolerance:
    """Return the :class:`Tolerance` for a class name (``default`` if unknown)."""
    return CLASSES.get(name, DEFAULT)


@dataclass
class Comparison:
    """Outcome of comparing a candidate array against a golden array."""

    passed: bool
    max_abs_dev: float
    max_rel_dev: float
    tolerance: str
    detail: str = ""


def compare(candidate: np.ndarray, golden: np.ndarray, tolerance: str) -> Comparison:
    """Compare ``candidate`` against ``golden`` under a named tolerance class."""
    tol = tolerance_for(tolerance)
    cand = np.asarray(candidate, dtype=np.float64)
    gold = np.asarray(golden, dtype=np.float64)

    if cand.shape != gold.shape:
        return Comparison(
            passed=False,
            max_abs_dev=float("inf"),
            max_rel_dev=float("inf"),
            tolerance=tolerance,
            detail=f"shape mismatch: candidate {cand.shape} vs golden {gold.shape}",
        )

    finite = np.isfinite(gold) & np.isfinite(cand)
    nan_mismatch = np.isnan(gold) != np.isnan(cand)
    if nan_mismatch.any():
        return Comparison(
            passed=False,
            max_abs_dev=float("inf"),
            max_rel_dev=float("inf"),
            tolerance=tolerance,
            detail=f"NaN pattern differs at {int(nan_mismatch.sum())} element(s)",
        )

    if not finite.any():
        # All non-finite but patterns match (handled above): nothing to compare.
        return Comparison(True, 0.0, 0.0, tolerance, "no finite elements")

    abs_dev = np.abs(cand[finite] - gold[finite])
    denom = np.abs(gold[finite])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_dev = np.where(denom > 0, abs_dev / denom, 0.0)
    max_abs = float(abs_dev.max())
    max_rel = float(rel_dev.max())

    passed = bool(np.all(abs_dev <= tol.atol + tol.rtol * denom))
    detail = (
        ""
        if passed
        else (
            f"max |dev|={max_abs:.3e} exceeds atol {tol.atol:.1e} "
            f"+ rtol {tol.rtol:.1e} * |golden|"
        )
    )
    return Comparison(passed, max_abs, max_rel, tolerance, detail)
