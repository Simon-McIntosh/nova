"""Locate the plasma edge on a Thomson chord from the measured profile alone.

The model is the modified hyperbolic tangent used for edge profiles:

    y(R) = b + (h - b) / 2 * (mtanh(x, s) + 1),   x = (R0 - R) / w

on the outboard side (``x = (R - R0) / w`` inboard), with

    mtanh(x, s) = ((1 + s x) e^x - e^-x) / (e^x + e^-x).

``R0`` is the centre of the steep gradient, ``w`` its half-width, ``h`` and
``b`` the levels inside and outside, and ``s`` a linear core slope so the
interior profile does not pull the step. The fit is weighted by each sample's
own stored uncertainty.

The edge is where the fitted profile falls through the separatrix
temperature, ``SEPARATRIX_TEMPERATURE``. Measured 27079 profiles are broad,
L-mode-like steps 0.09 to 0.16 m wide, so no fixed point on the step (its
centre or foot) tracks the edge: the foot sits about 0.1 m inside EFIT's
outboard boundary through the flat top. The two-point scrape-off-layer model
puts the upstream separatrix temperature of a MAST-sized device near 40 eV,
and the fitted profile crosses that level within about 0.03 m of EFIT's
outboard boundary. The threshold is a stated model parameter; nothing here
reads an equilibrium, which is what makes it a check on the solved boundary
rather than a restatement of it.

A side is reported only when its fit resolves it: the step must lie inside the
sampled radii, rise by at least a factor of two, and carry a positive finite
width. Otherwise the side is ``None``, because a step fitted outside the data
is an extrapolation and not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Upstream separatrix electron temperature of the two-point model, in eV.
SEPARATRIX_TEMPERATURE = 40.0

#: The step must rise at least this factor from outside to inside.
MINIMUM_CONTRAST = 2.0

#: Fewest samples on one side for a five-parameter fit.
MINIMUM_SAMPLES = 8


@dataclass(frozen=True)
class EdgeFit:
    """One side's fitted step and the edge it implies."""

    edge: float
    centre: float
    width: float
    inner_level: float
    outer_level: float
    samples: int


def mtanh(x: np.ndarray, slope: float) -> np.ndarray:
    """Return the modified hyperbolic tangent with linear core slope."""
    x = np.clip(x, -40.0, 40.0)
    return ((1.0 + slope * x) * np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def _profile(radius, centre, width, inner, outer, slope, sign):
    x = sign * (centre - radius) / width
    return outer + 0.5 * (inner - outer) * (mtanh(x, slope) + 1.0)


def fit_side(
    radius: np.ndarray,
    value: np.ndarray,
    sigma: np.ndarray,
    outboard: bool,
    level: float = SEPARATRIX_TEMPERATURE,
) -> EdgeFit | None:
    """Fit one side of a profile and return where it falls through ``level``."""
    from scipy.optimize import curve_fit

    keep = np.isfinite(radius) & np.isfinite(value) & np.isfinite(sigma) & (sigma > 0)
    radius, value, sigma = radius[keep], value[keep], sigma[keep]
    if radius.size < MINIMUM_SAMPLES:
        return None
    sign = 1.0 if outboard else -1.0
    order = np.argsort(radius)
    radius, value, sigma = radius[order], value[order], sigma[order]
    peak = float(np.max(value))
    floor = float(max(np.min(value), 0.0))
    # Start the step where the profile first falls below half its peak,
    # walking away from the core.
    walk = range(radius.size) if outboard else range(radius.size - 1, -1, -1)
    half = next(
        (i for i in walk if value[i] < 0.5 * peak and _beyond_peak(i, value, outboard)),
        None,
    )
    centre0 = (
        float(radius[half])
        if half is not None
        else float(radius[-1] if outboard else radius[0])
    )
    span = float(radius[-1] - radius[0])
    try:
        parameters, _ = curve_fit(
            lambda r, c, w, h, b, s: _profile(r, c, w, h, b, s, sign),
            radius,
            value,
            p0=(centre0, 0.03, peak, floor, 0.0),
            sigma=sigma,
            absolute_sigma=True,
            bounds=(
                (radius[0], 0.003, 0.0, 0.0, -1.0),
                (radius[-1], 0.5 * span, 10.0 * peak, peak, 1.0),
            ),
            maxfev=4000,
        )
    except RuntimeError, ValueError:
        return None
    centre, width, inner, outer, slope = (float(item) for item in parameters)
    if not (np.isfinite(width) and width > 0.0):
        return None
    if inner < MINIMUM_CONTRAST * max(outer, 1e-12 * inner):
        return None
    fine = np.linspace(radius[0], radius[-1], 4001)
    fitted = _profile(fine, centre, width, inner, outer, slope, sign)
    if not outboard:
        fine, fitted = fine[::-1], fitted[::-1]
    # Walk away from the core and take the first fall through the level.
    start = int(np.argmax(fitted))
    below = np.flatnonzero(fitted[start:] < level)
    if below.size == 0 or below[0] == 0:
        return None
    index = start + int(below[0])
    upper, lower = fitted[index - 1], fitted[index]
    fraction = (upper - level) / (upper - lower)
    edge = float(fine[index - 1] + fraction * (fine[index] - fine[index - 1]))
    return EdgeFit(edge, centre, width, inner, outer, int(radius.size))


def _beyond_peak(index: int, value: np.ndarray, outboard: bool) -> bool:
    """Return whether ``index`` lies on the far side of the profile maximum."""
    peak = int(np.argmax(value))
    return index > peak if outboard else index < peak


def outboard_edge(
    radius: np.ndarray,
    temperature: np.ndarray,
    sigma: np.ndarray,
    level: float = SEPARATRIX_TEMPERATURE,
) -> EdgeFit | None:
    """Return the outboard edge of one midplane electron-temperature profile.

    Only the outboard side is fitted. MAST's inboard boundary rests on or near
    the centre column at R = 0.195 m, inside the first core channel at about
    0.24 m, so the inboard step is not sampled: a fit there lands on the core
    gradient (0.4 to 0.6 m on 27079) rather than on an edge.

    The profile is split at its smoothed maximum so the outboard side is one
    decreasing step.
    """
    radius = np.asarray(radius, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    finite = np.isfinite(radius) & np.isfinite(temperature)
    if np.count_nonzero(finite) < 2 * MINIMUM_SAMPLES:
        return None
    order = np.argsort(radius[finite])
    r, v = radius[finite][order], temperature[finite][order]
    smooth = np.convolve(v, np.ones(5) / 5.0, mode="same")
    split = float(r[int(np.argmax(smooth))])
    side = radius >= split
    return fit_side(radius[side], temperature[side], sigma[side], True, level)
