"""Bernstein enclosures for straight paths on a bicubic tensor spline."""

from __future__ import annotations

import numpy as np
from scipy.special import comb


DEGREE = 6
ARITHMETIC_PAD = 1e-12
PARAMETERS = (1 - np.cos(np.linspace(0, np.pi, DEGREE + 1))) / 2
BERNSTEIN = np.column_stack(
    [
        comb(DEGREE, k) * PARAMETERS**k * (1 - PARAMETERS) ** (DEGREE - k)
        for k in range(DEGREE + 1)
    ]
)
INVERSE = np.linalg.inv(BERNSTEIN)


def split_at_knots(points: np.ndarray, rknots: np.ndarray, zknots: np.ndarray):
    """Split each line where the tensor spline changes polynomial pieces."""
    pieces = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        fractions = [0.0, 1.0]
        for coordinate, knots in enumerate((rknots, zknots)):
            if delta[coordinate] == 0:
                continue
            lo, hi = sorted((start[coordinate], end[coordinate]))
            internal = knots[(knots > lo) & (knots < hi)]
            fractions.extend((internal - start[coordinate]) / delta[coordinate])
        fractions = np.unique(np.clip(fractions, 0, 1))
        for a, b in zip(fractions[:-1], fractions[1:]):
            pieces.append((start + a * delta, start + b * delta))
    return np.asarray(pieces, dtype=float)


def bound_polyline(spline, polarity: int, points: np.ndarray, *, chunk=10000):
    """Enclose the minimum along every line, including exact endpoints.

    On one knot rectangle a bicubic restricted to a line has degree at most
    six. Seven values determine its Bernstein coefficients; their convex hull
    encloses every value on that line. A conservative arithmetic pad covers
    double-precision interpolation and the basis solve. The independent
    midpoint residual is recorded as a check on that pad.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("a polyline needs at least two R,Z points")
    rknots, zknots = spline.get_knots()
    pieces = split_at_knots(points, np.unique(rknots), np.unique(zknots))
    lower = np.inf
    upper = np.inf
    maximum_upper = -np.inf
    min_at = None
    residual = 0.0
    count = 0
    for offset in range(0, len(pieces), chunk):
        part = pieces[offset : offset + chunk]
        coordinates = part[:, 0, None, :] + PARAMETERS[None, :, None] * (
            part[:, 1, None, :] - part[:, 0, None, :]
        )
        values = polarity * spline.ev(
            coordinates[:, :, 0].ravel(), coordinates[:, :, 1].ravel()
        ).reshape(-1, DEGREE + 1)
        coefficients = values @ INVERSE.T
        lower = min(lower, float(coefficients.min()) - ARITHMETIC_PAD)
        maximum_upper = max(maximum_upper, float(coefficients.max()) + ARITHMETIC_PAD)
        index = np.unravel_index(np.argmin(values), values.shape)
        candidate = float(values[index])
        if candidate < upper:
            upper = candidate
            min_at = coordinates[index].tolist()
        middle = (part[:, 0, :] + part[:, 1, :]) / 2
        actual = polarity * spline.ev(middle[:, 0], middle[:, 1])
        predicted = coefficients @ np.asarray(
            [comb(DEGREE, k) / 2**DEGREE for k in range(DEGREE + 1)]
        )
        residual = max(residual, float(np.max(np.abs(actual - predicted))))
        count += len(part)
    if residual > ARITHMETIC_PAD / 10:
        raise ArithmeticError(
            f"Bernstein interpolation residual {residual} exceeds pad"
        )
    return {
        "lower": lower,
        "upper": upper,
        "maximum_upper": maximum_upper,
        "min_at": min_at,
        "pieces": count,
        "midpoint_residual": residual,
        "arithmetic_pad": ARITHMETIC_PAD,
    }


def enclosing_target_cut(spline, polarity: int, target, axis, level):
    """Certify a tiny closed sublevel square around a below-level endpoint."""
    target = np.asarray(target, dtype=float)
    axis = np.asarray(axis, dtype=float)
    value = float(polarity * spline.ev(*target))
    margin = level - value
    if margin <= 8 * ARITHMETIC_PAD:
        return None
    gradient = polarity * np.asarray(
        [spline.ev(*target, dx=1, dy=0), spline.ev(*target, dx=0, dy=1)],
        dtype=float,
    ).reshape(2)
    radius = min(margin / (16 * max(np.linalg.norm(gradient), 1e-3)), 1e-8)
    corners = target + radius * np.asarray(
        [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)], dtype=float
    )
    bound = bound_polyline(spline, polarity, corners)
    if not bound["maximum_upper"] < level:
        raise ArithmeticError("closed target cut is not strictly below the level")
    if np.max(np.abs(axis - target)) <= radius:
        raise ArithmeticError("axis lies inside target cut")
    return {"points": corners.tolist(), "radius": radius, "bound": bound}
