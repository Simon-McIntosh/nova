"""Integrate one axis-facing quadratic cell component by exact slice primitives.

The inner integral is analytic for the quadratic profile density. The outer
integral follows the conic's square-root branches with adaptive quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad_vec


@dataclass(frozen=True)
class CellIntegral:
    area: float
    current: float
    radial_moment: float
    vertical_moment: float
    error_estimate: float


def _quadratic(coefficients, x, y):
    a, b, c, d, e, f = coefficients
    return a + b * x + c * y + d * x * x + e * x * y + f * y * y


def _edge_roots(first, last, evaluate):
    halfway = (first + last) / 2
    start = evaluate(*first)
    end = evaluate(*last)
    middle = evaluate(*halfway)
    a = 2 * (start + end - 2 * middle)
    b = end - start - a
    roots = np.roots((a, b, start)) if abs(a) > 1e-15 else np.roots((b, start))
    return [
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-12 and 0 < root.real < 1
    ]


def integrate_cell(
    vertices,
    centre,
    pitch,
    field_coefficients,
    boundary_level,
    sigma,
    saddle,
    axis,
    density_coefficients,
):
    """Integrate the positive Bell region on the axis-facing saddle branch.

    Coordinates are scaled by the cell pitch. The branch is the half-plane
    through the saddle normal to the positive Hessian eigenvector; this is
    exact when the Bell conic has the same stationary point and level there.
    """
    vertices = np.asarray(vertices, dtype=float)
    centre = np.asarray(centre, dtype=float)
    saddle = np.asarray(saddle, dtype=float)
    axis = np.asarray(axis, dtype=float)
    field = np.asarray(field_coefficients, dtype=float).copy()
    field[0] -= boundary_level
    field *= sigma
    density = np.asarray(density_coefficients, dtype=float)
    hessian = np.array([[2 * field[3], field[4]], [field[4], 2 * field[5]]])
    values, vectors = np.linalg.eigh(hessian)
    if not (values[0] < 0 < values[1]):
        raise ValueError("quadratic cell does not have a saddle Hessian")
    positive = vectors[:, 1]
    if np.dot(axis - saddle, positive) < 0:
        positive = -positive
    negative = vectors[:, 0]
    basis = np.column_stack((positive, negative))
    polygon = ((vertices - saddle) / pitch) @ basis

    def field_at(u, v):
        point = saddle + pitch * (u * positive + v * negative)
        local = (point - centre) / pitch
        return _quadratic(field, *local)

    def slice_polynomial(u):
        origin = (saddle - centre) / pitch + u * positive
        x = Polynomial((origin[0], negative[0]))
        y = Polynomial((origin[1], negative[1]))
        q = field[0] + field[1] * x + field[2] * y
        q += field[3] * x * x + field[4] * x * y + field[5] * y * y
        rho = density[0] + density[1] * x + density[2] * y
        rho += density[3] * x * x + density[4] * x * y + density[5] * y * y
        return q, (Polynomial((1,)), rho, pitch * rho * x, pitch * rho * y)

    def slice_integral(u):
        intersections = []
        for first, last in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            span = last[0] - first[0]
            if abs(span) > 1e-14 and min(first[0], last[0]) <= u < max(
                first[0], last[0]
            ):
                intersections.append(
                    first[1] + (u - first[0]) * (last[1] - first[1]) / span
                )
        if len(intersections) < 2:
            return np.zeros(4)
        lower, upper = min(intersections), max(intersections)
        q, polynomials = slice_polynomial(u)
        roots = np.roots(q.coef[::-1])
        cuts = [lower, upper]
        cuts.extend(
            float(root.real)
            for root in roots
            if abs(root.imag) < 1e-12 and lower < root.real < upper
        )
        cuts.sort()
        total = np.zeros(4)
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            if q((left + right) / 2) > 0:
                total += np.array(
                    [p.integ()(right) - p.integ()(left) for p in polynomials]
                )
        return total

    bounds = [max(0.0, float(np.min(polygon[:, 0]))), float(np.max(polygon[:, 0]))]
    if bounds[0] >= bounds[1]:
        return CellIntegral(0.0, 0.0, 0.0, 0.0, 0.0)
    bounds.extend(
        float(point[0]) for point in polygon if bounds[0] < point[0] < bounds[1]
    )
    constant = field_at(0, 0)
    positive_side = field_at(1, 0)
    negative_side = field_at(-1, 0)
    upper_side = field_at(0, 1)
    lower_side = field_at(0, -1)
    diagonal = field_at(1, 1)
    linear_u = (positive_side - negative_side) / 2
    linear_v = (upper_side - lower_side) / 2
    squared_u = (positive_side + negative_side) / 2 - constant
    squared_v = (upper_side + lower_side) / 2 - constant
    mixed = diagonal - positive_side - upper_side + constant
    discriminant = (
        mixed * mixed - 4 * squared_v * squared_u,
        2 * linear_v * mixed - 4 * squared_v * linear_u,
        linear_v * linear_v - 4 * squared_v * constant,
    )
    for root in np.roots(discriminant):
        if abs(root.imag) < 1e-12 and bounds[0] < root.real < bounds[1]:
            bounds.append(float(root.real))
    for first, last in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        for fraction in _edge_roots(first, last, field_at):
            crossing = first + fraction * (last - first)
            if bounds[0] < crossing[0] < bounds[1]:
                bounds.append(float(crossing[0]))
    bounds = sorted(set(bounds))
    total = np.zeros(4)
    error = 0.0
    for left, right in zip(bounds[:-1], bounds[1:], strict=True):
        value, estimated = quad_vec(
            slice_integral, left, right, epsabs=1e-14, epsrel=1e-13
        )
        total += value
        error += estimated
    total *= pitch * pitch
    return CellIntegral(*map(float, total), float(error * pitch * pitch))
