"""Integrate one axis-facing quadratic cell component by exact slice primitives.

The inner integral is analytic for the quadratic profile density. The outer
integral follows the conic's square-root branches with adaptive quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
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
    *,
    outer_rule="adaptive",
    branch_shift=0.0,
    axis_component="halfplane",
):
    """Integrate the positive Bell region on its selected axis component.

    Coordinates are scaled by the cell pitch. The half-plane mode selects
    the side of the physical saddle facing the axis. The Bell mode uses the
    quadratic's own stationary point and includes both sides when its
    positive region is connected through that point.
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
    if axis_component == "bell":
        stationary_local = -np.linalg.solve(hessian, field[1:3])
        origin = centre + pitch * stationary_local
        critical_level = _quadratic(field, *stationary_local)
        restrict_branch = critical_level <= 0
    elif axis_component == "halfplane":
        origin = saddle
        restrict_branch = True
    else:
        raise ValueError(f"unknown component rule: {axis_component}")
    positive = vectors[:, 1]
    if np.dot(axis - origin, positive) < 0:
        positive = -positive
    negative = vectors[:, 0]
    basis = np.column_stack((positive, negative))
    polygon = ((vertices - origin) / pitch) @ basis

    def field_at(u, v):
        point = origin + pitch * (u * positive + v * negative)
        local = (point - centre) / pitch
        return _quadratic(field, *local)

    def slice_polynomial(u):
        local_origin = (origin - centre) / pitch + u * positive
        x = Polynomial((local_origin[0], negative[0]))
        y = Polynomial((local_origin[1], negative[1]))
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
        roots = [root + branch_shift for root in roots]
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

    lower_bound = float(np.min(polygon[:, 0]))
    if restrict_branch:
        lower_bound = max(0.0, lower_bound)
    bounds = [lower_bound, float(np.max(polygon[:, 0]))]
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
        if outer_rule == "adaptive":
            value, estimated = quad_vec(
                slice_integral, left, right, epsabs=1e-14, epsrel=1e-13
            )
        elif outer_rule == "dense":
            nodes, weights = leggauss(256)
            theta = (nodes + 1) * (np.pi / 4)
            locations = left + (right - left) * np.sin(theta) ** 2
            jacobian = (right - left) * np.sin(2 * theta) * (np.pi / 4)
            value = np.einsum(
                "n,ni->i",
                weights * jacobian,
                np.stack([slice_integral(u) for u in locations]),
            )
            estimated = np.nan
        else:
            raise ValueError(f"unknown outer quadrature rule: {outer_rule}")
        total += value
        error += estimated
    total *= pitch * pitch
    return CellIntegral(*map(float, total), float(error * pitch * pitch))
