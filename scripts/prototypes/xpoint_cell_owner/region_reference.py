"""Dense reference for the owner-slice current and first moments.

The owner region is the same axis-facing half-plane the traced integral cuts:
the side of the saddle facing the axis, bounded by the cell polygon and the
zero level of the region-selecting flux. This module integrates that region
with a far higher order rule than the traced one, so a comparison against it
separates the quadrature order from the flux representation used to weight the
region.

Two density representations are offered and they share one geometry:

``fitted``
    the quadratic fit of the composed density in local pitch coordinates, so
    the flux enters only through its own quadratic fit, and
``profile``
    the profile evaluated at every quadrature node from the region-selecting
    flux, so region and weighting share one representation.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

_OUTER_NODES, _OUTER_WEIGHTS = leggauss(256)
_INNER_NODES, _INNER_WEIGHTS = leggauss(64)


def _quadratic(coefficients, x, y):
    a, b, c, d, e, f = coefficients[:6]
    return a + b * x + c * y + d * x * x + e * x * y + f * y * y


def region_moments(
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
    span,
    profile=None,
    density_mode="fitted",
):
    """Integrate area, current and the two first moments over the half-plane.

    Returns ``(moments, valid)`` with ``moments`` ordered area, current, radial
    moment, vertical moment — the order the traced integral emits.
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
        return np.zeros(4), False
    origin = saddle
    plus = vectors[:, 1]
    if np.dot(axis - origin, plus) < 0:
        plus = -plus
    minus = vectors[:, 0]
    basis = np.column_stack((plus, minus))
    polygon = (vertices - origin) / pitch @ basis
    following = np.roll(polygon, -1, axis=0)
    offset = (origin - centre) / pitch

    def field_at(u, v):
        local = offset + u * plus + v * minus
        return _quadratic(field, local[..., 0], local[..., 1])

    def density_at(u, v):
        local = offset + u * plus + v * minus
        x, y = local[..., 0], local[..., 1]
        if density_mode == "profile":
            normalized = 1.0 + _quadratic(field, x, y) / (sigma * span)
            return np.asarray(
                profile.current_density(centre[0] + pitch * x, normalized)
            )
        return _quadratic(density, x, y)

    def slice_integral(u):
        spans = following[:, 0] - polygon[:, 0]
        keep = (
            (np.abs(spans) > 1e-14)
            & (np.minimum(polygon[:, 0], following[:, 0]) <= u)
            & (u < np.maximum(polygon[:, 0], following[:, 0]))
        )
        if np.count_nonzero(keep) < 2:
            return np.zeros(4)
        crossings = (
            polygon[keep, 1]
            + (u - polygon[keep, 0])
            * (following[keep, 1] - polygon[keep, 1])
            / spans[keep]
        )
        low, high = float(np.min(crossings)), float(np.max(crossings))
        roots = np.roots(
            (
                field[5],
                field[2] + field[4] * u,
                field[0] + field[1] * u + field[3] * u * u,
            )
        )
        cuts = [low, high]
        cuts.extend(
            float(root.real)
            for root in roots
            if abs(root.imag) < 1e-12 and low < root.real < high
        )
        cuts.sort()
        total = np.zeros(4)
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            middle = (left + right) / 2
            if field_at(np.float64(u), np.float64(middle)) <= 0:
                continue
            half = (right - left) / 2
            nodes = middle + half * _INNER_NODES
            us = np.full_like(nodes, u)
            rho = density_at(us, nodes)
            local_x = (offset + u * plus)[0] + nodes * minus[0]
            local_y = offset[1] + u * plus[1] + nodes * minus[1]
            terms = np.stack(
                (np.ones_like(rho), rho, pitch * rho * local_x, pitch * rho * local_y),
                axis=-1,
            )
            total += np.einsum("n,ni->i", half * _INNER_WEIGHTS, terms)
        return total

    lower = max(0.0, float(np.min(polygon[:, 0])))
    upper = float(np.max(polygon[:, 0]))
    if lower >= upper:
        return np.zeros(4), False
    theta = (_OUTER_NODES + 1) * (np.pi / 4)
    locations = lower + (upper - lower) * np.sin(theta) ** 2
    jacobian = (upper - lower) * np.sin(2 * theta) * (np.pi / 4)
    stacked = np.stack([slice_integral(u) for u in locations])
    total = np.einsum("n,ni->i", _OUTER_WEIGHTS * jacobian, stacked) * pitch * pitch
    return total, bool(np.all(np.isfinite(total)))
