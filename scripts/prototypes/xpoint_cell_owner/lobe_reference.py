"""Integrate the owner region's ring fan exactly, sharing no rule with the booking.

The booking integrates each fan triangle with a degree-five quadrature. This
module expands the fitted density quadratic in each triangle's barycentric
coordinates and integrates every monomial in closed form, so the two routes share
neither quadrature points nor weights. A quadratic density is degree two, so the
closed form is exact and the two agree to roundoff; any larger difference means
the fan the two consume is not the same region.

The fan is built by ``nova.equilibrium.current._normal_rings`` from the read's
own selection fields, so the check covers the slicing the booking consumes. The
whole-cell reference of the same rule integrates the cell polygon's own centre
fan, which is the fan the read uses for a cell it does not cut.
"""

from math import factorial

import numpy as np

from nova.equilibrium import current


def _simplex_moment(power_u, power_v):
    """Integral of ``u**power_u v**power_v`` over the reference simplex."""
    return (
        factorial(power_u)
        * factorial(power_v)
        / factorial(power_u + power_v + 2)
    )


def _power_polynomial(base_u, base_v, exponent, constant):
    """Expand ``(constant + base_u u + base_v v) ** exponent`` as {(i, j): c}."""
    poly = {(0, 0): 1.0}
    linear = ((0, 0, constant), (1, 0, base_u), (0, 1, base_v))
    for _ in range(exponent):
        product = {}
        for (i, j), coefficient in poly.items():
            for shift_u, shift_v, weight in linear:
                key = (i + shift_u, j + shift_v)
                product[key] = product.get(key, 0.0) + coefficient * weight
        poly = product
    return poly


def _triangle_monomial(vertices, power_x, power_y):
    """Integral of ``x**power_x y**power_y`` over one triangle, signed.

    ``P = A + u (B - A) + v (C - A)`` maps the reference simplex onto the
    triangle with jacobian the signed twice-area, so a quadratic integrates in
    closed form and a negatively oriented member of a signed fan contributes
    negatively, as the booking's own fan requires.
    """
    ax, ay = float(vertices[0][0]), float(vertices[0][1])
    bx, by = float(vertices[1][0]), float(vertices[1][1])
    cx, cy = float(vertices[2][0]), float(vertices[2][1])
    twice_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    x_poly = _power_polynomial(bx - ax, cx - ax, power_x, ax)
    y_poly = _power_polynomial(by - ay, cy - ay, power_y, ay)
    total = 0.0
    for (i, j), first in x_poly.items():
        for (p, q), second in y_poly.items():
            total += first * second * _simplex_moment(i + p, j + q)
    return twice_area * total


def triangle_integral(vertices, coefficients, origin, pitch):
    """Area, current and first moments over one triangle, in metres and amperes.

    The coefficients are the booking's fitted quadratic in the cell's local pitch
    coordinates about the cell centre, so ``x`` and ``y`` below are
    ``(P - origin) / pitch`` while the moments carry the displacement in metres.
    """
    local = tuple(((np.asarray(v) - origin) / pitch) for v in vertices)

    def moment(power_x, power_y):
        return _triangle_monomial(local, power_x, power_y)

    c = [float(value) for value in np.asarray(coefficients).reshape(-1)]
    area = moment(0, 0)
    charge = (
        c[0] * moment(0, 0)
        + c[1] * moment(1, 0)
        + c[2] * moment(0, 1)
        + c[3] * moment(2, 0)
        + c[4] * moment(1, 1)
        + c[5] * moment(0, 2)
    )
    first_x = (
        c[0] * moment(1, 0)
        + c[1] * moment(2, 0)
        + c[2] * moment(1, 1)
        + c[3] * moment(3, 0)
        + c[4] * moment(2, 1)
        + c[5] * moment(1, 2)
    )
    first_y = (
        c[0] * moment(0, 1)
        + c[1] * moment(1, 1)
        + c[2] * moment(0, 2)
        + c[3] * moment(2, 1)
        + c[4] * moment(1, 2)
        + c[5] * moment(0, 3)
    )
    return (
        area * pitch * pitch,
        charge * pitch * pitch,
        first_x * pitch * pitch * pitch,
        first_y * pitch * pitch * pitch,
    )


def _fan_integral(triangles, coefficients, origin, pitch):
    total = np.zeros(4)
    for triangle in triangles:
        total += np.asarray(triangle_integral(triangle, coefficients, origin, pitch))
    return total


def lobe_reference(geometry, reading, index, density, segments=4):
    """Exact fan integral over the lobe the booking cuts for one owner cell."""
    origin = np.asarray(geometry.centre[index], dtype=float)
    pitch = float(geometry.pitch[index])
    rings, sizes = current._normal_rings(
        geometry.vertices[index],
        geometry.vertex_count[index],
        geometry.centre[index],
        reading.saddle_form,
        reading.fragment_selected[index],
        reading.edge_interval[index],
        segments,
    )
    rings = np.asarray(rings, dtype=float)
    sizes = np.asarray(sizes)
    triangles = []
    for row in range(rings.shape[0]):
        count = int(sizes[row])
        for slot in range(count):
            following = slot + 1 if slot + 1 < count else 0
            triangles.append((origin, rings[row, slot], rings[row, following]))
    return _fan_integral(triangles, density[index], origin, pitch), rings, sizes


def polygon_reference(geometry, index, density, segments=1):
    """Exact integral over the cell's own polygon, from a fan about its centre.

    Each edge is subdivided into ``segments`` chords, so a dense reference uses
    the same polygon with a finer fan and the closed form still integrates it
    exactly.
    """
    origin = np.asarray(geometry.centre[index], dtype=float)
    pitch = float(geometry.pitch[index])
    vertices = np.asarray(geometry.vertices[index], dtype=float)
    count = int(np.asarray(geometry.vertex_count[index]))
    triangles = []
    for slot in range(count):
        start = vertices[slot]
        end = vertices[(slot + 1) % count]
        for fraction in range(segments):
            left = start + (end - start) * (fraction / segments)
            right = start + (end - start) * ((fraction + 1) / segments)
            triangles.append((origin, left, right))
    return _fan_integral(triangles, density[index], origin, pitch)