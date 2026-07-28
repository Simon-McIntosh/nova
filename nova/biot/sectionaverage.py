"""Area-weighted quadrature over a polygonal section, and what it is for.

A coupling matrix built from a finite-section source kernel evaluated at a single
target POINT is the midpoint rule for the quantity an inductance operator wants.
The quantity itself is a DOUBLE integral -- the source's current spread over its
own section, the result averaged over the target conductor's section, because the
flux a conductor links is the mean of the flux over the area its current occupies
and the force it feels is set by the mean field there. Away from the source that
midpoint rule is excellent: the kernel varies over the target section only through
its own curvature, which for a full ring is set by the major radius. Inside and
next to the source it is not. There the curvature is set by the section size, the
midpoint value and the mean part company at first order in the section, and on the
coincident term they differ by the whole gap between a section's arithmetic mean
logarithmic distance and its geometric mean distance -- about seven percent of the
self flux at a coil-filament or plasma-cell aspect ratio.

This module supplies the target-side rule that closes that gap: nodes and weights
over an arbitrary simple polygon, so a caller can average a kernel over a section
instead of sampling it at one point.

The rule
--------
A signed fan triangulation from the section's own area centroid, with a collapsed
tensor-product Gauss-Legendre rule on each triangle. The signed decomposition is
valid for any simple polygon, concave included, which a wall-clipped plasma cell
can be. The collapse puts the degenerate vertex at the CENTROID rather than at a
polygon corner, so the node clustering it produces lands where the coincident
kernel's curvature is largest. Weights sum to the polygon's area; a caller wanting
a mean divides by that sum rather than by an independently computed area, so the
mean of a constant is that constant to round-off whatever the rule.
"""

from functools import lru_cache

import numpy as np

from nova.biot.greens import section_centroid
from nova.biot.polygonanalytic import polygon_analytic_greens

ORDER = 3
"""Gauss-Legendre nodes per direction per fan triangle.

Nine nodes a triangle: 36 for a rectangular coil filament, 54 for a hexagonal
plasma cell. Set by the coincident term, the hardest case the rule sees -- the
integrand there is the section's own flux, smooth on the open section but with only
bounded second derivatives across it, so convergence is algebraic rather than
spectral and the order has to be measured rather than assumed. Measured against a
doubled rule: a compact section -- a coil filament, a hexagonal cell, a cell clipped
by the wall -- holds the self term to a few parts in ten million, and a section ten
times taller than it is wide to a few parts in ten thousand, since the collapsed fan
gives the long direction the same node count as the short one. A caller with a
genuinely slender section should raise it. See :mod:`tests.test_biotsectionaverage`.
"""


@lru_cache(maxsize=None)
def _legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights mapped onto the unit interval."""
    node, weight = np.polynomial.legendre.leggauss(order)
    return 0.5 * (node + 1.0), 0.5 * weight


def section_nodes(
    vertices: np.ndarray, order: int = ORDER
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, weights)`` integrating a function over the polygon's area.

    ``vertices`` -- ``(n, 2)`` polygon ``(r, z)`` corners, either orientation, no
    repeated closing vertex. ``points`` is ``(m, 2)``, ``weights`` is ``(m,)`` in
    square metres summing to the polygon's signed area, so ``weights @ f /
    weights.sum()`` is the area mean of ``f`` for either orientation.
    """
    corner = np.asarray(vertices, dtype=np.float64)
    centre = section_centroid(corner)
    towards_start = corner - centre
    towards_end = np.roll(corner, -1, axis=0) - centre
    signed = 0.5 * (
        towards_start[:, 0] * towards_end[:, 1]
        - towards_start[:, 1] * towards_end[:, 0]
    )
    line, weight = _legendre(order)
    radial, along = line[None, :, None], line[None, None, :]
    points = (
        centre
        + radial[..., None] * (1.0 - along[..., None]) * towards_start[:, None, None, :]
        + radial[..., None] * along[..., None] * towards_end[:, None, None, :]
    )
    weights = (
        2.0
        * signed[:, None, None]
        * radial
        * weight[None, :, None]
        * weight[None, None, :]
    )
    return points.reshape(-1, 2), weights.reshape(-1)


def averaged_greens(
    target_sections: list[np.ndarray],
    source_vertices: np.ndarray,
    order: int = ORDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, Br, Bz)`` per ampere: the DOUBLE integral, one per target section.

    The source's current spread uniformly over ``source_vertices`` and the result
    averaged over each polygon in ``target_sections``. Returned arrays have one entry
    per target section, in the order given, in the same raw SI units as the
    single-integral kernel they come from: total poloidal flux [Wb/A] and field
    components [T/A].

    Every target section's nodes go to the kernel in ONE call. The closed-form
    polygon kernel holds its corner parts live and amortises them across a call, so
    its cost per evaluation falls by more than an order of magnitude between a
    handful of points and a few thousand; a per-section call would pay that penalty
    once per pair and is what makes a naive double integral look unaffordable.
    """
    node = [section_nodes(vertices, order) for vertices in target_sections]
    if not node:
        empty = np.empty(0)
        return empty, empty.copy(), empty.copy()
    points = np.concatenate([point for point, _ in node])
    evaluated = polygon_analytic_greens(points[:, 0], points[:, 1], source_vertices)
    mean = np.empty((3, len(node)))
    start = 0
    for index, (point, weight) in enumerate(node):
        stop = start + len(weight)
        for row, value in enumerate(evaluated):
            mean[row, index] = weight @ value[start:stop] / weight.sum()
        start = stop
    return mean[0], mean[1], mean[2]
