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

This module supplies the target-side rule that closes that gap: positive nodes and
weights over polygonal material, including concave outlines, holes and disconnected
pieces, so a caller can average a kernel over a section instead of sampling it at
one point.

What the current is assumed to do
---------------------------------
An UNWEIGHTED area mean: the current sits at constant density over the whole of the
polygon, with no jacket, no insulation, no cooling channel, no void and no turn
structure anywhere inside it. Every figure quoted here and in
:mod:`nova.biot.circle` is that uniform-current limit, and it is the LOWEST self
inductance a given outline can carry -- concentrating the same current into discrete
sub-conductors inside the outline shrinks each one's own geometric mean distance and
raises the result. A real winding pack therefore sits above what this rule returns,
by an amount this module neither models nor bounds, and no convergence figure below
bears on that gap. The weights are where the assumption lives: a caller with a known
current distribution would scale them by it, and what came back would stop being an
area mean.

The rule
--------
A constrained triangulation of the material, with a collapsed tensor-product
Gauss-Legendre rule on each positive-area triangle. Shapely 2.1 supplies the
constrained decomposition directly; Shapely 2.0 uses its Delaunay triangles,
retaining only triangles covered by the polygon and validating that they tile the
same area. Every quadrature weight is therefore positive. Weights sum to the
material area; a caller wanting a mean divides by that sum rather than by an
independently computed area, so the mean of a constant is that constant to
round-off whatever the rule.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate

from nova.biot.polygonanalytic import polygon_analytic_greens

ORDER = 5
"""Gauss-Legendre nodes per direction per positive triangle.

Twenty-five nodes a triangle: 50 for a rectangular coil filament and 100 for a
convex hexagonal plasma cell. Set by the coincident term, the hardest case the
rule sees -- the
integrand there is the section's own flux, smooth on the open section but with only
bounded second derivatives across it, so convergence is algebraic rather than
spectral and the order has to be measured rather than assumed. Measured against a
doubled rule: representative compact, clipped, and ten-to-one sections hold the
self-flux and field to below two parts in ten thousand. A caller with a more slender
section should raise it. See :mod:`tests.test_biotsectionaverage`.
"""


@lru_cache(maxsize=None)
def _legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights mapped onto the unit interval."""
    node, weight = np.polynomial.legendre.leggauss(order)
    return 0.5 * (node + 1.0), 0.5 * weight


def _polygon_parts(geometry):
    """Yield polygon members from a polygonal Shapely result."""
    if isinstance(geometry, Polygon):
        yield geometry
        return
    for member in getattr(geometry, "geoms", ()):
        yield from _polygon_parts(member)


def _local_geometry(section) -> tuple[Polygon | MultiPolygon, np.ndarray]:
    """Return valid polygonal material translated near the origin and its offset."""
    if isinstance(section, Polygon | MultiPolygon):
        if section.is_empty:
            raise ValueError("section geometry must not be empty")
        members = list(_polygon_parts(section))
        if not members:
            raise ValueError("section geometry must contain polygonal material")
        origin = np.asarray(members[0].exterior.coords[0], dtype=np.float64)[:2]
        geometry = affinity.translate(section, xoff=-origin[0], yoff=-origin[1])
    else:
        corner = np.asarray(section, dtype=np.float64)
        if corner.ndim != 2 or corner.shape[1] != 2 or len(corner) < 3:
            raise ValueError(
                f"section vertices must have shape (N, 2), got {corner.shape}"
            )
        if not np.all(np.isfinite(corner)):
            raise ValueError("section vertices must be finite")
        origin = corner[0].copy()
        geometry = Polygon(corner - origin)
    if geometry.is_empty:
        raise ValueError("section geometry must not be empty")
    if not np.isfinite(geometry.area) or geometry.area <= 0.0:
        raise ValueError("section geometry must have positive finite area")
    if not geometry.is_valid:
        raise ValueError("section geometry must be valid")
    return geometry, origin


def _covered_triangles(polygon: Polygon) -> list[Polygon]:
    """Return positive triangles tiling one polygon, including its interior holes."""
    low_r, low_z, high_r, high_z = polygon.bounds
    scale = max(high_r - low_r, high_z - low_z)
    area_floor = 64.0 * np.finfo(np.float64).eps * scale * scale
    constrained = getattr(shapely, "constrained_delaunay_triangles", None)
    if constrained is None:
        candidates = triangulate(polygon)
    else:
        candidates = list(_polygon_parts(constrained(polygon)))
    kept = [
        triangle
        for triangle in candidates
        if triangle.area > area_floor and polygon.covers(triangle)
    ]
    tolerance = max(1.0e-12 * polygon.area, area_floor * max(len(kept), 1))
    if abs(sum(triangle.area for triangle in kept) - polygon.area) <= tolerance:
        return kept

    # The Shapely 2.0 Delaunay route can straddle an unusually conditioned
    # boundary.  Intersecting its non-overlapping triangles with the material
    # restores a partition, then triangulating each polygonal intersection gives
    # the same positive decomposition without relying on a version-only API.
    kept = []
    for candidate in triangulate(polygon):
        for piece in _polygon_parts(candidate.intersection(polygon)):
            for triangle in triangulate(piece):
                if triangle.area > area_floor and piece.covers(triangle):
                    kept.append(triangle)
    if abs(sum(triangle.area for triangle in kept) - polygon.area) > tolerance:
        raise ValueError("section triangulation does not cover its material area")
    return kept


def _triangles(section) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return local triangle vertices, positive areas, origin, and material area."""
    geometry, origin = _local_geometry(section)
    triangles: list[np.ndarray] = []
    areas: list[float] = []
    for polygon in _polygon_parts(geometry):
        for triangle in _covered_triangles(polygon):
            corner = np.asarray(triangle.exterior.coords, dtype=np.float64)[:-1, :2]
            if corner.shape != (3, 2):
                raise ValueError("section decomposition produced a non-triangle")
            triangles.append(corner)
            areas.append(float(triangle.area))
    if not triangles:
        raise ValueError("section geometry contains no positive-area triangles")
    return np.asarray(triangles), np.asarray(areas), origin, float(geometry.area)


def section_triangles(section) -> tuple[np.ndarray, np.ndarray]:
    """Return positive material triangles and their physical areas.

    The triangle vertices use the section's original coordinates. Their areas are
    strictly positive and sum to the actual polygonal material area, including
    holes and disconnected members.
    """
    corner, area, origin, material_area = _triangles(section)
    tolerance = 1.0e-12 * material_area
    if np.any(area <= 0.0) or abs(float(area.sum()) - material_area) > tolerance:
        raise ValueError("section triangles do not reproduce the material area")
    return corner + origin, area


def section_nodes(section, order: int = ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Return positive ``(points, weights)`` over polygonal section material.

    ``section`` may be an ``(n, 2)`` corner array, a Shapely ``Polygon`` whose
    interiors describe holes, or a ``MultiPolygon``. ``points`` is ``(m, 2)`` and
    ``weights`` is ``(m,)`` in square metres, strictly positive and summing to the
    material area.
    """
    if isinstance(order, bool) or not isinstance(order, int | np.integer) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    corner, area, origin, material_area = _triangles(section)
    if order == 1:
        # A one-node triangle rule lives at the centroid.  The collapsed square
        # mapping below needs two radial nodes to reproduce linear functions;
        # using it at order one would move every triangle's first moment towards
        # its first vertex even though its constant-area weight remains exact.
        points = corner.mean(axis=1) + origin
        weights = area.copy()
        weights *= material_area / weights.sum()
        return points, weights
    centre = corner[:, 0]
    towards_start = corner[:, 1] - centre
    towards_end = corner[:, 2] - centre
    line, weight = _legendre(order)
    radial, along = line[None, :, None], line[None, None, :]
    points = (
        centre[:, None, None, :]
        + radial[..., None] * (1.0 - along[..., None]) * towards_start[:, None, None, :]
        + radial[..., None] * along[..., None] * towards_end[:, None, None, :]
    )
    weights = (
        2.0
        * area[:, None, None]
        * radial
        * weight[None, :, None]
        * weight[None, None, :]
    )
    points = points.reshape(-1, 2) + origin
    weights = weights.reshape(-1)
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
        raise ValueError("section quadrature produced non-finite nodes or weights")
    if np.any(weights <= 0.0):
        raise ValueError("section quadrature weights must be strictly positive")
    weights *= material_area / weights.sum()
    return points, weights


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
        total_weight = float(weight.sum())
        if (
            not np.all(np.isfinite(weight))
            or np.any(weight <= 0.0)
            or not np.isfinite(total_weight)
            or total_weight <= 0.0
        ):
            raise ValueError("target-section weights must be positive and finite")
        for row, value in enumerate(evaluated):
            mean[row, index] = weight @ value[start:stop] / total_weight
        start = stop
    return mean[0], mean[1], mean[2]
