"""Piecewise-linear contour mesh on the hex carrier.

The contour tree is built on a discrete field, and its topology must not depend
on a tolerance. This module supplies that field's carrier: the three-direction
triangulation of the hex cell centres, clipped to the multi-unit vessel polygon,
with wall vertices placed where the triangulation meets the wall polyline.

The field is piecewise-linear. Flux is valued at the hex cell centres; a wall
vertex takes its flux by linear interpolation on the triangle that contains it,
from the SAME piecewise-linear field the tree is built on, so no smooth
representation is needed to value the boundary. The result is a fixed-capacity
ContourMesh of jax arrays, traceable through jit and vmap.

Reuse (contour-tree-topology-authority reuse-map.md rows 1-3):

* row 1 - scipy.spatial.Delaunay over the cell centres is the idiom of
  nova.biot.plasmagrid.PlasmaGrid.tessellate, which recovers the three-direction
  (six-neighbour ring) mesh from an unstructured centre set the same way.
* row 2 - nova.equilibrium.wall_mask.inside_polygon is the host-side vectorised
  (ray-cast) containment used for the occupiable region.
* row 3 - nova.equilibrium.wall_mask.WallUnit carries the multi-unit wall as
  data; the containing-triangle flux lookup it records as absent is
  _barycentric_flux below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay
import shapely
from shapely.geometry import MultiPolygon, Polygon

from nova.equilibrium.wall_mask import WallUnit, inside_polygon

__all__ = ["ContourMesh", "build_contour_mesh"]

DEFAULT_VERTEX_CAPACITY = 4096
DEFAULT_EDGE_CAPACITY = 16384
DEFAULT_TRIANGLE_CAPACITY = 8192


def _unit_polygon(unit: WallUnit) -> Polygon | None:
    """Return a unit's closed polygon, or None for an open line primitive."""
    if not unit.closed or unit.r.size < 3:
        return None
    polygon = Polygon(np.c_[unit.r, unit.z])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _occupiable_geometry(wall_units: Sequence[WallUnit]):
    """Return vessel interiors minus closed material units as one geometry."""
    vessel = None
    for unit in wall_units:
        if unit.kind != "vessel":
            continue
        polygon = _unit_polygon(unit)
        if polygon is None:
            continue
        vessel = polygon if vessel is None else vessel.union(polygon)
    if vessel is None:
        raise ValueError("no closed vessel unit supplied")
    material = None
    for unit in wall_units:
        if unit.kind == "vessel":
            continue
        polygon = _unit_polygon(unit)
        if polygon is None:
            continue
        material = polygon if material is None else material.union(polygon)
    if material is not None:
        vessel = vessel.difference(material)
    return vessel


def _covers(geometry, point_rz: np.ndarray) -> np.ndarray:
    """Return whether each point lies inside or on a closed unit geometry."""
    return np.asarray(
        shapely.covers(geometry, shapely.points(point_rz[:, 0], point_rz[:, 1]))
    )


def _inside_wall_units(
    point_rz: np.ndarray, wall_units: Sequence[WallUnit]
) -> np.ndarray:
    """Return whether each point lies in the occupiable region.

    The interior ray-cast reuses
    :func:`nova.equilibrium.wall_mask.inside_polygon`; the shapely ``covers``
    test supplies the closed-unit surface and the multi-unit union that the
    ray-cast alone leaves open, so a point exactly on a vessel surface counts as
    inside.
    """
    inside_vessel = np.zeros(len(point_rz), dtype=bool)
    vessel_geometry = None
    for unit in wall_units:
        if unit.kind != "vessel":
            continue
        inside_vessel |= inside_polygon(point_rz[:, 0], point_rz[:, 1], unit.r, unit.z)
        polygon = _unit_polygon(unit)
        if polygon is not None:
            vessel_geometry = (
                polygon if vessel_geometry is None else vessel_geometry.union(polygon)
            )
    if vessel_geometry is not None:
        inside_vessel |= _covers(vessel_geometry, point_rz)
    material = np.zeros(len(point_rz), dtype=bool)
    for unit in wall_units:
        if unit.kind == "vessel" or not unit.closed:
            continue
        material |= inside_polygon(point_rz[:, 0], point_rz[:, 1], unit.r, unit.z)
        polygon = _unit_polygon(unit)
        if polygon is not None:
            material |= _covers(polygon, point_rz)
    return inside_vessel & ~material


def _barycentric_flux(point_rz, triangle_rz, triangle_psi) -> float:
    """Return the linear field at point_rz on its containing triangle."""
    a, b, c = triangle_rz
    matrix = np.stack([b - a, c - a], axis=1)
    weights = np.linalg.solve(matrix, np.asarray(point_rz) - a)
    return float(
        triangle_psi[0]
        + weights[0] * (triangle_psi[1] - triangle_psi[0])
        + weights[1] * (triangle_psi[2] - triangle_psi[0])
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ContourMesh:
    """Fixed-capacity piecewise-linear carrier for the contour tree."""

    vertex_rz: jax.Array
    vertex_psi: jax.Array
    vertex_valid: jax.Array
    vertex_is_wall: jax.Array
    edges: jax.Array
    edge_valid: jax.Array
    triangles: jax.Array
    triangle_valid: jax.Array
    vertex_capacity: int = 0
    edge_capacity: int = 0
    triangle_capacity: int = 0
    overflow: bool = False

    def tree_flatten(self):
        children = (
            self.vertex_rz,
            self.vertex_psi,
            self.vertex_valid,
            self.vertex_is_wall,
            self.edges,
            self.edge_valid,
            self.triangles,
            self.triangle_valid,
        )
        return children, (
            self.vertex_capacity,
            self.edge_capacity,
            self.triangle_capacity,
            self.overflow,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        vertex_capacity, edge_capacity, triangle_capacity, overflow = aux
        return cls(
            *children,
            vertex_capacity=vertex_capacity,
            edge_capacity=edge_capacity,
            triangle_capacity=triangle_capacity,
            overflow=overflow,
        )


def _pad(rows, capacity: int, fill):
    """Return rows padded (or clipped) to a fixed first dimension."""
    array = np.asarray(rows)
    if array.shape[0] >= capacity:
        return array[:capacity]
    pad = np.full((capacity - array.shape[0],) + array.shape[1:], fill, array.dtype)
    return np.concatenate([array, pad], axis=0)


def build_contour_mesh(
    centre_rz,
    flux,
    wall_units: Sequence[WallUnit],
    *,
    vertex_capacity: int = DEFAULT_VERTEX_CAPACITY,
    edge_capacity: int = DEFAULT_EDGE_CAPACITY,
    triangle_capacity: int = DEFAULT_TRIANGLE_CAPACITY,
) -> ContourMesh:
    """Build the clipped hex-centre triangulation carrying the contour tree."""
    centre_rz = np.ascontiguousarray(centre_rz, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    if centre_rz.ndim != 2 or centre_rz.shape[1] != 2:
        raise ValueError("cell centres must have shape (cells, 2)")
    if flux.shape != (centre_rz.shape[0],):
        raise ValueError("one flux value is needed per cell centre")
    if any(unit.kind == "vessel" for unit in wall_units) is False:
        raise ValueError("no vessel unit supplied")

    scale = float(np.max(np.abs(centre_rz))) if centre_rz.size else 1.0
    tolerance = max(1e-12, 1e-9 * max(1.0, scale))
    centre_valid = _inside_wall_units(centre_rz, wall_units)

    vertices: list[tuple] = [(float(p[0]), float(p[1])) for p in centre_rz]
    vertex_psi: list[float] = [float(v) for v in flux]
    vertex_valid: list[bool] = [bool(v) for v in centre_valid]
    vertex_is_wall: list[bool] = [False] * len(centre_rz)
    dedup: dict[tuple, int] = {}
    for index, point in enumerate(vertices):
        dedup.setdefault(_key(point, tolerance), index)

    def vertex_index(point, psi_value) -> int:
        key = _key(point, tolerance)
        if key in dedup:
            return dedup[key]
        index = len(vertices)
        vertices.append((float(point[0]), float(point[1])))
        vertex_psi.append(float(psi_value))
        vertex_valid.append(True)
        vertex_is_wall.append(True)
        dedup[key] = index
        return index

    domain = _occupiable_geometry(wall_units)
    simplices = (
        Delaunay(centre_rz).simplices
        if centre_rz.size
        else np.empty((0, 3), dtype=np.int64)
    )

    triangles: list[tuple[int, int, int]] = []
    for simplex in simplices:
        corner = centre_rz[simplex]
        triangle = Polygon(corner)
        if not triangle.is_valid:
            triangle = triangle.buffer(0)
        if triangle.is_empty or not domain.intersects(triangle):
            continue
        clipped = triangle.intersection(domain)
        if clipped.is_empty:
            continue
        parts = list(clipped.geoms) if isinstance(clipped, MultiPolygon) else [clipped]
        for part in parts:
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            corners = np.asarray(part.exterior.coords)[:-1]
            if len(corners) < 3:
                continue
            indices = [
                vertex_index(point, _barycentric_flux(point, corner, flux[simplex]))
                for point in corners
            ]
            for k in range(1, len(indices) - 1):
                triangles.append((indices[0], indices[k], indices[k + 1]))

    edge_order: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key not in edge_order:
                edge_order[key] = len(edges)
                edges.append(key)

    overflow = (
        len(vertices) > vertex_capacity
        or len(edges) > edge_capacity
        or len(triangles) > triangle_capacity
    )

    vertex_rz_arr = _pad(vertices, vertex_capacity, (0.0, 0.0))
    vertex_psi_arr = _pad(vertex_psi, vertex_capacity, 0.0)
    vertex_valid_arr = _pad(vertex_valid, vertex_capacity, False)
    vertex_is_wall_arr = _pad(vertex_is_wall, vertex_capacity, False)
    edges_arr = _pad(edges, edge_capacity, (-1, -1)).astype(np.int32)
    triangle_arr = _pad(triangles, triangle_capacity, (-1, -1, -1)).astype(np.int32)
    edge_valid_arr = np.zeros(edge_capacity, dtype=bool)
    edge_valid_arr[: min(len(edges), edge_capacity)] = True
    triangle_valid_arr = np.zeros(triangle_capacity, dtype=bool)
    triangle_valid_arr[: min(len(triangles), triangle_capacity)] = True

    return ContourMesh(
        vertex_rz=jnp.asarray(vertex_rz_arr),
        vertex_psi=jnp.asarray(vertex_psi_arr),
        vertex_valid=jnp.asarray(vertex_valid_arr),
        vertex_is_wall=jnp.asarray(vertex_is_wall_arr),
        edges=jnp.asarray(edges_arr),
        edge_valid=jnp.asarray(edge_valid_arr),
        triangles=jnp.asarray(triangle_arr),
        triangle_valid=jnp.asarray(triangle_valid_arr),
        vertex_capacity=vertex_capacity,
        edge_capacity=edge_capacity,
        triangle_capacity=triangle_capacity,
        overflow=overflow,
    )


def _key(point, tolerance: float) -> tuple[int, int]:
    """Return a tolerance-quantised key for coordinate de-duplication."""
    return (
        int(round(float(point[0]) / tolerance)),
        int(round(float(point[1]) / tolerance)),
    )
