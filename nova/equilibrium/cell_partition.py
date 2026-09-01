"""Partition geometry carried by cell-centred equilibrium meshes."""

from collections.abc import Iterable

import numpy as np

from nova.biot.plasmagrid import hex_ring_slots
from nova.equilibrium.connectivity_boundary import _raster_hex_partition_geometry

__all__ = ["cell_partition_geometry", "missing_link_mask"]


def _tensor_product_axes(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return exact tensor-product axes, or ``None`` for a cell tiling."""
    radius = np.unique(centroids[:, 0])
    height = np.unique(centroids[:, 1])
    expected = np.c_[
        np.repeat(radius, height.size),
        np.tile(height, radius.size),
    ]
    if centroids.shape == expected.shape and np.array_equal(centroids, expected):
        return radius, height
    return None


def _polygon_vertices(cell) -> np.ndarray:
    """Return one open, finite, two-dimensional polygon exterior."""
    geometry = cell.poly if hasattr(cell, "poly") else cell
    if hasattr(geometry, "exterior"):
        vertices = np.asarray(geometry.exterior.coords, dtype=np.float64)[:, :2]
    else:
        vertices = np.asarray(geometry, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] < 2:
            raise ValueError("cell polygons must have shape (vertices, coordinates)")
        vertices = vertices[:, :2]
    if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if vertices.shape[0] < 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("cell polygons must have at least three finite vertices")
    return np.ascontiguousarray(vertices)


def _shared_polygon_edge(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the longest collinear boundary overlap of two cells."""
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        float(np.ptp(first)),
        float(np.ptp(second)),
        1.0,
    )
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    overlaps: list[tuple[float, np.ndarray, np.ndarray]] = []

    def cross(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(
            first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
        )

    for start, end in zip(first, np.roll(first, -1, axis=0), strict=True):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= tolerance:
            continue
        unit = delta / length
        for other_start, other_end in zip(
            second, np.roll(second, -1, axis=0), strict=True
        ):
            if (
                abs(cross(delta, other_start - start)) / length > tolerance
                or abs(cross(delta, other_end - start)) / length > tolerance
            ):
                continue
            other_projection = np.asarray(
                [np.dot(other_start - start, unit), np.dot(other_end - start, unit)]
            )
            lower = max(0.0, float(np.min(other_projection)))
            upper = min(length, float(np.max(other_projection)))
            if upper - lower > tolerance:
                overlaps.append(
                    (upper - lower, start + lower * unit, start + upper * unit)
                )
    if not overlaps:
        raise ValueError("neighbouring cells do not share a physical polygon edge")
    _length, start, end = max(overlaps, key=lambda item: item[0])
    if tuple(end) < tuple(start):
        start, end = end, start
    return np.stack((start, end))


def missing_link_mask(rings) -> np.ndarray:
    """Mark self-padded neighbour slots that an admissibility read must close."""
    indices = np.asarray(rings)
    if indices.ndim != 2 or indices.shape[1] != 7:
        raise ValueError("rings must have shape (rings, 7), centre first")
    missing = np.zeros(indices.shape, dtype=bool)
    missing[:, 1:] = indices[:, 1:] == indices[:, :1]
    return missing


def _normalise_rings(points: np.ndarray, stencil) -> np.ndarray:
    """Return centre-first rings with neighbours in ``HEX_RING`` angular slots."""
    authored = np.ascontiguousarray(stencil, dtype=np.intp)
    if authored.ndim != 2 or authored.shape[1] != 7:
        raise ValueError("stencil must have shape (rings, 7), centre first")
    if authored.size and (
        np.any(authored[:, 0] < 0) or np.any(authored[:, 0] >= len(points))
    ):
        raise ValueError("ring centres must index the centroid carrier")
    if np.any(authored[:, 1:] >= len(points)):
        raise ValueError("ring neighbours must index the centroid carrier")

    rings = np.repeat(authored[:, :1], 7, axis=1)
    for row, centre in enumerate(authored[:, 0]):
        neighbours = authored[row, 1:]
        present = (neighbours >= 0) & (neighbours != centre)
        present_neighbours = neighbours[present]
        slots = hex_ring_slots(points[centre], points[present_neighbours])
        rings[row, slots + 1] = present_neighbours
    return rings


def cell_partition_geometry(
    centroids,
    stencil,
    cell_polygons: Iterable,
):
    """Return centre-first rings and reciprocal physical shared edges.

    Exact tensor-product centroid lattices retain the raster adapter. Other
    carriers use their authored six-neighbour rings and polygon boundaries.
    Missing rim neighbours occupy self-padded slots identified by
    :func:`missing_link_mask`; admissibility consumers must close those links.
    """
    points = np.ascontiguousarray(centroids, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("centroids must have shape (cells, 2) and be finite")
    axes = _tensor_product_axes(points)
    if axes is not None:
        return _raster_hex_partition_geometry(*axes)

    rings = _normalise_rings(points, stencil)
    missing = missing_link_mask(rings)
    polygons = tuple(_polygon_vertices(cell) for cell in cell_polygons)
    if len(polygons) != len(points):
        raise ValueError("one polygon is required per cell centroid")

    edges = np.empty((*rings.shape, 2, 2), dtype=np.float64)
    edges[:, 0] = points[rings[:, 0], np.newaxis, :]
    shared: dict[tuple[int, int], np.ndarray] = {}
    for row, centre in enumerate(rings[:, 0]):
        for slot, neighbour in enumerate(rings[row, 1:], start=1):
            if missing[row, slot]:
                edges[row, slot] = points[centre, np.newaxis, :]
                continue
            key = tuple(sorted((int(centre), int(neighbour))))
            if key not in shared:
                shared[key] = _shared_polygon_edge(polygons[key[0]], polygons[key[1]])
            edges[row, slot] = shared[key]
    return rings, edges
