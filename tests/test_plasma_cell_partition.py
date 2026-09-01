"""Cell-carried geometry for the saddle-aware partition."""

import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay

from nova.biot.plasmagrid import PlasmaGrid
from nova.equilibrium.cell_partition import (
    cell_partition_geometry,
    missing_link_mask,
)
from nova.equilibrium.connectivity_boundary import _raster_hex_partition_geometry
from nova.geometry.hexstencil import HEX_RING, hex_stencil


def _hex_tiling(shape: tuple[int, int], pitch: float) -> np.ndarray:
    column, row = np.indices(shape)
    radius = 1.7 + pitch * (column + 0.5 * row)
    height = pitch * np.sqrt(3.0) / 2.0 * row
    return np.c_[radius.ravel(), height.ravel()]


def _hexagons(centroids: np.ndarray, pitch: float) -> tuple[np.ndarray, ...]:
    angle = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    offset = pitch / np.sqrt(3.0) * np.c_[np.cos(angle), np.sin(angle)]
    return tuple(centre + offset for centre in centroids)


def _expected_rings(centres: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Derive flat neighbour indices directly from the axial contract."""
    centre_coordinates = np.column_stack(np.unravel_index(centres, shape))
    neighbour_coordinates = centre_coordinates[:, np.newaxis, :] + HEX_RING
    neighbours = np.ravel_multi_index(np.moveaxis(neighbour_coordinates, -1, 0), shape)
    return np.column_stack((centres, neighbours))


def test_complete_hex_lattice_uses_contract_rings_and_physical_edges():
    shape = (7, 8)
    pitch = 0.12
    centroids = _hex_tiling(shape, pitch)
    triangulation = Delaunay(centroids)
    stencil, centre_index = PlasmaGrid.loop_neighbour_vertices(
        centroids,
        triangulation.vertex_neighbor_vertices,
        np.unique(triangulation.convex_hull),
    )
    expected_rings = _expected_rings(centre_index, shape)
    np.testing.assert_array_equal(stencil, expected_rings)

    descending_angle_order = stencil[:, [0, 1, 6, 5, 4, 3, 2]]
    polygons = _hexagons(centroids, pitch)
    rings, shared_edges = cell_partition_geometry(
        centroids, descending_angle_order, polygons
    )

    np.testing.assert_array_equal(rings, expected_rings)
    for row, ring in enumerate(rings):
        np.testing.assert_array_equal(
            shared_edges[row, 0], np.repeat(centroids[ring[0]][None, :], 2, axis=0)
        )
        for slot, neighbour in enumerate(ring[1:], start=1):
            shared_vertices = []
            for vertex in polygons[ring[0]]:
                if np.min(np.linalg.norm(polygons[neighbour] - vertex, axis=1)) < 1e-12:
                    shared_vertices.append(vertex)
            assert len(shared_vertices) == 2
            expected = np.asarray(sorted(shared_vertices, key=tuple))
            np.testing.assert_allclose(
                shared_edges[row, slot], expected, rtol=0.0, atol=1e-12
            )


def test_reciprocal_links_address_the_same_canonical_segment():
    shape = (6, 6)
    pitch = 0.18
    centroids = _hex_tiling(shape, pitch)
    rings, edges = cell_partition_geometry(
        centroids, hex_stencil(shape), _hexagons(centroids, pitch)
    )
    directed = {
        (int(ring[0]), int(neighbour)): edges[row, slot]
        for row, ring in enumerate(rings)
        for slot, neighbour in enumerate(ring[1:], start=1)
    }
    reciprocal = 0
    for (centre, neighbour), edge in directed.items():
        if (neighbour, centre) in directed:
            np.testing.assert_array_equal(edge, directed[neighbour, centre])
            reciprocal += 1
    assert reciprocal > 0


def test_missing_neighbour_is_explicitly_masked():
    pitch = 0.2
    centroids = _hex_tiling((3, 3), pitch)
    stencil = hex_stencil((3, 3))
    stencil[0, 3] = -1
    rings, edges = cell_partition_geometry(
        centroids, stencil, _hexagons(centroids, pitch)
    )
    expected_missing = np.zeros(rings.shape, dtype=bool)
    expected_missing[0, 3] = True
    np.testing.assert_array_equal(missing_link_mask(rings), expected_missing)
    assert rings[0, 3] == rings[0, 0]
    np.testing.assert_array_equal(
        edges[0, 3], np.repeat(centroids[4][None, :], 2, axis=0)
    )


def test_tensor_product_lattice_routes_to_the_raster_adapter_exactly():
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-0.5, 0.5, 9)
    radius_grid, height_grid = np.meshgrid(radius, height, indexing="ij")
    centroids = np.c_[radius_grid.ravel(), height_grid.ravel()]
    polygons = tuple(np.zeros((3, 2)) for _ in centroids)

    actual = cell_partition_geometry(centroids, np.empty((0, 7)), polygons)
    expected = _raster_hex_partition_geometry(jnp.asarray(radius), jnp.asarray(height))

    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))
