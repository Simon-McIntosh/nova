"""Conservative cell supports cut by a piecewise-linear separatrix."""

from __future__ import annotations

import numpy as np
import pytest

from nova.equilibrium.separatrix_clip import (
    AtomicCellMesh,
    padded_linear_current_moments,
)


def _staggered_rectangles(
    radial_count: int,
    vertical_count: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return the unsplit cells of a half-offset rectangular tiling."""
    vertical_span = 2.6
    height = vertical_span / (vertical_count - 1)
    width = height / (np.sqrt(3.0) / 2.0)
    radial_index, vertical_index = np.indices((radial_count, vertical_count))
    radius = 3.0 + width * (
        radial_index
        - 0.5 * (radial_count - 1)
        + 0.5 * (vertical_index - 0.5 * (vertical_count - 1))
    )
    vertical = height * (vertical_index - 0.5 * (vertical_count - 1))
    centre = np.column_stack((radius.ravel(), vertical.ravel()))
    half = np.asarray([0.5 * width, 0.5 * height])
    corner = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    cells = [point + corner * half for point in centre]
    return cells, centre


def _ellipse_level(points: np.ndarray) -> np.ndarray:
    radial = (points[:, 0] - 3.0) / 0.78
    vertical = points[:, 1] / 0.57
    return 1.0 - radial**2 - vertical**2


@pytest.mark.parametrize(
    ("radial_count", "vertical_count"),
    [(15, 17), (17, 27), (23, 35), (29, 45)],
)
def test_patch_area_matches_same_crossing_lcfs_on_every_grid(
    radial_count: int,
    vertical_count: int,
):
    cells, centre = _staggered_rectangles(radial_count, vertical_count)
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)

    clipped = mesh.clip(_ellipse_level(mesh.node_coordinates))

    assert clipped.contour_closed
    assert clipped.boundary.sum() > 0
    assert np.any(clipped.included & (_ellipse_level(centre) < 0.0))
    assert clipped.patch_area_sum == pytest.approx(
        clipped.contour_area, rel=0.0, abs=1.0e-12
    )


def test_t_junctions_share_one_atomic_edge_crossing():
    cells, centre = _staggered_rectangles(17, 27)
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)

    clipped = mesh.clip(_ellipse_level(mesh.node_coordinates))

    assert np.max(mesh.cell_vertex_count) == 6
    for crossing in clipped.contour_vertices[: clipped.contour_vertex_count]:
        occurrences = 0
        for vertices, count in zip(
            clipped.support_vertices, clipped.vertex_count, strict=True
        ):
            occurrences += np.count_nonzero(
                np.all(
                    vertices[:count].view(np.uint64) == crossing.view(np.uint64), axis=1
                )
            )
        assert occurrences == 2
    relative_residual = (
        abs(clipped.patch_area_sum - clipped.contour_area) / clipped.contour_area
    )
    assert relative_residual < 1.0e-12


def test_any_intersection_includes_a_centroid_outside_cell():
    cells, centre = _staggered_rectangles(17, 27)
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)

    clipped = mesh.clip(_ellipse_level(mesh.node_coordinates))
    counterexample = np.flatnonzero(clipped.included & (_ellipse_level(centre) < 0.0))

    assert len(counterexample) > 0
    assert np.all(clipped.area[counterexample] > 0.0)
    assert np.all(clipped.boundary[counterexample])


def test_exact_zero_shared_corner_has_one_contour_vertex():
    cells = []
    centroids = []
    for lower_radial in range(-2, 2):
        for lower_vertical in range(-2, 2):
            cells.append(
                np.asarray(
                    [
                        [lower_radial, lower_vertical],
                        [lower_radial + 1, lower_vertical],
                        [lower_radial + 1, lower_vertical + 1],
                        [lower_radial, lower_vertical + 1],
                    ],
                    dtype=float,
                )
            )
            centroids.append([lower_radial + 0.5, lower_vertical + 0.5])
    mesh = AtomicCellMesh.from_cells(cells, centroids=np.asarray(centroids))
    level = (
        1.0 - np.abs(mesh.node_coordinates[:, 0]) - np.abs(mesh.node_coordinates[:, 1])
    )

    clipped = mesh.clip(level)
    contour = clipped.contour_vertices[: clipped.contour_vertex_count]

    assert clipped.contour_closed
    assert clipped.contour_vertex_count == 4
    assert len(np.unique(contour, axis=0)) == 4
    assert clipped.patch_area_sum == pytest.approx(2.0, rel=0.0, abs=1.0e-12)
    assert clipped.contour_area == pytest.approx(2.0, rel=0.0, abs=1.0e-12)


def test_linear_current_moments_use_the_clipped_support_about_fixed_centroid():
    cell = np.asarray([[1.5, -0.5], [2.5, -0.5], [2.5, 0.5], [1.5, 0.5]])
    mesh = AtomicCellMesh.from_cells([cell], centroids=np.asarray([[2.0, 0.0]]))
    clipped = mesh.clip(mesh.node_coordinates[:, 0] - 2.0)

    moments = clipped.linear_current_moments(
        density=np.asarray([3.0]),
        gradient=np.asarray([[5.0, 7.0]]),
    )

    assert clipped.area[0] == pytest.approx(0.5)
    assert clipped.first_area_moment[0] == pytest.approx([0.125, 0.0])
    np.testing.assert_allclose(
        clipped.second_area_moment[0],
        [[1.0 / 24.0, 0.0], [0.0, 1.0 / 24.0]],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert moments.current[0] == pytest.approx(3.0 * 0.5 + 5.0 * 0.125)
    assert moments.first[0] == pytest.approx([3.0 * 0.125 + 5.0 / 24.0, 7.0 / 24.0])


def _padded_moments(mesh: AtomicCellMesh, flux: np.ndarray):
    clipped = mesh.clip(flux)
    density = 2.0 + 0.2 * (mesh.centroids[:, 0] - 3.0) - 0.1 * mesh.centroids[:, 1]
    gradient = np.broadcast_to(np.asarray([0.2, -0.1]), mesh.centroids.shape)
    current, first = padded_linear_current_moments(
        clipped.support_vertices,
        clipped.vertex_count,
        mesh.centroids,
        density,
        gradient,
    )
    return clipped, np.asarray(current), np.asarray(first)


def test_perturbed_flux_moves_every_boundary_current_moment_continuously():
    cells, centre = _staggered_rectangles(17, 27)
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)
    flux = _ellipse_level(mesh.node_coordinates)
    epsilon = 1.0e-5

    base, base_current, base_first = _padded_moments(mesh, flux)
    fine, fine_current, fine_first = _padded_moments(mesh, flux + epsilon)
    coarse, coarse_current, coarse_first = _padded_moments(mesh, flux + 2.0 * epsilon)

    np.testing.assert_array_equal(fine.boundary, base.boundary)
    np.testing.assert_array_equal(coarse.boundary, base.boundary)
    active = base.boundary
    fine_current_delta = np.abs(fine_current - base_current)[active]
    coarse_current_delta = np.abs(coarse_current - base_current)[active]
    fine_first_delta = np.linalg.norm(fine_first - base_first, axis=1)[active]
    coarse_first_delta = np.linalg.norm(coarse_first - base_first, axis=1)[active]
    assert np.all(fine_current_delta > 0.0)
    assert np.all(fine_first_delta > 0.0)
    assert np.linalg.norm(coarse_current_delta) / np.linalg.norm(
        fine_current_delta
    ) == pytest.approx(2.0, rel=2.0e-3)
    assert np.linalg.norm(coarse_first_delta) / np.linalg.norm(
        fine_first_delta
    ) == pytest.approx(2.0, rel=2.0e-3)


def test_centroid_selection_jumps_while_intersection_moments_are_continuous():
    cell = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    mesh = AtomicCellMesh.from_cells([cell], centroids=np.asarray([[0.0, 0.0]]))
    epsilon = 1.0e-6
    density = np.asarray([2.0])
    gradient = np.zeros((1, 2))

    clipped_current = []
    centroid_current = []
    for displacement in (-epsilon, epsilon):
        flux = mesh.node_coordinates[:, 0] - displacement
        clipped = mesh.clip(flux)
        current, _first = padded_linear_current_moments(
            clipped.support_vertices,
            clipped.vertex_count,
            mesh.centroids,
            density,
            gradient,
        )
        clipped_current.append(float(current[0]))
        centroid_inside = -displacement > 0.0
        centroid_current.append(float(centroid_inside * density[0] * 4.0))

    clipped_change = abs(clipped_current[1] - clipped_current[0])
    centroid_jump = abs(centroid_current[1] - centroid_current[0])
    assert clipped_change == pytest.approx(8.0 * epsilon, rel=1.0e-9)
    assert centroid_jump == pytest.approx(8.0)
    assert centroid_jump / clipped_change == pytest.approx(1.0 / epsilon)


def test_moving_separatrix_padded_contraction_traces_once():
    import jax
    import jax.numpy as jnp

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    cells, centre = _staggered_rectangles(17, 27)
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)
    density = jnp.asarray(
        2.0 + 0.2 * (mesh.centroids[:, 0] - 3.0) - 0.1 * mesh.centroids[:, 1]
    )
    gradient = jnp.broadcast_to(jnp.asarray([0.2, -0.1]), mesh.centroids.shape)
    trace_count = 0

    def traced(vertices, count, centroids):
        nonlocal trace_count
        trace_count += 1
        return padded_linear_current_moments(
            vertices, count, centroids, density, gradient
        )

    compiled = jax.jit(traced)
    shapes = set()
    for displacement in np.linspace(-0.025, 0.025, 12):
        points = mesh.node_coordinates.copy()
        points[:, 0] -= displacement
        clipped = mesh.clip(_ellipse_level(points))
        shapes.add((clipped.support_vertices.shape, clipped.vertex_count.shape))
        current, first = compiled(
            jnp.asarray(clipped.support_vertices),
            jnp.asarray(clipped.vertex_count),
            jnp.asarray(mesh.centroids),
        )
        jax.block_until_ready((current, first))
        expected = clipped.linear_current_moments(
            np.asarray(density), np.asarray(gradient)
        )
        np.testing.assert_allclose(
            current, expected.current, rtol=2.0e-13, atol=1.0e-14
        )
        np.testing.assert_allclose(first, expected.first, rtol=2.0e-13, atol=1.0e-14)

    assert shapes == {((459, 12, 2), (459,))}
    assert trace_count == 1
