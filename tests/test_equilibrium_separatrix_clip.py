"""Conservative cell supports cut by a piecewise-linear separatrix."""

from __future__ import annotations

import numpy as np
import pytest

from nova.equilibrium.separatrix_clip import AtomicCellMesh


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
