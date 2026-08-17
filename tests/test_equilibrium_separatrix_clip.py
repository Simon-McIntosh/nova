"""Conservative cell supports cut by a piecewise-linear separatrix."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from nova.equilibrium.separatrix_clip import (
    AtomicCellMesh,
    padded_linear_current_moments,
    padded_polynomial_current_moments,
)


def test_polynomial_current_moments_match_adaptive_triangle_quadrature():
    """Cubic density and its weighted moments are exact on a clipped polygon."""
    centre = np.asarray([1.7, -0.25])
    scale = np.asarray([0.31, 0.23])
    polygon = centre + scale * np.asarray(
        [[-0.8, -0.7], [0.9, -0.55], [0.65, 0.8], [-0.35, 0.95], [-0.9, 0.1]]
    )
    coefficients = np.asarray(
        [2.1, -0.7, 0.4, 0.31, -0.22, 0.18, 0.09, -0.06, 0.04, -0.03]
    )
    vertices = np.zeros((1, 8, 2))
    vertices[0, : len(polygon)] = polygon
    actual = padded_polynomial_current_moments(
        vertices,
        np.asarray([len(polygon)]),
        centre[None, :],
        scale[None, :],
        coefficients[None, :],
    )

    powers = (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
    )

    def density(radius, height):
        u = (radius - centre[0]) / scale[0]
        v = (height - centre[1]) / scale[1]
        return sum(
            coefficient * u**radial * v**vertical
            for coefficient, (radial, vertical) in zip(
                coefficients, powers, strict=True
            )
        )

    expected = np.zeros(3)
    anchor = polygon[0]
    for first, second in zip(polygon[1:-1], polygon[2:], strict=True):
        edge = np.column_stack((first - anchor, second - anchor))
        determinant = abs(float(np.linalg.det(edge)))
        for component, weight in enumerate(
            (
                lambda radius, height: 1.0,
                lambda radius, height: radius - centre[0],
                lambda radius, height: height - centre[1],
            )
        ):
            value, _error = integrate.dblquad(
                lambda vertical_coordinate, radial_coordinate: (
                    density(
                        anchor[0]
                        + radial_coordinate * edge[0, 0]
                        + vertical_coordinate * edge[0, 1],
                        anchor[1]
                        + radial_coordinate * edge[1, 0]
                        + vertical_coordinate * edge[1, 1],
                    )
                    * weight(
                        anchor[0]
                        + radial_coordinate * edge[0, 0]
                        + vertical_coordinate * edge[0, 1],
                        anchor[1]
                        + radial_coordinate * edge[1, 0]
                        + vertical_coordinate * edge[1, 1],
                    )
                    * determinant
                ),
                0.0,
                1.0,
                0.0,
                lambda radial_coordinate: 1.0 - radial_coordinate,
                epsabs=2.0e-13,
                epsrel=2.0e-13,
            )
            expected[component] += value

    np.testing.assert_allclose(
        np.r_[np.asarray(actual[0]), np.asarray(actual[1]).ravel()],
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
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


@pytest.mark.parametrize(
    ("radial_count", "vertical_count"),
    [(15, 17), (17, 27), (23, 35)],
)
def test_traced_clip_matches_host_supports_and_linear_moments(
    radial_count: int, vertical_count: int
):
    cells, centre = _staggered_rectangles(radial_count, vertical_count)
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)
    flux = _ellipse_level(mesh.node_coordinates)
    density = 2.0 + 0.2 * (centre[:, 0] - 3.0) - 0.1 * centre[:, 1]
    gradient = np.broadcast_to(np.asarray([0.2, -0.1]), centre.shape)

    host = mesh.clip(flux)
    traced = mesh.traced_clip(flux)
    host_moments = host.linear_current_moments(density, gradient)
    traced_current, traced_first = traced.linear_current_moments(density, gradient)

    np.testing.assert_array_equal(traced.vertex_count, host.vertex_count)
    np.testing.assert_allclose(
        traced.support_vertices, host.support_vertices, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(traced.area, host.area, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        traced_current, host_moments.current, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(
        traced_first, host_moments.first, rtol=1.0e-14, atol=1.0e-14
    )
    assert abs(float(traced.patch_area_sum - traced.contour_area)) < 1.0e-12


def test_traced_clip_matches_exact_zero_corner_and_tangential_cells():
    cells = [
        np.asarray([[r, z], [r + 1, z], [r + 1, z + 1], [r, z + 1]], dtype=float)
        for r in range(-2, 2)
        for z in range(-2, 2)
    ]
    centre = np.asarray(
        [[r + 0.5, z + 0.5] for r in range(-2, 2) for z in range(-2, 2)]
    )
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)
    flux = (
        1.0 - np.abs(mesh.node_coordinates[:, 0]) - np.abs(mesh.node_coordinates[:, 1])
    )

    host = mesh.clip(flux)
    traced = mesh.traced_clip(flux)
    density = 1.5 + 0.1 * centre[:, 0] - 0.2 * centre[:, 1]
    gradient = np.broadcast_to(np.asarray([0.1, -0.2]), centre.shape)
    host_moments = host.linear_current_moments(density, gradient)
    traced_current, traced_first = traced.linear_current_moments(density, gradient)

    np.testing.assert_array_equal(traced.vertex_count, host.vertex_count)
    np.testing.assert_array_equal(traced.boundary, host.boundary)
    np.testing.assert_allclose(
        traced.support_vertices, host.support_vertices, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(traced.area, host.area, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        traced_current, host_moments.current, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(
        traced_first, host_moments.first, rtol=1.0e-14, atol=1.0e-14
    )
    assert float(traced.patch_area_sum) == pytest.approx(2.0, abs=1.0e-14)
    assert float(traced.contour_area) == pytest.approx(2.0, abs=1.0e-14)


def test_clip_evaluation_weights_are_c1_without_changing_patch_geometry():
    """Evaluation weights smooth entry and hand-off, not polygon geometry."""
    import jax
    import jax.numpy as jnp

    cell = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    mesh = AtomicCellMesh.from_cells([cell], centroids=np.asarray([[0.0, 0.0]]))
    coordinates = jnp.asarray(mesh.node_coordinates)
    epsilon = 0.05

    def evaluated_current(displacement, entering):
        signed_flux = jax.lax.cond(
            entering,
            lambda: displacement - (coordinates[:, 0] + 1.0),
            lambda: 1.0 + displacement - coordinates[:, 0],
        )
        support = mesh.traced_clip(signed_flux)
        participation, clipped_path = support.evaluation_weights(epsilon)
        clipped = support.area
        full = jnp.full_like(clipped, 4.0)
        return (participation * ((1.0 - clipped_path) * full + clipped_path * clipped))[
            0
        ]

    step = 1.0e-7
    for entering, expected in ((True, 0.0), (False, 4.0)):
        value = evaluated_current(0.0, entering)
        left = jax.jvp(
            lambda shift: evaluated_current(shift, entering),
            (jnp.asarray(-step),),
            (jnp.asarray(1.0),),
        )[1]
        right = jax.jvp(
            lambda shift: evaluated_current(shift, entering),
            (jnp.asarray(step),),
            (jnp.asarray(1.0),),
        )[1]
        assert float(value) == pytest.approx(expected, abs=2.0e-14)
        assert float(left) == pytest.approx(float(right), abs=2.0e-4)

    flux = _ellipse_level(mesh.node_coordinates)
    support = mesh.traced_clip(jnp.asarray(flux))
    geometry = tuple(
        np.asarray(value)
        for value in (
            support.area,
            support.first_area_moment,
            support.second_area_moment,
            support.patch_area_sum,
        )
    )
    for width in (0.0, 1.0e-3, 1.0e-2, 1.0e-1):
        support.evaluation_weights(width)
        for observed, expected in zip(
            (
                support.area,
                support.first_area_moment,
                support.second_area_moment,
                support.patch_area_sum,
            ),
            geometry,
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected)


def test_edge_crossing_birth_is_c1_without_changing_exact_clip_geometry():
    """A zero-flux node ramps the evaluated path, not the exact support."""
    import jax
    import jax.numpy as jnp

    cell = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    mesh = AtomicCellMesh.from_cells([cell])
    base_flux = jnp.asarray([0.0, 1.0, -1.0, -1.0])
    density = jnp.asarray([1.0])
    gradient = jnp.asarray([[0.3, -0.2]])
    width = 0.05

    def evaluated(displacement, smoothing_width=width):
        flux = base_flux.at[0].set(displacement)
        support = mesh.traced_clip(flux, smoothing_width=smoothing_width)
        current, first = support.linear_current_moments(density, gradient)
        return jnp.r_[current, first.ravel()]

    step = 1.0e-7
    left = jax.jvp(evaluated, (jnp.asarray(-step),), (jnp.asarray(1.0),))[1]
    right = jax.jvp(evaluated, (jnp.asarray(step),), (jnp.asarray(1.0),))[1]
    np.testing.assert_allclose(left, right, rtol=0.0, atol=2.0e-4)

    raw_surface = evaluated(jnp.asarray(0.0), 0.0)
    np.testing.assert_allclose(
        evaluated(jnp.asarray(0.0)), raw_surface, rtol=0.0, atol=2.0e-14
    )
    for displacement in (-2.0 * width, 2.0 * width):
        np.testing.assert_allclose(
            evaluated(jnp.asarray(displacement)),
            evaluated(jnp.asarray(displacement), 0.0),
            rtol=0.0,
            atol=2.0e-14,
        )

    flux = np.asarray(base_flux.at[0].set(0.4 * width))
    raw = mesh.traced_clip(jnp.asarray(flux))
    smoothed = mesh.traced_clip(jnp.asarray(flux), smoothing_width=width)
    for observed, expected in zip(
        (
            smoothed.support_vertices,
            smoothed.vertex_count,
            smoothed.included,
            smoothed.boundary,
            smoothed.area,
            smoothed.first_area_moment,
            smoothed.second_area_moment,
            smoothed.patch_area_sum,
        ),
        (
            raw.support_vertices,
            raw.vertex_count,
            raw.included,
            raw.boundary,
            raw.area,
            raw.first_area_moment,
            raw.second_area_moment,
            raw.patch_area_sum,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(observed, expected)


def test_traced_clip_jits_once_and_vmaps_over_moving_separatrices():
    import jax
    import jax.numpy as jnp

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    cells, centre = _staggered_rectangles(15, 17)
    mesh = AtomicCellMesh.from_cells(cells, centroids=centre)
    density = jnp.asarray(2.0 + 0.2 * (centre[:, 0] - 3.0) - 0.1 * centre[:, 1])
    gradient = jnp.broadcast_to(jnp.asarray([0.2, -0.1]), centre.shape)
    flux_bank = np.asarray(
        [
            _ellipse_level(mesh.node_coordinates - np.asarray([displacement, 0.0]))
            for displacement in np.linspace(-0.02, 0.02, 12)
        ]
    )
    trace_count = 0

    def clip_and_measure(flux):
        nonlocal trace_count
        trace_count += 1
        clipped = mesh.traced_clip(flux)
        current, first = clipped.linear_current_moments(density, gradient)
        return (
            clipped.area,
            clipped.patch_area_sum,
            clipped.contour_area,
            current,
            first,
        )

    compiled = jax.jit(clip_and_measure)
    for flux in flux_bank:
        area, patch_area, contour_area, current, first = compiled(jnp.asarray(flux))
        jax.block_until_ready((area, patch_area, contour_area, current, first))
        assert abs(float(patch_area - contour_area)) < 1.0e-12
    assert trace_count == 1

    def clip_and_measure_without_count(flux):
        clipped = mesh.traced_clip(flux)
        return clipped, clipped.linear_current_moments(density, gradient)

    batched, batched_moments = jax.jit(jax.vmap(clip_and_measure_without_count))(
        jnp.asarray(flux_bank)
    )
    jax.block_until_ready((batched, batched_moments))
    np.testing.assert_allclose(
        batched.patch_area_sum, batched.contour_area, rtol=0.0, atol=1.0e-12
    )
    assert batched_moments[0].shape == (12, len(cells))
    assert batched_moments[1].shape == (12, len(cells), 2)
