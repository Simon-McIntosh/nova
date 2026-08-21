"""Branch-resolved clipping through a separatrix saddle cell."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.observation import clipped_support_quadrature
from nova.equilibrium.separatrix_clip import AtomicCellMesh, TracedClippedSupports


SADDLE = np.asarray([5.1101112852782995, -3.412348723801311])
AUDITED_SINGLE_CHORD_ERROR_A = 1.7982290765791427 - 1.584188736641894


def _audited_saddle_cell() -> tuple[AtomicCellMesh, np.ndarray]:
    half_width = np.asarray([0.2, 0.2])
    cell = SADDLE + half_width * np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    mesh = AtomicCellMesh.from_cells([cell], centroids=SADDLE[None, :])
    local = mesh.node_coordinates - SADDLE
    return mesh, local[:, 0] * local[:, 1]


def _legacy_single_chord_area(mesh: AtomicCellMesh, signed_flux: np.ndarray) -> float:
    """Reproduce the one-polygon path that joined successive crossings."""
    nodes = mesh.cell_nodes[0, : mesh.cell_vertex_count[0]]
    vertices = []
    for start, end in zip(nodes, np.roll(nodes, -1), strict=True):
        start_inside = signed_flux[start] > 0.0
        end_inside = signed_flux[end] > 0.0
        if start_inside:
            vertices.append(mesh.node_coordinates[start])
        if start_inside != end_inside:
            fraction = signed_flux[start] / (signed_flux[start] - signed_flux[end])
            vertices.append(
                mesh.node_coordinates[start]
                + fraction * (mesh.node_coordinates[end] - mesh.node_coordinates[start])
            )
    polygon = np.asarray(vertices)
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * abs(
        float(np.sum(polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]))
    )


def test_four_crossing_saddle_partitions_cell_with_explicit_branch_vertex():
    mesh, signed_flux = _audited_saddle_cell()

    inferred = mesh.traced_clip(jnp.asarray(signed_flux))
    compiled = jax.jit(
        lambda flux: mesh.traced_clip(flux, saddle_vertex=jnp.asarray(SADDLE))
    )
    common = compiled(jnp.asarray(signed_flux))
    private = compiled(jnp.asarray(-signed_flux))

    assert bool(inferred.saddle[0])
    np.testing.assert_allclose(inferred.saddle_vertex[0], SADDLE, atol=2.0e-15)
    assert bool(common.saddle[0])
    assert bool(private.saddle[0])
    np.testing.assert_array_equal(common.saddle_vertex[0], SADDLE)
    np.testing.assert_array_equal(private.saddle_vertex[0], SADDLE)
    assert int(common.vertex_count[0]) == 8
    assert int(private.vertex_count[0]) == 8
    common_saddle_count = np.count_nonzero(
        np.all(np.asarray(common.support_vertices[0]) == SADDLE, axis=1)
    )
    private_saddle_count = np.count_nonzero(
        np.all(np.asarray(private.support_vertices[0]) == SADDLE, axis=1)
    )
    assert common_saddle_count == 2
    assert private_saddle_count == 2
    np.testing.assert_array_equal(common.branch_vertex_count[0], [4, 4])
    np.testing.assert_array_equal(private.branch_vertex_count[0], [4, 4])
    np.testing.assert_array_equal(
        common.branch_support_vertices[0, :, 0], [SADDLE, SADDLE]
    )
    np.testing.assert_array_equal(
        private.branch_support_vertices[0, :, 0], [SADDLE, SADDLE]
    )

    whole_cell_area = float(common.full_area[0])
    partition_area = float(common.area[0] + private.area[0])
    assert partition_area == pytest.approx(whole_cell_area, rel=0.0, abs=2.0e-16)
    assert float(jnp.sum(common.branch_area[0])) == pytest.approx(
        float(common.area[0]), rel=0.0, abs=2.0e-16
    )
    assert float(jnp.sum(private.branch_area[0])) == pytest.approx(
        float(private.area[0]), rel=0.0, abs=2.0e-16
    )

    _points, common_weights = clipped_support_quadrature(common, jnp.asarray([True]))
    _points, private_weights = clipped_support_quadrature(private, jnp.asarray([True]))
    assert float(jnp.sum(common_weights) + jnp.sum(private_weights)) == pytest.approx(
        whole_cell_area, rel=0.0, abs=2.0e-15
    )


def test_branch_profiles_remove_the_audited_single_chord_current_error():
    mesh, signed_flux = _audited_saddle_cell()
    support = mesh.traced_clip(jnp.asarray(signed_flux), saddle_vertex=SADDLE)
    legacy_area = _legacy_single_chord_area(mesh, signed_flux)
    exact_area = float(support.area[0])
    density = AUDITED_SINGLE_CHORD_ERROR_A / (legacy_area - exact_area)

    branch_density = jnp.asarray([[density, density]])
    branch_gradient = jnp.zeros((1, 2, 2))
    branch_current, _branch_first = support.branch_linear_current_moments(
        branch_density, branch_gradient
    )
    resolved_current = float(jnp.sum(branch_current))
    legacy_current = density * legacy_area
    exact_current = density * exact_area

    assert abs(legacy_current - exact_current) == pytest.approx(
        AUDITED_SINGLE_CHORD_ERROR_A, rel=0.0, abs=2.0e-15
    )
    assert resolved_current == pytest.approx(exact_current, rel=0.0, abs=2.0e-15)
    assert abs(resolved_current - exact_current) <= 2.0e-15

    private_zero_density = branch_density.at[0, 1].set(0.0)
    selected_current, _selected_first = support.branch_linear_current_moments(
        private_zero_density, branch_gradient
    )
    assert float(selected_current[0, 1]) == 0.0
    assert float(selected_current[0, 0]) != 0.0


def test_traced_support_contract_is_additive_and_keeps_existing_shapes():
    mesh, signed_flux = _audited_saddle_cell()
    support = mesh.traced_clip(jnp.asarray(signed_flux), saddle_vertex=SADDLE)

    existing_fields = (
        "support_vertices",
        "vertex_count",
        "centroids",
        "included",
        "boundary",
        "area",
        "full_area",
        "first_area_moment",
        "second_area_moment",
        "contour_area",
        "patch_area_sum",
    )
    assert TracedClippedSupports._fields[: len(existing_fields)] == existing_fields
    assert support.support_vertices.shape == (1, mesh.support_capacity, 2)
    assert support.vertex_count.shape == (1,)
    assert support.centroids.shape == (1, 2)
    assert support.included.shape == (1,)
    assert support.boundary.shape == (1,)
    assert support.area.shape == (1,)
    assert support.full_area.shape == (1,)
    assert support.first_area_moment.shape == (1, 2)
    assert support.second_area_moment.shape == (1, 2, 2)
    assert support.contour_area.shape == ()
    assert support.patch_area_sum.shape == ()
