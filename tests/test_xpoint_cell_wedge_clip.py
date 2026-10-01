"""Fixed-capacity branch pairing in a four-crossing saddle cell."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.clip_quadrature import saddle_wedge_current_moments
from nova.equilibrium.separatrix_clip import (
    AtomicCellMesh,
    _sampled_spline_edge_roots,
    _traced_level_arc,
)
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True


def _saddle_cell() -> tuple[AtomicCellMesh, jax.Array]:
    vertices = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    mesh = AtomicCellMesh.from_cells([vertices], centroids=np.zeros((1, 2)))
    coordinates = mesh.node_coordinates
    return mesh, jnp.asarray(coordinates[:, 0] * coordinates[:, 1])


def _saddle_level(points):
    return points[..., 0] * points[..., 1]


def test_spline_chain_admits_the_supplied_saddle_vertex():
    mesh, signed_flux = _saddle_cell()

    support = mesh.traced_clip(
        signed_flux,
        saddle_vertex=jnp.zeros(2),
        curve_evaluator=_saddle_level,
        arc_tracer=_traced_level_arc,
    )

    assert bool(support.saddle[0])
    assert support.refused_cells() == 0
    np.testing.assert_array_equal(support.saddle_vertex[0], np.zeros(2))
    assert np.all(np.asarray(support.branch_vertex_count[0]) > mesh.support_capacity)
    for vertices, count in zip(
        np.asarray(support.branch_support_vertices[0]),
        np.asarray(support.branch_vertex_count[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(vertices[0], np.zeros(2))
        np.testing.assert_array_equal(vertices[count:], 0.0)


def test_spline_chain_refuses_a_non_finite_supplied_saddle():
    mesh, signed_flux = _saddle_cell()

    support = mesh.traced_clip(
        signed_flux,
        saddle_vertex=jnp.asarray([jnp.nan, 0.0]),
        curve_evaluator=_saddle_level,
        arc_tracer=_traced_level_arc,
    )

    assert not bool(support.saddle[0])
    assert support.refused_cells() == 1
    np.testing.assert_array_equal(support.saddle_vertex, 0.0)
    assert not bool(support.included[0])
    np.testing.assert_array_equal(support.area, 0.0)
    np.testing.assert_array_equal(support.vertex_count, 0)
    for branch in (support.branch_support_vertices, support.branch_area):
        assert np.all(np.isfinite(np.asarray(branch)))


def test_forward_operator_supplies_the_typed_census_saddle_to_the_clip():
    from nova.equilibrium import forward_operator

    source = inspect.getsource(forward_operator.ForwardFluxOperator._profile_support)
    calls = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_traced_clip"
    ]

    assert len(calls) == 1
    supplied = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert "saddle_vertex" in supplied, "the clip is called with no saddle vertex"
    attribute = supplied["saddle_vertex"]
    assert isinstance(attribute, ast.Attribute)
    assert isinstance(attribute.value, ast.Name)
    assert (attribute.value.id, attribute.attr) == ("topology", "x_point")


def test_spline_chain_does_not_infer_a_saddle_from_crossing_chords():
    mesh, signed_flux = _saddle_cell()

    support = mesh.traced_clip(
        signed_flux,
        curve_evaluator=_saddle_level,
        arc_tracer=_traced_level_arc,
    )

    assert not bool(support.saddle[0])
    assert support.refused_cells() == 1
    np.testing.assert_array_equal(support.saddle_vertex, 0.0)


def test_spline_chain_emits_four_profile_owned_wedges():
    mesh, signed_flux = _saddle_cell()

    wedges = mesh.traced_saddle_wedges(
        signed_flux,
        saddle_vertex=jnp.zeros(2),
        core_reference=jnp.ones(2),
        curve_evaluator=_saddle_level,
        arc_tracer=_traced_level_arc,
    )
    density = jnp.asarray([[2.0, 0.0, 0.0, 0.0]])
    gradient = jnp.zeros((1, 4, 2), dtype=jnp.float64)
    current, _first = wedges.linear_current_moments(density, gradient)

    assert bool(wedges.saddle[0])
    assert np.all(np.asarray(wedges.vertex_count[0]) > mesh.support_capacity)
    np.testing.assert_allclose(wedges.area[0], np.ones(4), rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(current, [[2.0, 0.0, 0.0, 0.0]], atol=2.0e-14)
    np.testing.assert_array_equal(wedges.saddle_vertex[0], np.zeros(2))
    for vertices, count in zip(
        np.asarray(wedges.support_vertices[0]),
        np.asarray(wedges.vertex_count[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(vertices[0], np.zeros(2))
        np.testing.assert_array_equal(vertices[count:], 0.0)


def test_spline_chain_retains_two_roots_on_one_edge():
    mesh, _signed_flux = _saddle_cell()
    saddle = jnp.asarray([0.0, 0.8])

    def displaced_saddle_level(points):
        return (points[..., 1] - saddle[1]) ** 2 - points[..., 0] ** 2

    signed_flux = displaced_saddle_level(jnp.asarray(mesh.node_coordinates))
    fractions, counts, _positive_after = _sampled_spline_edge_roots(
        mesh.node_coordinates,
        mesh.cell_nodes,
        mesh.cell_vertex_count,
        displaced_saddle_level,
        None,
    )
    wedges = mesh.traced_saddle_wedges(
        signed_flux,
        saddle_vertex=saddle,
        core_reference=jnp.asarray([0.0, -1.0]),
        curve_evaluator=displaced_saddle_level,
        arc_tracer=_traced_level_arc,
    )

    np.testing.assert_array_equal(counts[0], [0, 1, 2, 1])
    np.testing.assert_allclose(fractions[0, 2], [0.4, 0.6], atol=2.0e-14)
    assert bool(wedges.saddle[0])
    assert np.all(np.asarray(wedges.vertex_count[0]) > mesh.support_capacity)
    assert float(jnp.sum(wedges.area[0])) == pytest.approx(
        float(wedges.full_area[0]), rel=0.0, abs=2.0e-13
    )


class _FlatField:
    @staticmethod
    def sample(points, _cell_index):
        zero = jnp.zeros(points.shape[:-1], dtype=points.dtype)
        return zero, zero, zero


@dataclass(frozen=True)
class _ConstantProfile:
    value: float

    def current_density(self, radius, _normalised_flux):
        return jnp.full_like(radius, self.value)


def test_saddle_cell_emits_core_private_and_two_sol_wedges():
    mesh, signed_flux = _saddle_cell()
    compiled = jax.jit(
        lambda flux: mesh.traced_saddle_wedges(
            flux,
            saddle_vertex=jnp.zeros(2),
            core_reference=jnp.ones(2),
        )
    )
    wedges = compiled(signed_flux)

    assert bool(wedges.saddle[0])
    assert wedges.support_vertices.shape == (1, 4, mesh.support_capacity, 2)
    np.testing.assert_array_equal(wedges.vertex_count[0], [4, 4, 4, 4])
    np.testing.assert_allclose(wedges.area[0], np.ones(4), rtol=0.0, atol=2.0e-15)
    assert float(jnp.sum(wedges.area[0])) == pytest.approx(
        float(wedges.full_area[0]), rel=0.0, abs=2.0e-15
    )

    geometric_centres = np.asarray(
        wedges.first_area_moment[0] / wedges.area[0, :, None]
    )
    assert np.all(geometric_centres[0] > 0.0)
    assert np.all(geometric_centres[1] < 0.0)
    assert np.prod(geometric_centres[2]) < 0.0
    assert np.prod(geometric_centres[3]) < 0.0
    for wedge, count in zip(
        np.asarray(wedges.support_vertices[0]),
        np.asarray(wedges.vertex_count[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(wedge[0], np.zeros(2))
        np.testing.assert_array_equal(wedge[count:], 0.0)


def test_per_wedge_profiles_integrate_without_cross_region_current():
    mesh, signed_flux = _saddle_cell()
    wedges = mesh.traced_saddle_wedges(
        signed_flux,
        saddle_vertex=jnp.zeros(2),
        core_reference=jnp.ones(2),
    )
    density = jnp.asarray([[2.0, 0.0, 0.0, 0.0]])
    gradient = jnp.zeros((1, 4, 2), dtype=jnp.float64)
    exact_current, exact_first = wedges.linear_current_moments(density, gradient)
    quadrature = saddle_wedge_current_moments(
        wedges,
        _FlatField(),
        tuple(_ConstantProfile(value) for value in (2.0, 0.0, 0.0, 0.0)),
    )

    np.testing.assert_allclose(exact_current, [[2.0, 0.0, 0.0, 0.0]], atol=2.0e-15)
    np.testing.assert_allclose(exact_first[0, 0], [1.0, 1.0], atol=2.0e-15)
    np.testing.assert_allclose(quadrature.cell_current, exact_current, atol=2.0e-14)
    np.testing.assert_allclose(
        quadrature.radial_moment, exact_first[..., 0], atol=2.0e-14
    )
    np.testing.assert_allclose(
        quadrature.vertical_moment, exact_first[..., 1], atol=2.0e-14
    )
    np.testing.assert_array_equal(quadrature.cell_current[0, 1:], 0.0)


def test_non_saddle_rows_and_padding_are_exact_zero_with_static_shapes():
    mesh, signed_flux = _saddle_cell()
    compiled = jax.jit(
        lambda flux: mesh.traced_saddle_wedges(
            flux,
            saddle_vertex=jnp.zeros(2),
            core_reference=jnp.ones(2),
        )
    )
    saddle = compiled(signed_flux)
    ordinary = compiled(jnp.asarray(mesh.node_coordinates[:, 0]))

    assert saddle.support_vertices.shape == ordinary.support_vertices.shape
    assert saddle.vertex_count.shape == ordinary.vertex_count.shape
    assert not bool(ordinary.saddle[0])
    np.testing.assert_array_equal(ordinary.vertex_count, 0)
    np.testing.assert_array_equal(ordinary.support_vertices, 0.0)
    np.testing.assert_array_equal(ordinary.area, 0.0)


def test_two_roots_on_one_edge_still_emit_four_wedges():
    mesh, _signed_flux = _saddle_cell()
    signed_flux = jnp.asarray([-1.0, 1.0, 1.0, 1.0])
    fractions = np.zeros((1, 4, 2))
    fractions[0, 0, 0] = 0.5
    fractions[0, 2] = (0.25, 0.75)
    fractions[0, 3, 0] = 0.5
    counts = np.asarray([[1, 0, 2, 1]])
    positive_after = np.zeros((1, 4, 2), dtype=bool)
    positive_after[0, 0, 0] = True
    positive_after[0, 2] = (False, True)

    wedges = jax.jit(
        lambda flux: mesh.traced_saddle_wedges(
            flux,
            saddle_vertex=jnp.zeros(2),
            core_reference=jnp.ones(2),
            edge_root_fraction=jnp.asarray(fractions),
            edge_root_count=jnp.asarray(counts),
            edge_root_positive_after=jnp.asarray(positive_after),
        )
    )(signed_flux)

    assert bool(wedges.saddle[0])
    assert np.all(np.asarray(wedges.vertex_count[0]) >= 3)
    assert float(jnp.sum(wedges.area[0])) == pytest.approx(
        float(wedges.full_area[0]), rel=0.0, abs=2.0e-15
    )
    for vertices, count in zip(
        np.asarray(wedges.support_vertices[0]),
        np.asarray(wedges.vertex_count[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(vertices[0], np.zeros(2))
        np.testing.assert_array_equal(vertices[count:], 0.0)


def test_edge_coincident_saddle_keeps_one_exact_zero_area_wedge():
    mesh, _signed_flux = _saddle_cell()
    saddle = jnp.asarray([0.0, 1.0])
    fractions = np.zeros((1, 4, 2))
    fractions[0, 0] = (0.25, 0.75)
    fractions[0, 2] = (0.5, 0.5)
    counts = np.asarray([[2, 0, 2, 0]])
    positive_after = np.zeros((1, 4, 2), dtype=bool)
    positive_after[0, 0] = (False, True)
    positive_after[0, 2] = (False, True)

    wedges = mesh.traced_saddle_wedges(
        jnp.ones(4),
        saddle_vertex=saddle,
        core_reference=jnp.asarray([0.0, 0.0]),
        edge_root_fraction=jnp.asarray(fractions),
        edge_root_count=jnp.asarray(counts),
        edge_root_positive_after=jnp.asarray(positive_after),
    )

    assert bool(wedges.saddle[0])
    assert np.count_nonzero(np.asarray(wedges.area[0]) == 0.0) == 1
    assert float(jnp.sum(wedges.area[0])) == pytest.approx(
        float(wedges.full_area[0]), rel=0.0, abs=2.0e-15
    )
    zero_wedge = int(np.flatnonzero(np.asarray(wedges.area[0]) == 0.0)[0])
    count = int(wedges.vertex_count[0, zero_wedge])
    vertices = np.asarray(wedges.support_vertices[0, zero_wedge])
    np.testing.assert_array_equal(vertices[0], saddle)
    np.testing.assert_array_equal(vertices[count:], 0.0)


def test_wedge_profile_count_refuses_an_incomplete_region_declaration():
    mesh, signed_flux = _saddle_cell()
    wedges = mesh.traced_saddle_wedges(
        signed_flux,
        saddle_vertex=jnp.zeros(2),
        core_reference=jnp.ones(2),
    )
    with pytest.raises(ValueError, match="exactly four wedge profiles"):
        saddle_wedge_current_moments(
            wedges,
            _FlatField(),
            (_ConstantProfile(1.0),) * 3,
        )


def _signed_polygon_area(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    return 0.5 * float(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )


def test_a_root_fraction_names_a_different_segment_in_each_traversal():
    """A fraction is an edge index, so the traversal it is read in is the contract."""
    from benchmarks.xpoint_cell_wedge_oracle import _orientation_normalised

    clockwise = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    assert _signed_polygon_area(clockwise) < 0.0

    as_exposed = AtomicCellMesh.from_cells([clockwise], centroids=np.zeros((1, 2)))
    normalised = _orientation_normalised(clockwise)
    aligned = AtomicCellMesh.from_cells([normalised], centroids=np.zeros((1, 2)))

    assert _signed_polygon_area(normalised) > 0.0
    np.testing.assert_array_equal(normalised, clockwise[::-1])
    np.testing.assert_array_equal(
        np.asarray(as_exposed.node_coordinates), clockwise[::-1]
    )
    np.testing.assert_array_equal(np.asarray(aligned.node_coordinates), normalised)

    exposed_coordinates = np.asarray(as_exposed.node_coordinates)
    midpoint_as_read = clockwise[0] + 0.5 * (clockwise[1] - clockwise[0])
    midpoint_as_stored = exposed_coordinates[0] + 0.5 * (
        exposed_coordinates[1] - exposed_coordinates[0]
    )
    np.testing.assert_allclose(midpoint_as_read, [-1.0, 0.0])
    np.testing.assert_allclose(midpoint_as_stored, [1.0, 0.0])
    assert np.linalg.norm(midpoint_as_read - midpoint_as_stored) > 1.0


def test_saddle_wedges_wind_consistently_and_fill_the_cell_interior():
    mesh, _signed_flux = _saddle_cell()
    saddle = jnp.asarray([0.0, 0.8])

    def displaced_saddle_level(points):
        return (points[..., 1] - saddle[1]) ** 2 - points[..., 0] ** 2

    signed_flux = displaced_saddle_level(jnp.asarray(mesh.node_coordinates))
    wedges = mesh.traced_saddle_wedges(
        signed_flux,
        saddle_vertex=saddle,
        core_reference=jnp.asarray([0.0, -1.0]),
        curve_evaluator=displaced_saddle_level,
        arc_tracer=_traced_level_arc,
    )

    vertices = np.asarray(wedges.support_vertices)[0]
    counts = np.asarray(wedges.vertex_count)[0]
    signed_area = np.asarray(
        [
            _signed_polygon_area(wedge[:count])
            for wedge, count in zip(vertices, counts, strict=True)
        ]
    )
    interior = abs(_signed_polygon_area(np.asarray(mesh.node_coordinates)))

    assert np.all(signed_area > 0.0), "wedges do not wind consistently"
    assert float(np.max(np.abs(signed_area))) <= interior + 1.0e-14
    assert float(np.sum(signed_area)) == pytest.approx(interior, rel=0.0, abs=2.0e-13)
