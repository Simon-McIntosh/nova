"""Density moments integrated over the clip's own sampled arc."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.clip_quadrature import (
    clipped_support_current_moments,
    clipped_support_quadrature,
)

try:
    from nova.equilibrium.clip_quadrature import (
        cut_capacity_edge_bound,
        cut_cell_moment_evaluation_bound,
    )
except ImportError:
    cut_capacity_edge_bound = None
    cut_cell_moment_evaluation_bound = None
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


class _Support(NamedTuple):
    support_vertices: jax.Array
    vertex_count: jax.Array
    centroids: jax.Array
    included: jax.Array
    boundary: jax.Array


class _QuadraticDensity:
    def current_density(self, radius, psi_norm):
        return 1.5 + 0.75 * radius - 2.0 * psi_norm + 0.5 * psi_norm**2


_BOUNDARY_ROUTE_AVAILABLE = cut_cell_moment_evaluation_bound is not None
requires_boundary_route = pytest.mark.skipif(
    not _BOUNDARY_ROUTE_AVAILABLE,
    reason="boundary reduction is absent at the comparison revision",
)


@pytest.fixture(scope="module", autouse=True)
def _binary64_cpu() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"


def _curved_support(segments: int, *, capacity: int | None = None) -> _Support:
    parameter = np.linspace(0.0, 1.0, segments + 1)
    arc = np.column_stack(
        (
            1.5 + parameter,
            0.5 + 0.4 * 4.0 * parameter * (1.0 - parameter),
        )
    )
    polygon = np.vstack((arc, [2.5, -0.5], [1.5, -0.5]))
    width = len(polygon) if capacity is None else capacity
    vertices = np.zeros((1, width, 2), dtype=np.float64)
    vertices[0, : len(polygon)] = polygon
    return _Support(
        support_vertices=jnp.asarray(vertices),
        vertex_count=jnp.asarray([len(polygon)]),
        centroids=jnp.asarray([[2.0, 0.0]]),
        included=jnp.asarray([True]),
        boundary=jnp.asarray([True]),
    )


def _field(cell_count: int = 1) -> FluxFieldPolynomial:
    return FluxFieldPolynomial(
        coefficient=jnp.tile(
            jnp.asarray([[0.2, 0.3, -0.25, 0.0, 0.0, 0.0]]),
            (cell_count, 1),
        ),
        centre=jnp.tile(jnp.asarray([[2.0, 0.0]]), (cell_count, 1)),
        scale=jnp.ones((cell_count, 2)),
        active=jnp.ones(cell_count, dtype=bool),
    )


def _quadratic_flux_field(cell_count: int = 1) -> FluxFieldPolynomial:
    """A genuinely quadratic flux: every second-order monomial is carried."""
    return FluxFieldPolynomial(
        coefficient=jnp.tile(
            jnp.asarray([[0.2, 0.3, -0.25, 0.4, 0.3, -0.35]]),
            (cell_count, 1),
        ),
        centre=jnp.tile(jnp.asarray([[2.0, 0.0]]), (cell_count, 1)),
        scale=jnp.ones((cell_count, 2)),
        active=jnp.ones(cell_count, dtype=bool),
    )


def _fan_moments(support: _Support) -> np.ndarray:
    points, weights = clipped_support_quadrature(support, support.included)
    psi_norm, _radial, _vertical = _field().sample(points)
    density = _QuadraticDensity().current_density(points[..., 0], psi_norm)
    weighted = density * weights
    first = jnp.sum(
        weighted[..., None] * (points - support.centroids[:, None, :]), axis=1
    )
    return np.asarray([jnp.sum(weighted), jnp.sum(first[:, 0]), jnp.sum(first[:, 1])])


@requires_boundary_route
def test_quadratic_boundary_moments_reach_the_fan_refinement_floor():
    support = _curved_support(128)
    refined = _curved_support(256)
    observed = clipped_support_current_moments(
        support,
        support.included,
        _field(),
        _QuadraticDensity(),
        cut_cell_capacity=1,
        boundary_reduction=True,
    )
    observed_values = np.asarray(
        [
            observed.cell_current[0],
            observed.radial_moment[0],
            observed.vertical_moment[0],
        ]
    )
    fan = _fan_moments(support)
    refined_fan = _fan_moments(refined)
    fan_floor = np.abs(refined_fan - fan)
    difference = np.abs(observed_values - fan)
    roundoff = 64.0 * np.finfo(np.float64).eps * np.maximum(np.abs(refined_fan), 1.0)
    np.testing.assert_array_less(difference, 1.5 * fan_floor + roundoff)


def test_default_cut_moments_match_the_fan_to_roundoff():
    support = _curved_support(128)
    observed = clipped_support_current_moments(
        support,
        support.included,
        _field(),
        _QuadraticDensity(),
        cut_cell_capacity=1,
    )
    observed_values = np.asarray(
        [
            observed.cell_current[0],
            observed.radial_moment[0],
            observed.vertical_moment[0],
        ]
    )
    np.testing.assert_allclose(
        observed_values,
        _fan_moments(support),
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )


@requires_boundary_route
def test_shifted_first_moment_paths_set_the_required_per_edge_order():
    """The order is set by the first moments, not by the zeroth one."""
    from benchmarks.exact_clip_moment_floor import edge_order_study

    from nova.equilibrium.clip_quadrature import _ARC_EDGE_ORDER

    fixture_field = _quadratic_flux_field()
    fixture_coefficient = np.asarray(fixture_field.coefficient)[0]
    assert np.all(fixture_coefficient[3:] != 0.0)
    study = edge_order_study(
        flux=tuple(Fraction(str(value)) for value in fixture_coefficient),
        sliver=(
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(63, 100), Fraction(35, 1000)),
        ),
    )
    defect = study["relative_defect_by_path_and_order"]
    assert study["integrand_degree_in_edge_parameter"] == {
        "zero": 5,
        "radial_shift": 6,
        "vertical_shift": 6,
    }
    # Positive control on the fixture: the zeroth path is exact at the third
    # order while both shifted paths are not, so the fixture can distinguish the
    # two rules. A linear flux would make every path exact and the test vacuous.
    assert defect["zero"]["3"] <= 1e-12
    assert defect["radial_shift"]["3"] > 1e-12
    assert defect["vertical_shift"]["3"] > 1e-12
    assert study["lowest_order_exact_on_every_path"] == 4
    assert _ARC_EDGE_ORDER == study["lowest_order_exact_on_every_path"]
    for path in ("radial_shift", "vertical_shift"):
        assert defect[path]["4"] <= 1e-12


@requires_boundary_route
def test_order_study_receipt_carries_the_map_the_record_is_transcribed_from():
    """The independent Gauss reference records its coefficients and defects."""
    import json
    import tempfile
    from pathlib import Path

    from benchmarks.exact_clip_moment_floor import (
        EDGE_ORDER_STUDY_EXACTNESS_THRESHOLD,
        write_edge_order_study_receipt,
    )

    from nova.equilibrium.clip_quadrature import _ARC_EDGE_ORDER

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "edge-order-study.json"
        receipt = write_edge_order_study_receipt(path)
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
    assert receipt["schema"] == "nova.exact-clip-edge-order-study.v1"
    assert receipt["selected_per_edge_order"] == _ARC_EDGE_ORDER
    assert receipt["exactness_threshold"] == EDGE_ORDER_STUDY_EXACTNESS_THRESHOLD
    study = receipt["study"]
    assert study["study_flux_quadratic_coefficients"] == [0.2, 0.3, -0.35]
    defect = study["relative_defect_by_path_and_order"]
    assert defect["radial_shift"]["1"] == pytest.approx(2.53e-1, rel=0.01)
    assert defect["zero"]["1"] == pytest.approx(2.61e-3, rel=0.01)
    # The selection is the smallest order that satisfies every path at once, so
    # it follows from the map rather than being a constant restated beside it.
    first_exact = {
        path_name: min(int(order) for order, value in orders.items() if value <= 1e-12)
        for path_name, orders in defect.items()
    }
    assert first_exact["zero"] == 3
    assert set(first_exact.values()) == {3, _ARC_EDGE_ORDER}
    assert max(first_exact.values()) == _ARC_EDGE_ORDER
    assert study["lowest_order_exact_on_every_path"] == _ARC_EDGE_ORDER


@requires_boundary_route
def test_arc_route_carries_the_fixed_edge_and_point_bound():
    support = _curved_support(128, capacity=3072)
    edges = cut_capacity_edge_bound()
    assert edges == 149
    # Endpoint moments sample only the density fit, independently of edge count.
    assert cut_cell_moment_evaluation_bound() == 25
    observed = np.asarray(
        clipped_support_current_moments(
            support,
            support.included,
            _field(),
            _QuadraticDensity(),
            cut_cell_capacity=1,
            boundary_reduction=True,
        )
    )
    assert np.all(np.isfinite(observed))


def test_endpoint_recurrence_integrates_every_monomial_through_degree_six():
    from nova.equilibrium.clip_quadrature import _straight_edge_monomial_moments

    vertices = jnp.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    observed = np.asarray(
        jax.jit(
            lambda point: _straight_edge_monomial_moments(
                point, jnp.asarray([3]), max_degree=6
            )
        )(vertices)
    )[0]
    expected = np.asarray(
        [
            factorial(p) * factorial(total - p) / factorial(total + 2)
            for total in range(7)
            for p in range(total + 1)
        ]
    )
    assert len(expected) == 28 and np.all(expected > 0)
    np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_endpoint_moments_ignore_padding_and_match_a_translated_rectangle(reverse):
    from nova.equilibrium.clip_quadrature import _straight_edge_monomial_moments

    polygon = np.asarray([[0.2, -0.4], [1.3, -0.4], [1.3, 0.8], [0.2, 0.8]])
    if reverse:
        polygon = polygon[::-1]
    vertices = np.full((1, 19, 2), np.nan)
    vertices[0, :4] = polygon
    observed = np.asarray(
        _straight_edge_monomial_moments(vertices, np.asarray([4]), max_degree=6)
    )[0]
    expected = np.asarray(
        [
            (1.3 ** (p + 1) - 0.2 ** (p + 1))
            * (0.8 ** (q + 1) - (-0.4) ** (q + 1))
            / ((p + 1) * (q + 1))
            for total in range(7)
            for p in range(total + 1)
            for q in (total - p,)
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=3e-14, atol=0)


def test_sampled_density_moments_and_vertex_tangent_match_independent_fan():
    from nova.equilibrium.clip_quadrature import (
        _DENSITY_POWERS,
        _sampled_arc_polynomial_moments,
    )

    polygon = np.asarray(
        [[0.1, -0.2], [1.3, -0.35], [1.9, 0.6], [1.1, 1.4], [-0.2, 0.9]]
    )
    coefficients = np.linspace(0.1, 0.8, len(_DENSITY_POWERS))
    centre = np.asarray([[2.0, -0.5]])
    scale = np.asarray([[0.7, 1.3]])
    moment_centre = np.asarray([[2.1, -0.4]])
    vertices = centre[:, None] + scale[:, None] * polygon[None]

    def integrate(point):
        return jnp.stack(
            _sampled_arc_polynomial_moments(
                point,
                jnp.asarray([len(polygon)]),
                centre,
                scale,
                coefficients[None],
                moment_centre,
            )
        )[:, 0]

    def reference(point):
        local_polygon = (point[0] - centre[0]) / scale[0]
        points, weights = fixture._polygon_rule(local_polygon, order=8)
        density = sum(
            c * points[:, 0] ** p * points[:, 1] ** q
            for c, (p, q) in zip(coefficients, _DENSITY_POWERS, strict=True)
        )
        weighted = density * weights * np.prod(scale)
        offset = centre + scale * points - moment_centre
        return np.asarray([np.sum(weighted), *(weighted @ offset)])

    actual = np.asarray(jax.jit(integrate)(jnp.asarray(vertices)))
    np.testing.assert_allclose(actual, reference(vertices), rtol=2e-14, atol=0)
    direction = np.linspace(-0.3, 0.4, vertices.size).reshape(vertices.shape)
    _, tangent = jax.jvp(integrate, (jnp.asarray(vertices),), (jnp.asarray(direction),))
    epsilon = 1e-5
    difference = (
        reference(vertices + epsilon * direction)
        - reference(vertices - epsilon * direction)
    ) / (2 * epsilon)
    assert np.all(np.abs(difference) > 1e-3)
    np.testing.assert_allclose(tangent, difference, rtol=2e-9, atol=0)


@requires_boundary_route
def test_weak_cut_cells_resolve_the_fixed_arc_layout():
    carrier_case, source_case, exact = certificate._case("weak-rotation-reactor-static")
    machine = certificate._case_machine(
        "weak-rotation-reactor-static", carrier_case, exact, -110
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = certificate._exact_state("weak-rotation-reactor-static", exact, coordinates)
    operator = fixture.forward_operator(source_case, machine)
    support = fixture._analytic_profile_support(exact, operator, state)
    cells = np.asarray([35, 36, 54, 101, 102, 128])
    np.testing.assert_array_equal(
        np.asarray(support.vertex_count)[cells],
        np.asarray([131, 132, 132, 131, 131, 131]),
    )
    assert np.all(np.asarray(support.boundary)[cells])


@requires_boundary_route
def test_empty_sampled_arc_refuses_with_nonfinite_moments():
    one = _curved_support(128)
    support = _Support(
        support_vertices=one.support_vertices,
        vertex_count=jnp.zeros(1, dtype=one.vertex_count.dtype),
        centroids=one.centroids,
        included=one.included,
        boundary=one.boundary,
    )
    moments = clipped_support_current_moments(
        support,
        support.included,
        _field(),
        _QuadraticDensity(),
        cut_cell_capacity=1,
        boundary_reduction=True,
    )
    assert np.all(~np.isfinite(np.asarray(moments)))


def test_cut_cell_bank_overflow_remains_fail_closed():
    one = _curved_support(128)
    support = _Support(
        support_vertices=jnp.tile(one.support_vertices, (2, 1, 1)),
        vertex_count=jnp.tile(one.vertex_count, 2),
        centroids=jnp.tile(one.centroids, (2, 1)),
        included=jnp.ones(2, dtype=bool),
        boundary=jnp.ones(2, dtype=bool),
    )
    moments = clipped_support_current_moments(
        support,
        support.included,
        _field(2),
        _QuadraticDensity(),
        cut_cell_capacity=1,
    )
    assert np.all(~np.isfinite(np.asarray(moments)))
