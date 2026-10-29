"""Closed-form support-boundary and bank Jacobian checks."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys
from typing import NamedTuple
from types import SimpleNamespace

from nova.equilibrium import clip_quadrature, current, support_read, topology
from scripts.prototypes.bank_newton_step import measure
from scripts.prototypes.bank_newton_step.measure import (
    _assert_update_agreement,
    _terminal_verdict,
)


class ConstantDensity:
    def current_density(self, radius, psi_norm):
        return jnp.ones_like(radius)


def test_level_set_boundary_integral_matches_area_derivative():
    """A horizontal level moves an exact two-unit-wide clipped boundary."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    measured = clip_quadrature.level_set_current_moment_tangent(
        vertices,
        jnp.asarray(4),
        jnp.zeros(2),
        jnp.asarray(1.0),
        coefficient,
        tangent,
        selected,
        ConstantDensity(),
    )
    step = 1e-5
    plus = topology.quadratic_cell_fragments(vertices, 4, coefficient + step * tangent)
    minus = topology.quadratic_cell_fragments(vertices, 4, coefficient - step * tangent)
    difference = (jnp.sum(plus.area) - jnp.sum(minus.area)) / (2.0 * step)
    assert jax.config.jax_enable_x64 is True
    np.testing.assert_allclose(measured.cell_current, 2.0, rtol=1e-6)
    np.testing.assert_allclose(measured.cell_current, difference, rtol=1e-6)
    np.testing.assert_allclose(measured.radial_moment, 0.0, atol=1e-8)


def test_curved_level_boundary_integral_matches_area_derivative():
    """The conic tangent includes both sides of a closed curved contour."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.49, 0.0, 0.0, -1.0, 0.0, -1.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    measured = clip_quadrature.level_set_current_moment_tangent(
        vertices,
        4,
        jnp.zeros(2),
        1.0,
        coefficient,
        tangent,
        selected,
        ConstantDensity(),
    )
    step = 1e-5
    plus = topology.quadratic_cell_fragments(vertices, 4, coefficient + step * tangent)
    minus = topology.quadratic_cell_fragments(vertices, 4, coefficient - step * tangent)
    difference = (jnp.sum(plus.area) - jnp.sum(minus.area)) / (2.0 * step)
    np.testing.assert_allclose(measured.cell_current, np.pi, rtol=1e-6)
    np.testing.assert_allclose(measured.cell_current, difference, rtol=1e-6)


def test_fixed_polygon_density_chain_rule():
    """The interior term integrates the supplied flux-density derivative."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    measured = clip_quadrature.fixed_polygon_current_moment_tangent(
        vertices,
        jnp.asarray(4),
        jnp.zeros(2),
        1.0,
        coefficient,
        tangent,
        lambda radius, flux: 2.0 * jnp.ones_like(radius),
    )
    np.testing.assert_allclose(measured.cell_current, 8.0, rtol=1e-12)
    np.testing.assert_allclose(measured.radial_moment, 0.0, atol=1e-12)
    np.testing.assert_allclose(measured.vertical_moment, 0.0, atol=1e-12)


def test_sampled_conic_vertex_velocity_matches_central_difference():
    """The polygonal boundary velocity follows its moving conic roots."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.49, 0.0, 0.0, -1.0, 0.0, -1.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    velocity = current._conic_ring_tangents(
        vertices, 4, jnp.zeros(2), 1.0, coefficient, tangent, selected, 64
    )
    step = 1e-6
    plus = current._conic_rings(
        vertices, 4, jnp.zeros(2), 1.0, coefficient + step * tangent, selected, 64
    )[0]
    minus = current._conic_rings(
        vertices, 4, jnp.zeros(2), 1.0, coefficient - step * tangent, selected, 64
    )[0]
    # At a vertical tangency the two sampled sides share a zero-length edge;
    # its endpoint velocity is immaterial to the polygon integral.
    interior = np.delete(np.arange(velocity.shape[1]), [0, 64, 65, 129])
    np.testing.assert_allclose(
        velocity[:, interior], ((plus - minus) / (2 * step))[:, interior], atol=1e-6
    )


def test_sampled_conic_shape_integral_matches_discrete_area_derivative():
    """The moving ring integral follows the same chords as the booking."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.49, 0.0, 0.0, -1.0, 0.0, -1.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    rings, counts, _ = current._conic_rings(
        vertices, 4, jnp.zeros(2), 1.0, coefficient, selected, 64
    )
    velocity = current._conic_ring_tangents(
        vertices, 4, jnp.zeros(2), 1.0, coefficient, tangent, selected, 64
    )
    boundary = jax.vmap(
        lambda ring, speed, count: (
            clip_quadrature.moving_polygon_current_moment_tangent(
                ring, speed, count, jnp.zeros(2), 1.0, coefficient, ConstantDensity()
            )
        )
    )(rings, velocity, counts)

    def area(shift):
        moving, number, _ = current._conic_rings(
            vertices, 4, jnp.zeros(2), 1.0, coefficient + shift * tangent, selected, 64
        )
        points, weights = clip_quadrature._quadrature_from_arrays(
            moving, number, jnp.zeros((len(moving), 2)), number >= 3, signed=True
        )
        del points
        return jnp.sum(weights)

    step = 1e-4
    difference = (area(step) - area(-step)) / (2 * step)
    np.testing.assert_allclose(jnp.sum(boundary.cell_current), difference, rtol=1e-6)


def test_krylov_condition_estimate_resolves_two_scales():
    """The linear tolerance reads a condition estimate from the operator."""
    matrix = jnp.diag(jnp.asarray([1.0, 4.0]))
    estimate = support_read._krylov_condition_estimate(lambda value: matrix @ value, 2)
    np.testing.assert_allclose(estimate, 4.0, rtol=1e-12)


def test_bank_jvp_composes_flux_and_level_derivatives(monkeypatch):
    """The bank carries all three booking inputs into the moment residual."""

    class Reading(NamedTuple):
        field_coefficients: jax.Array
        axis_flux: jax.Array
        boundary_flux: jax.Array
        axis: jax.Array
        boundary: jax.Array

    class Field:
        def __init__(self, jets):
            self.jets = jets

        def sampled_coefficient(self, geometry):
            del geometry
            return jnp.asarray([[self.jets[0, 0], 0.0, 0.0, 0.0, 0.0, 0.0]])

        def value(self, point):
            return self.jets[0, 0] * point[0]

    def book(psi, read, geometry, profile, quadrature):
        del geometry, profile
        value = (
            psi[0, 0]
            + 2.0 * read.field_coefficients[0, 0]
            + 3.0 * read.axis_flux
            + 4.0 * read.boundary_flux
        )
        normalization = (
            1.0
            if quadrature is None or quadrature.target_current is None
            else quadrature.target_current / value
        )
        return current.Booking(
            jnp.asarray([value * normalization]),
            jnp.zeros(1),
            jnp.zeros(1),
            value,
            jnp.asarray(normalization),
            jnp.asarray(True),
        )

    def book_tangent(
        psi,
        psi_tangent,
        read,
        field_tangent,
        axis_tangent,
        boundary_tangent,
        geometry,
        profile,
        density_gradient,
        quadrature,
        booked,
    ):
        del psi, read, geometry, profile, density_gradient
        value = (
            psi_tangent[0, 0]
            + 2.0 * field_tangent[0, 0]
            + 3.0 * axis_tangent
            + 4.0 * boundary_tangent
        )
        correction = (
            0.0
            if quadrature is None or quadrature.target_current is None
            else -value * booked.cell_current[0] / booked.raw_current
        )
        return clip_quadrature.ClippedCurrentMoments(
            jnp.asarray([booked.normalization * value + correction]),
            jnp.zeros(1),
            jnp.zeros(1),
        )

    monkeypatch.setattr(
        support_read.fixed_points, "field_from_jets", lambda jets, *args: Field(jets)
    )
    monkeypatch.setattr(current, "book", book)
    monkeypatch.setattr(current, "book_tangent", book_tangent)
    rows = jnp.zeros((1, 1, 3, 6)).at[0, 0, 0, 0].set(2.0)
    operands = SimpleNamespace(
        rows=rows,
        exterior=jnp.zeros((1, 6)),
        sample_index=jnp.zeros((1, 1), dtype=jnp.int32),
        geometry=object(),
        mesh=object(),
    )
    reading = Reading(
        jnp.zeros((1, 6)), 0.0, 0.0, jnp.asarray([1.0, 0.0]), jnp.asarray([2.0, 0.0])
    )
    support = SimpleNamespace(reading=reading, valid=jnp.asarray(True))
    moments = (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
    direction = (jnp.ones(1), jnp.zeros(1), jnp.zeros(1))
    applied = support_read.step_jacobian_vector(
        operands, moments, direction, support, None, None, None
    )
    step = 1e-5
    plus = tuple(a + step * d for a, d in zip(moments, direction, strict=True))
    minus = tuple(a - step * d for a, d in zip(moments, direction, strict=True))
    high = support_read.step_residual(operands, moments, plus, support, None, None)
    low = support_read.step_residual(operands, moments, minus, support, None, None)
    difference = (high.residual[0] - low.residual[0]) / (2 * step)
    np.testing.assert_allclose(applied[0], difference, rtol=1e-10)
    np.testing.assert_allclose(applied[0], -27.0, rtol=1e-10)
    matrix = support_read.step_jacobian(
        operands, moments, support, None, None, SimpleNamespace(target_current=None)
    )
    np.testing.assert_allclose(
        matrix @ jnp.concatenate(direction), jnp.concatenate(applied)
    )
    constrained = current.Quadrature(target_current=5.0)
    nonzero = (jnp.ones(1), jnp.zeros(1), jnp.zeros(1))
    constrained_jvp = support_read.step_jacobian_vector(
        operands, nonzero, direction, support, None, None, constrained
    )
    constrained_matrix = support_read.step_jacobian(
        operands, nonzero, support, None, None, constrained
    )
    np.testing.assert_allclose(
        constrained_matrix @ jnp.concatenate(direction),
        jnp.concatenate(constrained_jvp),
    )


def test_diagonal_equilibration_reduces_scale_disparity():
    matrix = jnp.asarray([[1e-6, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e6]])
    moments = (jnp.asarray([1.0]), jnp.asarray([1.0]), jnp.asarray([1.0]))
    booked = (*moments,)
    balanced, row, column = support_read._equilibrate(matrix, moments, booked)
    np.testing.assert_allclose(balanced, matrix * column[None, :] / row[:, None])
    assert float(jnp.linalg.cond(balanced)) < float(jnp.linalg.cond(matrix))


def test_dense_newton_step_solves_scaled_residual(monkeypatch):
    """The delivered step applies the scaled LU update to all moment blocks."""
    matrix = jnp.asarray([[1e-6, 2e-7, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1e6]])
    moments = tuple(jnp.asarray([1.0]) for _ in range(3))
    residual = tuple(jnp.asarray([value]) for value in (1e-6, 1.0, 1e6))
    booked = current.Booking(*moments, jnp.asarray(1.0), jnp.asarray(1.0), True)
    monkeypatch.setattr(
        support_read,
        "step_residual",
        lambda *args: support_read.StepResidual(residual, booked, None, True),
    )
    monkeypatch.setattr(support_read, "step_jacobian", lambda *args: matrix)
    step = support_read.newton_step(None, moments, None, None, None, None)
    actual = jnp.concatenate(step.update)
    expected = np.linalg.solve(np.asarray(matrix), -np.asarray([1e-6, 1.0, 1e6]))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert float(step.linear_residual) < 1e-12
    assert float(step.condition_after) < float(step.condition_before)
    assert bool(step.converged)


def test_terminal_verdict_requires_fresh_read():
    assert _terminal_verdict(8e-13, 9e-13, True)
    assert not _terminal_verdict(8e-13, 0.0303684, True)
    assert not _terminal_verdict(8e-13, 9e-13, False)
    assert not _terminal_verdict(8e-13, np.nan, True)


def test_forward_route_update_comparison_refuses_regressions():
    assert _assert_update_agreement([1.0, 2.0], [1.0, 2.0]) == 0.0
    with np.testing.assert_raises(AssertionError):
        _assert_update_agreement([1.01, 2.0], [1.0, 2.0])
    with np.testing.assert_raises(AssertionError):
        _assert_update_agreement([np.nan, 2.0], [1.0, 2.0])


def test_route_comparison_uses_production_update(monkeypatch):
    """A distinct route residual must reach the final update assertion."""
    from scripts.prototypes.fast_map_production import program, solver
    from scripts.prototypes.lean_map import jets, quadrature, values

    geometry = SimpleNamespace(
        vertices=jnp.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
        vertex_count=jnp.asarray([3]),
    )
    operands = SimpleNamespace(
        geometry=geometry,
        rows=jnp.zeros(1),
        exterior=jnp.zeros(1),
        nodes=jnp.zeros(1),
        sample_index=jnp.zeros(1),
        mesh=jnp.zeros(1),
    )
    moments = tuple(jnp.zeros(1) for _ in range(3))
    expected_route = jnp.asarray([[2.0, 3.0, 4.0]])
    observed = {}

    monkeypatch.setattr(
        measure,
        "jax",
        SimpleNamespace(
            jit=lambda function: function, block_until_ready=lambda _: None
        ),
    )
    monkeypatch.setattr(
        measure, "emit", lambda name, **row: observed.update({name: row})
    )
    monkeypatch.setattr(quadrature, "triangle_rule", lambda _: jnp.zeros(1))
    monkeypatch.setattr(
        quadrature, "quadratic_fit", lambda *_: (jnp.zeros(1), jnp.zeros(1))
    )
    monkeypatch.setattr(
        values, "build_values", lambda *_args, **_kwargs: (jnp.zeros(1), {})
    )
    monkeypatch.setattr(jets, "JetMap", lambda *_: jnp.zeros(1))
    monkeypatch.setattr(
        program,
        "map_moments",
        lambda state, *_args: SimpleNamespace(
            moment_residual=state - expected_route, booking_valid=jnp.asarray(True)
        ),
    )
    monkeypatch.setattr(
        solver, "operand_block", lambda *_: lambda directions, *args: directions
    )
    monkeypatch.setattr(
        support_read,
        "newton_step",
        lambda *_: SimpleNamespace(
            update=tuple(jnp.asarray([value]) for value in (1.0, 2.0, 3.0)),
            linear_residual=jnp.asarray(0.0),
        ),
    )
    monkeypatch.setattr(
        support_read,
        "step_residual",
        lambda *_: SimpleNamespace(
            residual=tuple(jnp.asarray([value]) for value in (1.0, 2.0, 3.0)),
            valid=jnp.asarray(True),
        ),
    )
    analytic = SimpleNamespace(evaluate=lambda _: SimpleNamespace(hessian=jnp.eye(2)))
    with pytest.raises(AssertionError, match="forward-route update difference"):
        measure.compare_route(
            operands,
            jnp.zeros(1),
            jnp.zeros((1, 2)),
            None,
            None,
            None,
            current.Quadrature(),
            moments,
            None,
            analytic,
        )
    assert observed["ROUTE_COMPARISON"]["relative_update"] > 0.1


def test_route_comparison_cli_reaches_comparator(monkeypatch):
    """The comparison option must dispatch before the tangent-only path."""

    class ReachedComparison(Exception):
        pass

    geometry = SimpleNamespace(pitch=jnp.ones(1))
    operands = SimpleNamespace(geometry=geometry, rows=jnp.zeros((1, 1)), bucket=1)
    moments = tuple(jnp.zeros(1) for _ in range(3))
    reading = SimpleNamespace(
        valid=jnp.asarray(True),
        qualified=jnp.asarray(True),
        membership=jnp.ones(1),
        normal_form_cells=jnp.zeros(1, dtype=bool),
    )
    support = SimpleNamespace(valid=jnp.asarray(True), reading=reading)
    monkeypatch.setattr(sys, "argv", [measure.__file__, "--compare-route"])
    monkeypatch.setattr(
        measure.subprocess, "check_output", lambda *_args, **_kwargs: "test-revision"
    )
    monkeypatch.setattr(
        measure,
        "jax",
        SimpleNamespace(
            jit=lambda function: function,
            block_until_ready=lambda _: None,
            default_backend=lambda: "cpu",
            devices=lambda: [],
        ),
    )
    monkeypatch.setattr(measure, "emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        measure,
        "machine",
        lambda kind: (
            operands,
            None,
            jnp.zeros((1, 2)),
            None,
            None,
            None,
            None,
            moments,
            None,
        ),
    )
    monkeypatch.setattr(support_read, "read", lambda *_: support)

    def reached(*_args):
        raise ReachedComparison

    monkeypatch.setattr(measure, "compare_route", reached)
    with pytest.raises(ReachedComparison):
        measure.main()
