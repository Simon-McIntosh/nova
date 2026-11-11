"""Closed-form support-boundary and bank Jacobian checks."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import re
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

    def book_channels(
        psi,
        directions,
        read,
        fields,
        axes,
        boundaries,
        geometry,
        profile,
        gradient,
        quadrature,
        booked,
        edge_indices,
    ):
        del edge_indices
        response = jax.vmap(
            lambda direction, field, axis, boundary: book_tangent(
                psi,
                direction,
                read,
                field,
                axis,
                boundary,
                geometry,
                profile,
                gradient,
                quadrature,
                booked,
            )
        )(directions, fields, axes, boundaries)
        return jnp.stack(tuple(response), axis=1).transpose(2, 1, 0)

    monkeypatch.setattr(current, "book_tangent_channels", book_channels)
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
        operands, moments, support, None, None, current.Quadrature()
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
    """The equilibrated Krylov update solves all physical moment blocks."""
    matrix = jnp.asarray([[1e-6, 2e-7, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1e6]])
    moments = tuple(jnp.asarray([1.0]) for _ in range(3))
    residual = tuple(jnp.asarray([value]) for value in (1e-6, 1.0, 1e6))
    booked = current.Booking(*moments, jnp.asarray(1.0), jnp.asarray(1.0), True)
    monkeypatch.setattr(
        support_read,
        "step_residual",
        lambda *args: support_read.StepResidual(residual, booked, None, True),
    )
    monkeypatch.setattr(support_read, "step_jacobian", lambda *args, **kwargs: matrix)
    step = support_read.newton_step(None, moments, None, None, None, None)
    actual = jnp.concatenate(step.update)
    expected = np.linalg.solve(np.asarray(matrix), -np.asarray([1e-6, 1.0, 1e6]))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert float(step.linear_residual) < 1e-12
    assert float(step.condition_after) < float(step.condition_before)
    assert bool(step.converged)


@pytest.fixture(scope="module")
def diverted_linear_states():
    from scripts.prototypes.bank_solve_profile import fixture

    batch = fixture.build(1)
    assert batch.states[0].shape == (1, 550)

    @jax.jit
    def converge(bank, cache, states, seeds, previous):
        return support_read.solve_batch(
            bank,
            cache,
            states,
            seeds,
            previous,
            batch.convention,
            batch.policy,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
        )

    result = jax.block_until_ready(
        converge(
            batch.operands,
            batch.cache,
            batch.states,
            batch.seeds_batch,
            batch.previous_batch,
        )
    )
    assert bool(jnp.all(result.converged))
    assert float(jnp.max(result.residual)) <= 1e-10
    return batch, {"trial": batch.states, "converged": result.moments}


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("state", ["trial", "converged"])
def test_diverted_physical_update_matches_double_reference(
    diverted_linear_states, state
):
    """The Krylov coordinate bound must also preserve the physical update."""
    from scripts.prototypes.bank_solve_profile import fixture

    batch, states = diverted_linear_states
    moments = tuple(value[0] for value in states[state])
    support = jax.jit(
        lambda bank, cache, values: support_read.read(
            bank,
            cache,
            values,
            batch.seeds,
            batch.roots,
            batch.convention,
            batch.policy,
        )
    )(batch.operands, batch.cache, moments)

    @jax.jit
    def update(bank, values, reading):
        return support_read.newton_step(
            bank,
            values,
            reading,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )

    step = jax.block_until_ready(update(batch.operands, moments, support))
    residual = support_read.step_residual(
        batch.operands,
        moments,
        moments,
        support,
        batch.profile,
        batch.quadrature,
    )
    right = -jnp.concatenate(residual.residual)
    reference = jnp.linalg.solve(step.factor.matrix, right)
    assert reference.dtype == jnp.float64 and jax.config.jax_enable_x64
    error = float(
        jnp.linalg.norm(jnp.concatenate(step.update) - reference)
        / jnp.linalg.norm(reference)
    )
    print(
        "PHYSICAL_UPDATE",
        state,
        "error",
        error,
        "condition",
        float(step.krylov_condition),
        "iterations",
        int(step.krylov_iterations),
        "relative_residual",
        float(step.krylov_residual),
        flush=True,
    )
    assert error <= 1e-8
    assert bool(step.converged)
    assert float(step.krylov_condition * step.krylov_residual) <= 1e-8


def test_block_gmres_zero_right_and_iteration_exhaustion():
    matrix = jnp.asarray([[2.0, 0.3, 0.1], [0.2, 3.0, 0.1], [0.0, 0.1, 4.0]])
    inverse, indices = support_read._block_preconditioner(matrix)
    solution, condition, residual, steps = support_read._block_gmres(
        matrix,
        jnp.zeros(3),
        inverse,
        indices,
    )
    np.testing.assert_array_equal(solution, 0.0)
    assert float(residual) == 0 and int(steps) == 0 and np.isfinite(condition)
    _, condition, residual, steps = support_read._block_gmres(
        matrix,
        jnp.ones(3),
        jnp.eye(3),
        jnp.arange(3),
        max_iterations=1,
    )
    assert int(steps) == 1 and float(condition * residual) > 1e-8


def test_block_gmres_breakdown_leaves_unwritten_columns_out_of_the_update():
    """A cycle that breaks down early must not solve over empty columns.

    Block-diagonal cells make the preconditioned operator the identity, so the
    Arnoldi run breaks down on its first step, with a ``width`` far larger than
    the single column written. The reference solve is independent of the
    Krylov path.
    """
    block = jnp.asarray([[2.0, 0.3, 0.1], [0.2, 3.0, 0.1], [0.0, 0.1, 4.0]])
    cells = 4
    matrix = jnp.zeros((3 * cells, 3 * cells), jnp.float64)
    for cell in range(cells):
        for row in range(3):
            for column in range(3):
                matrix = matrix.at[cell + cells * row, cell + cells * column].set(
                    block[row, column] * (cell + 1)
                )
    right = jnp.arange(1.0, 3 * cells + 1.0, dtype=jnp.float64)[::-1]
    inverse, indices = support_read._block_preconditioner(matrix)
    solution, condition, residual, steps = support_read._block_gmres(
        matrix, right, inverse, indices
    )
    reference = jnp.linalg.solve(matrix, right)
    error = float(jnp.linalg.norm(solution - reference) / jnp.linalg.norm(reference))
    print(
        "BREAKDOWN_UPDATE",
        "steps",
        int(steps),
        "error",
        error,
        "relative_residual",
        float(residual),
        flush=True,
    )
    assert int(steps) == 1 and 3 * cells > int(steps)
    assert error <= 1e-10


def _arnoldi_least_squares_reference(matrix, right, columns):
    """One restarted Arnoldi cycle in float64 numpy, independent of the Jax path."""
    values = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(right, dtype=np.float64)
    basis = np.zeros((target.size, columns + 1))
    hessenberg = np.zeros((columns + 1, columns))
    basis[:, 0] = target / np.linalg.norm(target)
    for column in range(columns):
        candidate = values @ basis[:, column]
        for _ in range(2):
            projection = basis[:, : column + 1].T @ candidate
            candidate = candidate - basis[:, : column + 1] @ projection
            hessenberg[: column + 1, column] += projection
        hessenberg[column + 1, column] = np.linalg.norm(candidate)
        basis[:, column + 1] = candidate / hessenberg[column + 1, column]
    right_hand = np.zeros(columns + 1)
    right_hand[0] = np.linalg.norm(target)
    coefficients = np.linalg.lstsq(hessenberg, right_hand, rcond=None)[0]
    return basis[:, :columns] @ coefficients


def _restart_width(size):
    """The Arnoldi restart width ``_block_gmres`` sets for a system of ``size``.

    The production function caps one cycle at ``min(20, right.size)``. Reading
    that literal from its own source keeps this test tracking the module
    rather than duplicating the number, and fails loudly if the form changes.
    """
    pattern = re.compile(r"width = min\((\d+), right\.size\)")
    match = pattern.search(inspect.getsource(support_read._block_gmres))
    assert match is not None, (
        "the restart width is no longer a literal min(<int>, right.size) in "
        "_block_gmres; update this derivation before trusting the cap"
    )
    return min(int(match.group(1)), size)


def test_block_gmres_cap_mid_cycle_leaves_unwritten_columns_out_of_the_update():
    """A cap that cuts a cycle part-way must stop at the cap.

    One cycle writes the restart width, so a cap at a quarter of that width
    leaves most of the columns unwritten and those must stay out of the
    least-squares. The update is compared with an independent float64 Arnoldi
    cycle over the written columns.
    """
    size = 24
    matrix = jnp.eye(size, dtype=jnp.float64)
    for row in range(size - 1):
        matrix = matrix.at[row + 1, row].set(0.35)
        matrix = matrix.at[row, row + 1].set(0.25)
    pattern = jnp.asarray([2.0, -1.0, 3.0, 1.5, -1.5], dtype=jnp.float64)
    right = jnp.concatenate(
        (pattern, jnp.full(size - pattern.size, 0.4, dtype=jnp.float64))
    )
    inverse, indices = support_read._block_preconditioner(matrix)
    width = _restart_width(size)
    cap = max(1, width // 4)
    # The reference below applies the raw matrix; it agrees with the
    # preconditioned run only because this matrix's applied block inverse
    # is exactly the identity, leaving both the matrix and right side unchanged.
    applied = np.stack(
        [
            np.asarray(support_read._apply_block_inverse(inverse, indices, vector))
            for vector in jnp.eye(size, dtype=jnp.float64)
        ]
    )
    np.testing.assert_array_equal(applied, np.eye(size))
    solution, condition, residual, steps = support_read._block_gmres(
        matrix, right, inverse, indices, max_iterations=cap
    )
    reference = _arnoldi_least_squares_reference(matrix, right, cap)
    error = float(jnp.linalg.norm(solution - reference) / jnp.linalg.norm(reference))
    print(
        "CAP_UPDATE",
        "width",
        width,
        "cap",
        cap,
        "steps",
        int(steps),
        "error",
        error,
        "relative_residual",
        float(residual),
        flush=True,
    )
    assert int(steps) == cap and cap % width != 0
    assert error <= 1e-10


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


@pytest.mark.slow
@pytest.mark.parametrize("state", ["trial", "converged"])
def test_booking_jacobian_matches_fresh_residual_difference(
    diverted_linear_states, state
):
    """Include moving saddle geometry when differentiating a fresh support read."""
    from scripts.prototypes.bank_solve_profile import fixture

    assert jax.config.jax_enable_x64
    batch, states = diverted_linear_states
    moments = tuple(value[0] for value in states[state])

    @jax.jit
    def fresh(bank, cache, values):
        reading = support_read.read(
            bank,
            cache,
            values,
            batch.seeds,
            batch.roots,
            batch.convention,
            batch.policy,
        )
        return support_read.step_residual(
            bank, values, values, reading, batch.profile, batch.quadrature
        ).residual, reading

    _, reading = jax.block_until_ready(fresh(batch.operands, batch.cache, moments))
    matrix = jax.jit(
        lambda b, m, s: support_read.step_jacobian(
            b,
            m,
            s,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )
    )(batch.operands, moments, reading)
    cells = moments[0].size
    shape = jnp.sin(1.7 * jnp.arange(cells)) + jnp.cos(2.3 * jnp.arange(cells))
    direction = tuple(
        jnp.max(jnp.abs(value)) * jnp.roll(shape, shift) / 4
        for value, shift in zip(moments, (0, 17, 31), strict=True)
    )
    applied = matrix @ jnp.concatenate(direction)
    errors = []
    for epsilon in (1e-8, 1e-9, 1e-10):
        plus = tuple(v + epsilon * d for v, d in zip(moments, direction, strict=True))
        minus = tuple(v - epsilon * d for v, d in zip(moments, direction, strict=True))
        high, high_read = fresh(batch.operands, batch.cache, plus)
        low, low_read = fresh(batch.operands, batch.cache, minus)
        difference = jnp.concatenate(
            tuple(
                (high_value - low_value) / (2 * epsilon)
                for high_value, low_value in zip(high, low, strict=True)
            )
        )
        assert bool(high_read.valid & low_read.valid)
        scale = jnp.linalg.norm(difference)
        assert float(scale) > 0
        errors.append(float(jnp.linalg.norm(applied - difference) / scale))
    print("FRESH_JACOBIAN_DIFFERENCE", state, errors, flush=True)
    assert np.isfinite(errors).all() and min(errors) <= 1e-6


def test_local_booking_reuses_constant_field_boundary_velocity(monkeypatch):
    """The level channel shares a ring velocity and keeps its density derivative."""
    from tests.test_current_book_vectorised import _cells

    psi, reading, geometry, profile = _cells(1, slope=0.2)
    operands = SimpleNamespace(
        geometry=geometry, sample_index=jnp.arange(psi.size).reshape(psi.shape)
    )
    jets = jnp.zeros((psi.size, 6)).at[:, 0].set(psi.reshape(-1))
    monkeypatch.setattr(support_read, "nodal_jets", lambda *_: jets)
    channels_seen = []
    original = current.book_tangent_channels

    def observed(*args):
        channels_seen.append(args[-1].shape[0])
        return original(*args)

    monkeypatch.setattr(current, "book_tangent_channels", observed)
    quadrature = current.Quadrature(segments=8)
    booked = current.book(psi, reading, geometry, profile, quadrature)
    actual = jax.jit(
        lambda: support_read._local_booking_sensitivities(
            operands,
            None,
            SimpleNamespace(reading=reading),
            profile,
            lambda r, f: jnp.full_like(r, 0.2),
            quadrature,
            booked,
        )
    )()
    indices = jnp.arange(14)
    flux = jnp.zeros((14, 1, 6)).at[:6, 0, :].set(jnp.eye(6))
    field = jnp.zeros((14, 1, 6)).at[6:12, 0, :].set(jnp.eye(6))
    expected = jax.jit(
        lambda: original(
            psi,
            flux,
            reading,
            field,
            (indices == 12).astype(psi.dtype),
            (indices == 13).astype(psi.dtype),
            geometry,
            profile,
            lambda r, f: jnp.full_like(r, 0.2),
            quadrature,
            booked,
            jnp.asarray([6, 7, 8, 9, 10, 11, 13]),
        )
    )()
    relative = float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))
    print("LOCAL_CHANNEL_PARITY", relative, "BOUNDARY_CHANNELS", channels_seen)
    assert relative <= 1e-12
    assert channels_seen and all(number == 6 for number in channels_seen)


@pytest.mark.parametrize(
    "requested_width, expected_width",
    [(None, 128), (16, 16), (64, 64)],
    ids=["default", "narrow-explicit", "wide-explicit"],
)
def test_local_booking_uses_wide_default_cell_tile(
    monkeypatch, requested_width, expected_width
):
    """The default tangent tile stays wide while explicit memory bounds survive."""
    assert jax.config.jax_enable_x64
    operands = SimpleNamespace(
        sample_index=jnp.arange(6).reshape(1, 6), geometry=object()
    )
    monkeypatch.setattr(
        support_read, "nodal_jets", lambda *_: jnp.zeros((6, 6), dtype=jnp.float64)
    )
    observed_widths = []

    def capture(*arguments):
        observed_widths.append(arguments[9].cut_batch_width)
        return jnp.zeros((1, 3, 14), dtype=jnp.float64)

    monkeypatch.setattr(current, "book_tangent_channels", capture)
    rule = (
        current.Quadrature()
        if requested_width is None
        else current.Quadrature(cut_batch_width=requested_width)
    )
    support_read._local_booking_sensitivities(
        operands, None, SimpleNamespace(reading=object()), None, None, rule, None
    )
    assert observed_widths == [expected_width]


def test_booking_matrix_parity_guard_refuses_perturbed_matrix():
    """The timing run must reject an altered matrix through its real comparison."""
    from scripts.prototypes.bank_booking_cost.measure import compare_matrices

    reference = np.arange(1.0, 10.0, dtype=np.float64).reshape(3, 3)
    assert compare_matrices(reference, reference, 1) == 0.0
    with pytest.raises(AssertionError):
        compare_matrices(1.001 * reference, reference, 1)
