"""Bounded solve budgets share one traced body and retain padded receipts."""

from __future__ import annotations

from functools import partial
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_point, reduced_newton
from nova.equilibrium.forward_operator import CellCurrentMoments, ForwardFluxOperator
from nova.jax.config import Precision, configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True


class _CandidateCurrentOperator(ForwardFluxOperator):
    """A normalized two-cell map whose current read varies with the candidate."""

    def __init__(self):
        self.use_linear_moments = False

    def cell_current_moments(self, psi, requested_class=None):
        current = jnp.stack((1.0 + 0.25 * psi[0], jnp.ones_like(psi[1])))
        zero = jnp.zeros_like(current)
        return CellCurrentMoments(current, zero, zero)

    def current_moment_image(self, moments):
        return moments.cell_current

    def _frozen_topology_partition(self, *_args, **_kwargs):
        raise AssertionError(
            "normalized-current read was frozen across Newton candidates"
        )


@pytest.mark.parametrize("target", [2.0, 3.0])
def test_normalized_current_trip_keeps_candidate_reads_live(target):
    operator = _CandidateCurrentOperator()
    initial = jnp.full(2, target / 2.0, dtype=jnp.float64)
    external = jnp.zeros_like(initial)
    shadow = jnp.zeros_like(initial, dtype=bool)
    mapped = operator.traced_flux_map_with_shadow(target_current=target)
    bound = fixed_point._bind_traced_map_arguments(
        mapped, (external, operator, jnp.asarray(target))
    )

    def solve(state):
        return fixed_point.newton_krylov(
            lambda candidate: bound(candidate, shadow),
            state,
            newton_steps=8,
            gmres_iterations=2,
            warmup=0,
            shadow_mask_fn=lambda _candidate: shadow,
            promoted_shadow_mask_fn=lambda _candidate, _previous: shadow,
            shadowed_map_fn=bound,
            active_set_steps=2,
            convergence_tolerance=1e-14,
            precision=Precision.DOUBLE,
        )

    result = jax.jit(solve)(initial)
    linear = 2.0 - 0.25 * target
    first = (-linear + np.sqrt(linear**2 + target)) / 0.5
    expected = np.asarray((first, target - first))
    assert bool(result.converged)
    np.testing.assert_allclose(result.state, expected, rtol=0.0, atol=1e-12)
    # A read retained at the entry state produces a different fixed point.
    assert np.max(np.abs(np.asarray(bound(initial, shadow)) - expected)) > 1e-3


def _active_solve(initial, *, trips=4, steps=2):
    def mask(state):
        return state >= 0.5

    def mapped(_state, shadow):
        return jnp.where(shadow, 2.0, 1.0)

    return fixed_point.newton_krylov(
        lambda state: mapped(state, mask(state)),
        initial,
        newton_steps=steps,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda state, _previous: mask(state),
        shadowed_map_fn=mapped,
        active_set_steps=trips,
        precision=Precision.DOUBLE,
    )


def _barrier_count(function, *arguments):
    lowered = jax.jit(function).lower(*arguments)
    return str(lowered.compiler_ir(dialect="stablehlo")).count(
        "stablehlo.optimization_barrier"
    )


def _optimized_read_body_count(function, *arguments):
    """Count the distinctive read operation after the compiler has inlined calls."""
    compiled = jax.jit(function).lower(*arguments).compile()
    return len(re.findall(r"^\s*(?:ROOT )?%\S+ = .* sine\(", compiled.as_text(), re.M))


def test_read_body_counter_rejects_lowered_call_sharing():
    """A shared lowered callee must not hide its optimized copies."""

    @jax.jit
    def read(state):
        return jnp.sin(state)

    def repeated(first, second, third):
        return read(first), read(second), read(third)

    def shared(states):
        return jax.lax.map(read, states)

    states = jnp.arange(51, dtype=jnp.float64).reshape(3, 17) / 51.0
    lowered = jax.jit(repeated).lower(*states)
    assert str(lowered.compiler_ir(dialect="stablehlo")).count("stablehlo.sine") == 1
    assert _optimized_read_body_count(read, states[0]) == 1
    assert _optimized_read_body_count(repeated, *states) == 3
    assert _optimized_read_body_count(shared, states) == 1
    np.testing.assert_array_equal(
        np.stack(jax.jit(repeated)(*states)), jax.jit(shared)(states)
    )


@pytest.mark.parametrize(
    "helper", ["_backtracked_promotion", "_rebuilt_model_promotion"]
)
def test_initial_and_retried_promotions_share_one_optimized_body(monkeypatch, helper):
    original = getattr(fixed_point, helper)

    def marked(*args, **kwargs):
        result = original(*args, **kwargs)
        return result._replace(state=jnp.sin(result.state))

    monkeypatch.setattr(fixed_point, helper, marked)

    def solve(initial):
        return fixed_point.newton_krylov(
            lambda state: jnp.tanh(state) + 0.5,
            initial,
            newton_steps=3,
            gmres_iterations=2,
            warmup=0,
            precision=Precision.DOUBLE,
        )

    initial = jnp.linspace(0.1, 0.9, 17, dtype=jnp.float64)
    assert _optimized_read_body_count(jnp.sin, initial) == 1
    assert _optimized_read_body_count(solve, initial) == 1


@pytest.mark.parametrize("use_incumbent", [False, True])
def test_shadow_selection_lowers_one_live_map(use_incumbent):
    initial = jnp.asarray([0.25, 0.75], dtype=jnp.float64)
    previous = jnp.asarray([True, False])

    def induced(candidate, _previous):
        return candidate >= 0.5

    def mapped(candidate, shadow):
        value = jax.lax.optimization_barrier(candidate)
        return jnp.where(shadow, value, 0.25 * value + 1.0)

    def selected(candidate, incumbent):
        return fixed_point._map_on_selected_shadow(
            candidate, previous, incumbent, induced, mapped
        )

    def reference(candidate, incumbent):
        return fixed_point._acceptance_map_on_selected_partition(
            candidate,
            previous,
            incumbent,
            lambda value: mapped(value, previous),
            induced,
            mapped,
        )

    assert _barrier_count(jax.lax.optimization_barrier, initial) == 1
    assert _barrier_count(selected, initial, jnp.asarray(use_incumbent)) == 1
    for evaluate in (
        lambda fn: fn(initial, use_incumbent),
        lambda fn: jax.jacfwd(fn)(initial, use_incumbent),
    ):
        np.testing.assert_array_equal(evaluate(selected), evaluate(reference))


def test_backtracking_incumbent_shares_the_candidate_read_body():
    def mapped(candidate):
        return jnp.tanh(jax.lax.optimization_barrier(candidate)) + 0.5

    def scores(state):
        return fixed_point._backtracking_scores(
            mapped,
            lambda candidate: 0.25 * candidate + 1.0,
            state,
            jnp.ones_like(state),
            jnp.asarray(1.0),
            False,
        )

    state = jnp.asarray([0.25, 0.75], dtype=jnp.float64)
    assert _barrier_count(jax.lax.optimization_barrier, state) == 1
    assert _barrier_count(scores, state) == 1
    observed = scores(state)
    np.testing.assert_array_equal(
        observed.incumbent_residual,
        fixed_point._relative_residual(mapped(state), state),
    )


@pytest.mark.parametrize("trips", [2, 5])
def test_active_trip_body_is_lowered_once(monkeypatch, trips):
    original = fixed_point._newton_krylov_inner

    def marked_inner(*args, **kwargs):
        result, globalization = original(*args, **kwargs)
        return result._replace(
            state=jax.lax.optimization_barrier(result.state)
        ), globalization

    monkeypatch.setattr(fixed_point, "_newton_krylov_inner", marked_inner)
    initial = jnp.zeros(1, dtype=jnp.float64)
    assert _barrier_count(jax.lax.optimization_barrier, initial) == 1
    count = _barrier_count(partial(_active_solve, trips=trips), initial)
    assert count == 1, f"the lowered solve contains {count} copies of its trip body"


@pytest.mark.parametrize("trips,steps", [(4, 2), (7, 5)])
def test_active_telemetry_has_fixed_capacity_after_early_convergence(trips, steps):
    result = jax.jit(partial(_active_solve, trips=trips, steps=steps))(
        jnp.zeros(1, dtype=jnp.float64)
    )
    assert int(result.active_set_iterations) == 2
    assert bool(result.converged)
    np.testing.assert_array_equal(result.state, [2.0])
    assert result.active_set_residuals.shape == (trips,)
    assert result.inner_iteration_residuals_before.shape == (steps,)
    assert result.trace.shape == (steps * 3,)
    np.testing.assert_array_equal(result.active_set_mask_differences[:2], [1, 0])
    np.testing.assert_array_equal(result.active_set_mask_differences[2:], -1)
    assert np.isnan(np.asarray(result.active_set_residuals)[2:]).all()
    np.testing.assert_array_equal(result.inner_iteration_decisions[1:], -1)


def _reduced_solver(*, trips, steps):
    def scores(value, _shadow, _base, **_kwargs):
        residual = 1.0 - value
        norm = jnp.max(jnp.abs(residual))
        return reduced_newton.ReducedScores(residual, norm, norm, norm)

    def boundary(value, shadow, _base, **_kwargs):
        state = jax.lax.optimization_barrier(value)
        observed = jnp.max(jnp.abs(1.0 - state))
        return state, shadow, jnp.asarray(0, jnp.int32), observed, value, observed

    kernels = {
        "initial_gather": lambda state, **_kwargs: state,
        "jacobian": lambda value, *_args, **_kwargs: jnp.eye(value.size),
        "step_scores": scores,
        "direction": lambda _jacobian, residual: residual,
        "boundary": boundary,
    }
    return reduced_newton._compiled_slice_solver(
        kernels, tolerance=1e-8, newton_steps=steps, active_set_steps=trips
    )


@pytest.mark.parametrize("steps", [2, 5])
def test_newton_step_body_is_lowered_once(monkeypatch, steps):
    original = fixed_point._qualified_krylov_step

    def marked_step(*args, **kwargs):
        result = original(*args, **kwargs)
        return result._replace(step=jax.lax.optimization_barrier(result.step))

    monkeypatch.setattr(fixed_point, "_qualified_krylov_step", marked_step)

    def solve(initial):
        return fixed_point.newton_krylov(
            lambda state: 0.5 * state + 1.0,
            initial,
            newton_steps=steps,
            warmup=0,
            gmres_iterations=1,
            precision=Precision.DOUBLE,
        )

    initial = jnp.zeros(1, dtype=jnp.float64)
    assert _barrier_count(jax.lax.optimization_barrier, initial) == 1
    assert _barrier_count(solve, initial) == 1


@pytest.mark.parametrize("trips,steps", [(3, 5), (6, 9)])
def test_reduced_trip_body_and_newton_budget_are_scans(trips, steps):
    solve = _reduced_solver(trips=trips, steps=steps)
    arguments = (jnp.zeros(1, jnp.float64), jnp.zeros(1, bool), jnp.zeros(1))
    assert _barrier_count(solve, *arguments) == 1
    program = jax.make_jaxpr(solve)(*arguments)

    def scan_lengths(value):
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                if equation.primitive.name == "scan":
                    assert equation.params["unroll"] == 1
                    yield equation.params["length"]
                for parameter in equation.params.values():
                    yield from scan_lengths(parameter)
        elif hasattr(value, "jaxpr"):
            yield from scan_lengths(value.jaxpr)
        elif isinstance(value, tuple | list):
            for item in value:
                yield from scan_lengths(item)

    lengths = list(scan_lengths(program))
    assert trips in lengths
    assert steps in lengths
    result = jax.jit(solve)(*arguments)
    assert int(result[7]) == 1
    for slots in result[10:16]:
        assert slots.shape == (trips,)
    np.testing.assert_array_equal(result[11], [0] + [-1] * (trips - 1))
    assert np.isnan(np.asarray(result[10])[1:]).all()
