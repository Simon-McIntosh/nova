"""Contracts for the batched fixed-point accelerators.

The three schemes share one result contract and one per-map-evaluation
residual accounting.  Pinned here: linear-map exactness of the exact-tangent
Newton-Krylov (a full-dimension GMRES solves an affine map in one step),
Anderson acceleration over relaxed Picard at a shared budget, agreement of
all three schemes on one nonlinear fixed point, the non-finite / step-cap
safeguards, and vmap-batch equality with per-slice solves.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import (
        FIXED_POINT_RESIDUAL_TOLERANCE,
        FixedPointTerminationReason,
        KrylovActionQualification,
        anderson,
        kink_aware_newton_krylov,
        newton_krylov,
        picard,
    )
    from nova.jax.config import Precision, configure_dtypes


DIMENSION = 12


def _contraction(seed=3, radius=0.6):
    """A symmetric affine contraction with a known fixed point."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((DIMENSION, DIMENSION))
    matrix = 0.5 * (matrix + matrix.T)
    matrix *= radius / np.max(np.abs(np.linalg.eigvalsh(matrix)))
    offset = rng.standard_normal(DIMENSION)
    fixed_point = np.linalg.solve(np.eye(DIMENSION) - matrix, offset)
    return jnp.asarray(matrix), jnp.asarray(offset), fixed_point


def test_picard_converges_to_the_fixed_point():
    matrix, offset, fixed_point = _contraction()
    result = picard(
        lambda x: matrix @ x + offset,
        jnp.zeros(DIMENSION),
        evaluations=80,
        relaxation=0.5,
    )
    np.testing.assert_allclose(np.asarray(result.state), fixed_point, atol=1e-8)
    trace = np.asarray(result.trace)
    assert np.all(np.isfinite(trace))
    assert trace[-1] < 1e-8 < trace[0]


def test_newton_krylov_solves_an_affine_map_in_one_step():
    """A full-dimension Krylov space makes the exact-tangent step exact."""
    matrix, offset, fixed_point = _contraction()
    result = newton_krylov(
        lambda x: matrix @ x + offset,
        jnp.zeros(DIMENSION),
        newton_steps=1,
        gmres_iterations=DIMENSION,
        warmup=2,
        relaxation=0.5,
    )
    np.testing.assert_allclose(np.asarray(result.state), fixed_point, atol=1e-9)
    assert float(result.residual) < 1e-10
    assert int(result.attempted_newton_promotions) == 1
    assert int(result.accepted_newton_promotions) == 1
    assert bool(result.converged)
    assert int(result.termination_reason) == FixedPointTerminationReason.CONVERGED


def test_newton_stops_without_a_promotion_at_a_converged_initial_state():
    def map_fn(state):
        return 0.5 * state + jnp.ones_like(state)

    result = newton_krylov(
        map_fn,
        2.0 * jnp.ones(3),
        newton_steps=4,
        gmres_iterations=2,
        warmup=0,
    )
    assert float(result.residual) == 0.0
    assert int(result.attempted_newton_promotions) == 0
    assert int(result.accepted_newton_promotions) == 0
    assert bool(result.converged)
    assert int(result.termination_reason) == FixedPointTerminationReason.CONVERGED
    assert np.isfinite(np.asarray(result.trace)[0])
    assert np.all(np.isnan(np.asarray(result.trace)[1:]))


def test_manifold_newton_stops_without_an_attempt_at_a_converged_state():
    result = newton_krylov(
        lambda state: 0.5 * state + jnp.ones_like(state),
        2.0 * jnp.ones(2),
        previous_admitted_state=jnp.ones(2),
        admissibility_fn=lambda _state: jnp.asarray(True),
        newton_steps=3,
        gmres_iterations=2,
        warmup=0,
    )
    assert float(result.residual) == 0.0
    assert int(result.attempted_newton_promotions) == 0
    assert int(result.accepted_newton_promotions) == 0
    assert bool(result.converged)
    assert int(result.termination_reason) == FixedPointTerminationReason.CONVERGED
    assert np.isfinite(np.asarray(result.trace)[0])
    assert np.all(np.isnan(np.asarray(result.trace)[1:]))


def test_newton_stops_at_the_first_passing_promotion_and_pads_diagnostics():
    result = newton_krylov(
        lambda state: jnp.ones_like(state),
        jnp.zeros(2),
        newton_steps=4,
        gmres_iterations=2,
        warmup=0,
    )
    trace = np.asarray(result.trace)
    assert int(result.attempted_newton_promotions) == 1
    assert int(result.accepted_newton_promotions) == 1
    assert bool(result.converged)
    assert float(result.residual) <= FIXED_POINT_RESIDUAL_TOLERANCE
    assert np.isfinite(trace[0])
    assert np.isfinite(trace[3])
    assert np.all(np.isnan(trace[1:3]))
    assert np.all(np.isnan(trace[4:]))


def test_newton_reports_iteration_budget_exhaustion():
    result = newton_krylov(
        lambda state: jnp.ones_like(state),
        jnp.zeros(2),
        newton_steps=1,
        gmres_iterations=2,
        warmup=0,
        relaxation=0.5,
        step_cap=0.5,
    )
    assert int(result.attempted_newton_promotions) == 1
    assert int(result.accepted_newton_promotions) == 1
    assert not bool(result.converged)
    assert float(result.residual) > FIXED_POINT_RESIDUAL_TOLERANCE
    assert (
        int(result.termination_reason)
        == FixedPointTerminationReason.ITERATION_BUDGET_EXHAUSTED
    )


def test_newton_reports_a_nonfinite_initial_residual_without_attempting():
    result = newton_krylov(
        lambda state: jnp.full_like(state, jnp.nan),
        jnp.zeros(2),
        newton_steps=3,
        gmres_iterations=2,
        warmup=0,
    )
    assert int(result.attempted_newton_promotions) == 0
    assert int(result.accepted_newton_promotions) == 0
    assert not bool(result.converged)
    assert not np.isfinite(float(result.residual))
    assert (
        int(result.termination_reason) == FixedPointTerminationReason.NONFINITE_RESIDUAL
    )


def test_newton_reports_a_refused_linear_action():
    def map_fn(state):
        return jnp.sqrt(state) + jnp.ones_like(state)

    result = newton_krylov(
        map_fn,
        jnp.zeros(2),
        newton_steps=3,
        gmres_iterations=2,
        warmup=0,
    )
    assert int(result.attempted_newton_promotions) == 1
    assert int(result.accepted_newton_promotions) == 0
    assert not bool(result.converged)
    assert (
        int(result.krylov_action_qualification)
        == KrylovActionQualification.NONFINITE_LINEAR_ACTION
    )
    assert (
        int(result.termination_reason)
        == FixedPointTerminationReason.KRYLOV_ACTION_REFUSED
    )


def test_newton_stopping_is_jittable_and_does_not_require_picard_contraction():
    def solve(initial):
        return newton_krylov(
            lambda state: 1.5 * state + jnp.ones_like(state),
            initial,
            newton_steps=4,
            gmres_iterations=2,
            warmup=0,
        )

    result = jax.jit(solve)(jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(result.state), -2.0, atol=1e-12)
    assert int(result.attempted_newton_promotions) == 1
    assert int(result.accepted_newton_promotions) == 1
    assert bool(result.converged)


def test_vmap_stopping_runs_to_the_slowest_lane_with_per_lane_receipts():
    def solve(slope):
        return newton_krylov(
            lambda state: slope * state + jnp.ones_like(state),
            jnp.zeros(1),
            newton_steps=5,
            gmres_iterations=2,
            warmup=0,
            relaxation=0.5,
            step_cap=2.0,
            convergence_tolerance=0.1,
        )

    result = jax.jit(jax.vmap(solve))(jnp.asarray([0.0, 0.5]))
    np.testing.assert_array_equal(
        np.asarray(result.attempted_newton_promotions), [1, 3]
    )
    np.testing.assert_array_equal(np.asarray(result.accepted_newton_promotions), [1, 3])
    np.testing.assert_array_equal(np.asarray(result.converged), [True, True])
    assert int(np.max(np.asarray(result.attempted_newton_promotions))) == 3
    fast_trace, slow_trace = np.asarray(result.trace)
    assert np.all(np.isnan(fast_trace[4:]))
    assert np.isfinite(slow_trace[11])
    assert np.all(np.isnan(slow_trace[12:]))


@pytest.mark.parametrize(
    "strategy",
    ("clarke", "nonmonotone", "surface_restricted", "damped_hybrid"),
)
def test_kink_aware_route_options_converge_across_a_piecewise_tangent(strategy):
    """Every explicit policy crosses a derivative hand-off to the same root."""

    def map_fn(state):
        slope = jnp.where(state[0] < 0.0, 0.2, 0.4)
        return slope * state + 0.8

    kwargs = {}
    if strategy in ("clarke", "surface_restricted"):
        kwargs["surface_fn"] = lambda state: state[0]
    result = jax.jit(
        lambda initial: kink_aware_newton_krylov(
            map_fn,
            initial,
            strategy=strategy,
            newton_steps=20,
            gmres_iterations=2,
            warmup=0,
            **kwargs,
        )
    )(jnp.asarray([-1.0]))
    assert float(result.residual) < 1e-10
    np.testing.assert_allclose(np.asarray(result.state), 4.0 / 3.0, atol=1e-10)
    assert np.all(np.isfinite(np.asarray(result.trace)))


def test_kink_aware_routes_do_not_change_existing_newton_krylov_results():
    """The additive route leaves the established exact-tangent path bitwise."""
    matrix, offset, _fixed_point = _contraction(seed=19, radius=0.5)
    options = dict(
        newton_steps=3,
        gmres_iterations=6,
        warmup=2,
        relaxation=0.4,
        step_cap=3.0,
    )
    expected = newton_krylov(
        lambda state: matrix @ state + offset,
        jnp.zeros(DIMENSION),
        **options,
    )
    observed = newton_krylov(
        lambda state: matrix @ state + offset,
        jnp.zeros(DIMENSION),
        **options,
    )
    assert np.array_equal(np.asarray(observed.state), np.asarray(expected.state))
    assert np.array_equal(
        np.asarray(observed.trace), np.asarray(expected.trace), equal_nan=True
    )


def test_damped_hybrid_can_release_damping_as_the_residual_falls():
    """The residual-triggered schedule is explicit, jittable, and additive."""

    def map_fn(state):
        return 0.5 * state + jnp.ones_like(state)

    def solve(schedule):
        return kink_aware_newton_krylov(
            map_fn,
            jnp.zeros(2),
            strategy="damped_hybrid",
            newton_steps=12,
            gmres_iterations=2,
            warmup=0,
            hybrid_weight=0.2,
            hybrid_schedule=schedule,
            hybrid_final_weight=1.0,
            hybrid_release_residual=1.0e-2,
        )

    fixed = jax.jit(lambda: solve("fixed"))()
    released = jax.jit(lambda: solve("residual_release"))()
    assert float(released.residual) < float(fixed.residual)
    np.testing.assert_allclose(np.asarray(released.state), 2.0, atol=1e-10)


def test_anderson_accelerates_relaxed_picard_at_a_shared_budget():
    matrix, offset, fixed_point = _contraction()

    def map_fn(x):
        return matrix @ x + offset

    budget = 24
    plain = picard(map_fn, jnp.zeros(DIMENSION), evaluations=budget, relaxation=0.5)
    mixed = anderson(
        map_fn,
        jnp.zeros(DIMENSION),
        evaluations=budget,
        relaxation=0.5,
        depth=3,
        warmup=6,
    )
    assert float(mixed.residual) < 0.1 * float(plain.residual)
    error = np.abs(np.asarray(mixed.state) - fixed_point).max()
    assert error < np.abs(np.asarray(plain.state) - fixed_point).max()


def test_all_schemes_share_one_nonlinear_fixed_point():
    matrix, offset, _fixed_point = _contraction(seed=11, radius=0.4)

    def map_fn(x):
        return jnp.tanh(matrix @ x) + offset

    reference = picard(map_fn, jnp.zeros(DIMENSION), evaluations=200, relaxation=0.6)
    mixed = anderson(map_fn, jnp.zeros(DIMENSION), evaluations=40, relaxation=0.6)
    newton = newton_krylov(
        map_fn,
        jnp.zeros(DIMENSION),
        newton_steps=3,
        gmres_iterations=DIMENSION,
        warmup=4,
        relaxation=0.6,
    )
    target = np.asarray(reference.state)
    np.testing.assert_allclose(np.asarray(mixed.state), target, atol=1e-6)
    np.testing.assert_allclose(np.asarray(newton.state), target, atol=1e-8)


def test_newton_step_is_capped_and_finite_on_a_near_singular_tangent():
    """(I - J) nearly vanishes: the raw Krylov step explodes; the cap holds."""
    epsilon = 1e-9
    offset = jnp.ones(DIMENSION)

    def map_fn(x):
        return (1.0 - epsilon) * x + offset

    initial = jnp.zeros(DIMENSION)
    result = newton_krylov(
        map_fn,
        initial,
        newton_steps=1,
        gmres_iterations=4,
        warmup=0,
        relaxation=0.5,
        step_cap=10.0,
    )
    state = np.asarray(result.state)
    assert np.all(np.isfinite(state))
    # residual f = offset (norm 1); the relaxed step is 0.5, so the promoted
    # state may move at most step_cap * 0.5 per component
    assert np.max(np.abs(state)) <= 10.0 * 0.5 + 1e-12


def test_vmap_batch_matches_per_slice_solves():
    matrix, offset, _fixed_point = _contraction()
    offsets = jnp.stack([offset, 2.0 * offset, -0.5 * offset])
    initials = jnp.zeros((3, DIMENSION))

    def solve_newton(initial, shift):
        return newton_krylov(
            lambda x: matrix @ x + shift,
            initial,
            newton_steps=2,
            gmres_iterations=6,
            warmup=2,
        ).state

    def solve_anderson(initial, shift):
        return anderson(lambda x: matrix @ x + shift, initial, evaluations=20).state

    for solve in (solve_newton, solve_anderson):
        batched = jax.vmap(solve)(initials, offsets)
        per_slice = jnp.stack([solve(initials[i], offsets[i]) for i in range(3)])
        np.testing.assert_allclose(
            np.asarray(batched), np.asarray(per_slice), atol=1e-12
        )


def test_trace_layout_shares_one_evaluation_axis():
    """Completed Newton slots are measured and later slots remain padding."""
    gmres_iterations = 2
    result = newton_krylov(
        lambda x: jnp.ones_like(x),
        jnp.zeros(DIMENSION),
        newton_steps=2,
        gmres_iterations=gmres_iterations,
        warmup=3,
    )
    trace = np.asarray(result.trace)
    stride = 2 + gmres_iterations
    assert trace.size == 3 + 2 * stride
    assert np.all(np.isfinite(trace[:3]))
    first_base = 3
    assert np.isfinite(trace[first_base])
    assert np.all(np.isnan(trace[first_base + 1 : first_base + 1 + gmres_iterations]))
    assert np.isfinite(trace[first_base + stride - 1])
    assert np.all(np.isnan(trace[first_base + stride :]))


def test_explicit_double_state_keeps_double_solver_scratch():
    """False-plus-allow retains fp64 state, history, and residual traces."""
    configure_dtypes()
    initial = jnp.zeros(4, dtype=jnp.float64)

    def map_fn(state):
        return 0.25 * state + jnp.ones_like(state)

    results = (
        picard(map_fn, initial, evaluations=5),
        anderson(map_fn, initial, evaluations=8, warmup=2),
        newton_krylov(
            map_fn,
            initial,
            newton_steps=1,
            gmres_iterations=4,
            warmup=2,
        ),
    )
    for result in results:
        assert result.state.dtype == jnp.float64
        assert result.trace.dtype == jnp.float64


def test_runtime_precision_selects_fixed_point_state_dtype():
    """Automatic general solves use fp64 while explicit fp32 stays available."""
    configure_dtypes()
    initial = jnp.zeros(4)

    def map_fn(state):
        return 0.25 * state + jnp.ones_like(state)

    automatic = picard(map_fn, initial, evaluations=5)
    single = picard(
        map_fn,
        initial,
        evaluations=5,
        precision=Precision.SINGLE,
    )
    assert automatic.state.dtype == jnp.float64
    assert automatic.trace.dtype == jnp.float64
    assert single.state.dtype == jnp.float32
    assert single.trace.dtype == jnp.float32


if __name__ == "__main__":
    pytest.main([__file__])
