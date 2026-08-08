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

    from nova.jax.fixed_point import anderson, newton_krylov, picard


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
    """Newton tangent passes appear as NaN slots; measured slots are finite."""
    matrix, offset, _fixed_point = _contraction()
    gmres_iterations = 5
    result = newton_krylov(
        lambda x: matrix @ x + offset,
        jnp.zeros(DIMENSION),
        newton_steps=2,
        gmres_iterations=gmres_iterations,
        warmup=3,
    )
    trace = np.asarray(result.trace)
    stride = 2 + gmres_iterations
    assert trace.size == 3 + 2 * stride
    assert np.all(np.isfinite(trace[:3]))  # warmup sweeps measured
    for step in range(2):
        base = 3 + step * stride
        assert np.isfinite(trace[base])  # linearisation value
        assert np.all(np.isnan(trace[base + 1 : base + 1 + gmres_iterations]))
        assert np.isfinite(trace[base + stride - 1])  # promotion read


if __name__ == "__main__":
    pytest.main([__file__])
