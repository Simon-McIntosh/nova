"""Candidate admission on each state's induced active set."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import (
        _backtracked_promotion,
        newton_krylov,
    )
    from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _enable_float64():
    configure_dtypes()


def _assert_result_equal(left, right):
    for field in left._fields:
        left_value = np.asarray(getattr(left, field))
        right_value = np.asarray(getattr(right, field))
        if np.issubdtype(left_value.dtype, np.inexact):
            assert np.array_equal(left_value, right_value, equal_nan=True), field
        else:
            np.testing.assert_array_equal(left_value, right_value, err_msg=field)


def _settled_mask(_state):
    return jnp.zeros(2, dtype=bool)


def _solve_without_mask_changes(enabled, **options):
    def shadowed_map(state, _mask):
        return 0.5 * state + jnp.asarray([1.0, -0.5])

    return newton_krylov(
        lambda state: shadowed_map(state, _settled_mask(state)),
        jnp.zeros(2),
        newton_steps=2,
        gmres_iterations=2,
        warmup=0,
        shadow_mask_fn=_settled_mask,
        promoted_shadow_mask_fn=lambda _state, previous: previous,
        shadowed_map_fn=shadowed_map,
        active_set_steps=2,
        stop_on_active_set_stagnation=False,
        own_mask_acceptance=enabled,
        **options,
    )


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_own_mask_acceptance_is_inert_when_candidates_keep_the_mask(compiled):
    def solve(enabled):
        return _solve_without_mask_changes(
            enabled,
            presettlement_incumbent_scoring=False,
        )

    evaluate = jax.jit(solve, static_argnums=0) if compiled else solve
    guarded = evaluate(True)
    frozen = evaluate(False)

    _assert_result_equal(guarded, frozen)


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_default_presettlement_scoring_accepts_incumbent_mask_decrease(compiled):
    def solve():
        return _solve_without_mask_changes(True)

    result = jax.jit(solve)() if compiled else solve()

    np.testing.assert_allclose(result.state, [2.0, -1.0])
    assert float(result.residual) < 1.0
    assert int(result.attempted_newton_promotions) == 1
    assert int(result.accepted_newton_promotions) == 1


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_candidate_worsening_by_measured_own_mask_ratio_is_refused(compiled):
    incumbent_residual = jnp.asarray(0.04070199248460991)
    candidate_residual = jnp.asarray(0.08368091815180548)

    def mapped_for_residual(state, residual):
        return state / (1.0 - residual)

    def frozen_map(state):
        value = state[0]
        residual = jnp.where(
            value == 1.0,
            incumbent_residual,
            jnp.where(value == 2.0, 0.02, 0.5),
        )
        return mapped_for_residual(state, residual)

    def own_mask_map(state):
        value = state[0]
        residual = jnp.where(
            value == 1.0,
            incumbent_residual,
            jnp.where(value == 2.0, candidate_residual, 0.5),
        )
        return mapped_for_residual(state, residual)

    def promote(enabled):
        return _backtracked_promotion(
            frozen_map,
            frozen_map,
            jnp.ones(1),
            jnp.ones(1),
            jnp.zeros(1),
            incumbent_residual,
            incumbent_residual,
            jnp.asarray(1.0),
            False,
            acceptance_map_fn=own_mask_map if enabled else frozen_map,
            own_mask_acceptance=enabled,
        )

    evaluate = jax.jit(promote, static_argnums=0) if compiled else promote
    guarded = evaluate(True)
    frozen = evaluate(False)

    assert float(candidate_residual / incumbent_residual) == pytest.approx(2.05594)
    np.testing.assert_array_equal(frozen.state, [2.0])
    assert bool(frozen.accepted)
    np.testing.assert_array_equal(guarded.state, [1.0])
    assert not bool(guarded.accepted)
