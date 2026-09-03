"""Acceptance scoring on the incumbent partition while the mask is unsettled."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import _newton_krylov_inner, newton_krylov
    from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _enable_float64():
    configure_dtypes()


# A candidate landing past the threshold switches the acceptance-test target
# from OFF_TARGET to ON_TARGET; the resulting jump makes that same candidate
# fail induced-mask scoring (its own mapped value is far from itself) while
# it passes incumbent-mask scoring (the incumbent's frozen mask still maps
# every candidate to OFF_TARGET, which the full Newton step reaches exactly).
_FROZEN_MASK = jnp.asarray([False])
_ON_TARGET = jnp.asarray([3.0])
_OFF_TARGET = jnp.asarray([10.0])
_INDUCED_THRESHOLD = 8.0


def _shadowed_map(_state, mask):
    return jnp.where(mask, _ON_TARGET, _OFF_TARGET)


def _frozen_mask(_state):
    return _FROZEN_MASK


def _induced_mask(candidate, _previous):
    return candidate >= _INDUCED_THRESHOLD


def _solve(presettlement):
    return _newton_krylov_inner(
        lambda state: _shadowed_map(state, _frozen_mask(state)),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=_frozen_mask,
        promoted_shadow_mask_fn=lambda state, _previous: _frozen_mask(state),
        shadowed_map_fn=_shadowed_map,
        model_trust_selection=False,
        acceptance_shadow_mask_fn=_induced_mask,
        acceptance_shadowed_map_fn=_shadowed_map,
        own_mask_acceptance=True,
        presettlement_incumbent_scoring=presettlement,
    )


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_presettlement_scoring_accepts_what_induced_scoring_refuses(compiled):
    evaluate = jax.jit(_solve) if compiled else _solve

    settled = evaluate(jnp.asarray(False))
    presettlement = evaluate(jnp.asarray(True))

    # Post-settlement: the full Newton step (candidate == OFF_TARGET, its own
    # induced mask flips ON) is refused on its own induced-mask residual, so
    # backtracking falls to the next rung, which stays under the threshold
    # and is scored on its own (unchanged) induced mask.
    np.testing.assert_allclose(settled.state, [5.0])
    np.testing.assert_allclose(settled.residual, 0.5)
    assert int(settled.promotion_backtrack_counts[0]) == 1

    # Pre-settlement: the same first-rung candidate is scored against the
    # incumbent's own frozen mask instead of its induced mask, so it is
    # accepted outright at the full Newton step.
    np.testing.assert_allclose(presettlement.state, [10.0])
    np.testing.assert_allclose(presettlement.residual, 0.0)
    assert int(presettlement.promotion_backtrack_counts[0]) == 0

    assert bool(settled.accepted_newton_promotions)
    assert bool(presettlement.accepted_newton_promotions)


def test_presettlement_incumbent_scoring_defaults_to_induced_mask_scoring():
    default = _solve(False)
    unspecified = _newton_krylov_inner(
        lambda state: _shadowed_map(state, _frozen_mask(state)),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=_frozen_mask,
        promoted_shadow_mask_fn=lambda state, _previous: _frozen_mask(state),
        shadowed_map_fn=_shadowed_map,
        model_trust_selection=False,
        acceptance_shadow_mask_fn=_induced_mask,
        acceptance_shadowed_map_fn=_shadowed_map,
        own_mask_acceptance=True,
    )
    np.testing.assert_array_equal(default.state, unspecified.state)
    np.testing.assert_array_equal(default.residual, unspecified.residual)


def test_presettlement_incumbent_scoring_requires_own_mask_acceptance():
    with pytest.raises(ValueError, match="own-mask acceptance"):
        newton_krylov(
            lambda state: state,
            jnp.zeros(1),
            newton_steps=1,
            presettlement_incumbent_scoring=True,
        )


def test_active_set_wiring_threads_presettlement_through_newton_krylov():
    def mask_fn(state):
        return state >= 0.5

    def shadowed_map(_state, mask):
        return jnp.where(mask, 2.0, 1.0)

    def solve(enabled):
        return newton_krylov(
            lambda state: shadowed_map(state, mask_fn(state)),
            jnp.zeros(1),
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            shadow_mask_fn=mask_fn,
            promoted_shadow_mask_fn=lambda state, _previous: mask_fn(state),
            shadowed_map_fn=shadowed_map,
            active_set_steps=4,
            presettlement_incumbent_scoring=enabled,
        )

    disabled_result = solve(False)
    enabled_result = solve(True)

    np.testing.assert_allclose(disabled_result.state, [2.0])
    np.testing.assert_allclose(enabled_result.state, [2.0])
    assert bool(disabled_result.converged)
    assert bool(enabled_result.converged)
