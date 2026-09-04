"""Active-set invariants of augmented constraint rows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.constraint import assemble_augmented_system
from nova.jax.config import configure_dtypes
from tests.test_equilibrium_constraint_protocol import (
    _CoordinateFunctional,
    _pair,
    _profile,
)
from nova.equilibrium.constraint import ConstraintMultiplier


def _system():
    profile = _profile(changing_mask=True)
    pair = _pair(
        _CoordinateFunctional(1),
        ConstraintMultiplier(jnp.asarray([1.0])),
        target=1.0,
        payload=1.0,
    )

    def base_mask(flux):
        return profile.operator.residual_shadow_mask(flux)

    def promoted(flux, previous):
        return profile.operator.residual_shadow_mask(flux, previous_shadow=previous)

    return assemble_augmented_system(
        profile,
        jnp.asarray([0.25, 0.5]),
        (pair,),
        base_map=profile.operator.flux_map(),
        base_shadow_mask=base_mask,
        base_promoted_shadow_mask=promoted,
        base_shadowed_map=profile.operator.flux_map_with_shadow(),
        requested_class=None,
        target_current=None,
    )


def test_constraint_mask_tail_stays_false_when_flux_mask_changes() -> None:
    configure_dtypes()
    system = _system()
    initial_mask = system.shadow_mask_fn(system.initial)
    changed = system.initial.at[0].set(1.5)
    promoted = system.promoted_shadow_mask_fn(changed, initial_mask)

    np.testing.assert_array_equal(initial_mask, [False, False, False])
    np.testing.assert_array_equal(promoted, [True, False, False])
    assert not bool(promoted[-1])
    assert system.shadowed_map_fn(changed, promoted).shape == changed.shape


def test_full_augmented_residual_action_matches_central_difference() -> None:
    configure_dtypes()
    system = _system()
    state = system.initial
    direction = jnp.asarray([0.2, -0.1, 0.3])

    def residual(value):
        return value - system.map_fn(value)

    _value, tangent = jax.jvp(residual, (state,), (direction,))
    step = 1.0e-5
    difference = (
        residual(state + step * direction) - residual(state - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(tangent, difference, rtol=3.0e-9, atol=3.0e-9)
    assert tangent.shape == state.shape
