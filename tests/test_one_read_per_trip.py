"""Contracts for freezing discrete topology through each Newton trip."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import newton_krylov
    from nova.jax.config import configure_dtypes


def _partitioned_solver(*, count_reads=False, changing_mask=True):
    counts = defaultdict(int)

    def mask(state):
        if changing_mask:
            return state >= 0.5
        return jnp.zeros_like(state, dtype=bool)

    def shadowed_map(state, frozen_mask):
        if changing_mask:
            return jnp.where(frozen_mask, 2.0, 1.0)
        return 0.25 * state + 0.75

    def read_partition(state, previous_shadow=None):
        if count_reads:
            kind = "initial" if previous_shadow is None else "boundary"
            jax.debug.callback(lambda: counts.__setitem__(kind, counts[kind] + 1))
        return mask(state)

    shadowed_map._read_frozen_partition = read_partition
    shadowed_map._map_frozen_partition = shadowed_map
    shadowed_map._frozen_partition_shadow = lambda partition: partition

    result = newton_krylov(
        lambda state: shadowed_map(state, mask(state)),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda state, _previous: mask(state),
        shadowed_map_fn=shadowed_map,
        active_set_steps=3,
        model_trust_selection=False,
    )
    jax.block_until_ready(result.state)
    return result, counts


def test_topology_is_read_once_at_each_trip_boundary():
    configure_dtypes()
    result, counts = _partitioned_solver(count_reads=True)

    np.testing.assert_array_equal(result.active_set_mask_differences, [1, 0, -1])
    assert int(result.active_set_iterations) == 2
    assert counts == {"initial": 1, "boundary": 2}


def test_frozen_partition_is_bit_identical_when_the_mask_never_changes():
    configure_dtypes()
    frozen, _counts = _partitioned_solver(changing_mask=False)

    def stable_mask(state):
        return jnp.zeros_like(state, dtype=bool)

    def stable_map(state, _mask):
        return 0.25 * state + 0.75

    current = newton_krylov(
        lambda state: stable_map(state, stable_mask(state)),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=stable_mask,
        promoted_shadow_mask_fn=lambda state, _previous: stable_mask(state),
        shadowed_map_fn=stable_map,
        active_set_steps=3,
        model_trust_selection=False,
    )

    for frozen_leaf, current_leaf in zip(
        jax.tree.leaves(frozen), jax.tree.leaves(current), strict=True
    ):
        np.testing.assert_array_equal(frozen_leaf, current_leaf)
