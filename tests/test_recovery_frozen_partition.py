"""Recovery scoring contracts for one selected topology partition."""

from __future__ import annotations

from collections import Counter

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.biot.target import FluxTarget
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium import fixed_point
    from nova.equilibrium.fixed_point import newton_krylov
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.jax.config import configure_dtypes
    from nova.geometry.hexstencil import hex_stencil


def _eager_partition_selection(
    candidate,
    previous_shadow,
    use_incumbent_partition,
    frozen_map_fn,
    induced_shadow_fn,
    induced_map_fn,
):
    candidate_shadow = jnp.ravel(induced_shadow_fn(candidate, previous_shadow))
    induced = induced_map_fn(candidate, candidate_shadow)
    return jnp.where(
        jnp.asarray(use_incumbent_partition),
        frozen_map_fn(candidate),
        induced,
    )


def _recovery_and_rebuild_solve(selector):
    counts = Counter()
    tangent = jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])

    def record(kind):
        counts.update((kind,))

    def map_fn(state):
        at_origin = jnp.all(state == 0.0)
        origin_model = jnp.asarray([1.0, 0.0]) + tangent @ state
        inside_damped_basin = (
            (state[1] > 0.0) & (state[1] < 0.02) & (jnp.abs(state[0]) < 1.0e-12)
        )
        candidate_model = jnp.where(inside_damped_basin, state, -state)
        return jnp.where(at_origin, origin_model, candidate_model)

    def mask(_state):
        return jnp.zeros(1, dtype=bool)

    def shadowed_map(state, _mask):
        return map_fn(state)

    def read_partition(state, _previous_shadow=None):
        jax.debug.callback(lambda: record("partition_read"), ordered=True)
        return mask(state)

    def map_partition(state, _partition):
        jax.debug.callback(lambda: record("partition_map"), ordered=True)
        return map_fn(state)

    shadowed_map._read_frozen_partition = read_partition
    shadowed_map._map_frozen_partition = map_partition
    shadowed_map._frozen_partition_shadow = lambda partition: partition

    original = fixed_point._acceptance_map_on_selected_partition
    fixed_point._acceptance_map_on_selected_partition = selector
    try:
        solve = jax.jit(
            lambda: newton_krylov(
                map_fn,
                jnp.zeros(2),
                newton_steps=6,
                gmres_iterations=2,
                warmup=0,
                shadow_mask_fn=mask,
                promoted_shadow_mask_fn=lambda state, _previous: mask(state),
                shadowed_map_fn=shadowed_map,
                active_set_steps=2,
            )
        )
        result = solve()
        jax.block_until_ready(result.state)
    finally:
        fixed_point._acceptance_map_on_selected_partition = original
        jax.clear_caches()
    return result, counts


def _bank_row_shaped_operator() -> ForwardFluxOperator:
    """Build the smallest structured point-current carrier used by a bank row."""
    lattice = FluxLattice(np.linspace(1.05, 2.35, 7), np.linspace(-0.72, 0.72, 7))
    angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    wall = np.c_[1.7 + 0.64 * np.cos(angle), 0.68 * np.sin(angle)]

    def zero_profile(psi_norm):
        return jnp.zeros_like(psi_norm)

    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((lattice.node_count, 1)),
            plasma_target=jnp.zeros((lattice.node_count, 1)),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape), maxsize=5
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((len(wall), 1)),
            plasma_target=jnp.zeros((len(wall), 1)),
            null=Null1D(jnp.asarray(wall, dtype=jnp.float64)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=zero_profile, ff_prime=zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        use_linear_moments=False,
    )


def test_point_current_bank_row_attaches_frozen_partition_hooks():
    configure_dtypes()
    mapped = _bank_row_shaped_operator().flux_map_with_shadow()

    assert callable(getattr(mapped, "_read_frozen_partition"))
    assert callable(getattr(mapped, "_map_frozen_partition"))
    assert callable(getattr(mapped, "_frozen_partition_shadow"))


def test_recovery_and_rebuild_reuse_the_frozen_partition_bit_identically():
    configure_dtypes()
    eager, eager_counts = _recovery_and_rebuild_solve(_eager_partition_selection)
    selected, selected_counts = _recovery_and_rebuild_solve(
        fixed_point._acceptance_map_on_selected_partition
    )

    for eager_leaf, selected_leaf in zip(
        jax.tree.leaves(eager), jax.tree.leaves(selected), strict=True
    ):
        np.testing.assert_array_equal(eager_leaf, selected_leaf)

    assert int(selected.promotion_recovery_activations[0]) == 1
    assert int(selected.promotion_model_rebuild_activations[0]) == 1
    assert selected_counts["partition_read"] == eager_counts["partition_read"] == 2
    assert selected_counts["partition_map"] == 27
    assert eager_counts["partition_map"] == 47


def test_candidate_induced_partition_is_read_only_when_it_is_authoritative():
    configure_dtypes()
    counts = Counter()

    def frozen_map(candidate):
        jax.debug.callback(lambda: counts.update(("frozen",)), ordered=True)
        return candidate + 1.0

    def induced_shadow(candidate, _previous):
        jax.debug.callback(lambda: counts.update(("induced_read",)), ordered=True)
        return candidate > 0.0

    def induced_map(candidate, shadow):
        jax.debug.callback(lambda: counts.update(("induced_map",)), ordered=True)
        return jnp.where(shadow, candidate + 2.0, candidate)

    evaluate = jax.jit(
        lambda use_incumbent: fixed_point._acceptance_map_on_selected_partition(
            jnp.ones(1),
            jnp.zeros(1, dtype=bool),
            use_incumbent,
            frozen_map,
            induced_shadow,
            induced_map,
        )
    )

    frozen = evaluate(jnp.asarray(True))
    jax.block_until_ready(frozen)
    assert counts == {"frozen": 1}

    counts.clear()
    induced = evaluate(jnp.asarray(False))
    jax.block_until_ready(induced)
    assert counts == {"induced_read": 1, "induced_map": 1}
    np.testing.assert_array_equal(frozen, [2.0])
    np.testing.assert_array_equal(induced, [3.0])
