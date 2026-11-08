"""The receipt's support population is separate from its change count.

``ForwardSolveReceipt.support_cell_count`` publishes the axis-connected
support size observed at each active-set trip, and
``ForwardSolveReceipt.changed_mask_entries`` publishes how many labels and
support memberships moved.  A settled trip changes nothing, so its change
count is zero while its support is still populated; the counters below are
the two the receipt carries verbatim, and the contract is that a zero change
count never stands in for an empty support.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium.domain import PlasmaDomain
    from nova.equilibrium.fixed_point import newton_krylov
    from nova.jax.config import configure_dtypes


SUPPORT_CELLS = 5


def _support_labels():
    """Five axis-connected cells beside one of every other domain label."""
    return np.asarray(
        [PlasmaDomain.CORE] * SUPPORT_CELLS
        + [
            PlasmaDomain.PRIVATE_FLUX,
            PlasmaDomain.COMMON_SOL,
            PlasmaDomain.EXCLUDED_MATERIAL,
        ],
        dtype=np.int8,
    )


class _Partition(NamedTuple):
    """Minimal frozen partition: a domain label per cell and its shadow."""

    label: Any
    profile_support: Any
    residual_shadow: Any


def _settling_solver(labels=None):
    """Return one active-set solve whose mask settles on its second trip."""
    if labels is None:
        labels = _support_labels()

    def mask(state):
        return state >= 0.5

    def shadowed_map(state, frozen_mask):
        return jnp.where(frozen_mask, 2.0, 1.0)

    def read_partition(state, previous_shadow=None):
        return _Partition(
            label=jnp.asarray(labels),
            profile_support=None,
            residual_shadow=mask(state),
        )

    shadowed_map._read_frozen_partition = read_partition
    shadowed_map._map_frozen_partition = lambda state, partition: shadowed_map(
        state, partition.residual_shadow
    )
    shadowed_map._frozen_partition_shadow = lambda partition: partition.residual_shadow

    return newton_krylov(
        lambda state: shadowed_map(state, mask(state)),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda state, _previous: mask(state),
        shadowed_map_fn=shadowed_map,
        active_set_steps=4,
        model_trust_selection=False,
    )


def _executed(values):
    values = np.asarray(values)
    return values[values >= 0]


def test_support_count_stays_positive_where_changed_mask_entries_reach_zero():
    configure_dtypes()
    result = _settling_solver()

    support = _executed(result.active_set_support_cells)
    changes = _executed(result.active_set_mask_differences)

    assert support.size > 0
    np.testing.assert_array_equal(support, np.full(support.size, SUPPORT_CELLS))
    assert support[-1] > 0
    assert changes[-1] == 0
    assert changes[0] > 0


def test_support_count_is_the_axis_connected_population_not_the_mask_total():
    """Excluded, private-flux and common-SOL cells are not support cells."""

    configure_dtypes()
    labels = np.asarray(
        [
            PlasmaDomain.CORE,
            PlasmaDomain.PRIVATE_FLUX,
            PlasmaDomain.COMMON_SOL,
            PlasmaDomain.EXCLUDED_MATERIAL,
        ],
        dtype=np.int8,
    )
    result = _settling_solver(labels=labels)

    np.testing.assert_array_equal(_executed(result.active_set_support_cells), [1, 1])


def test_a_settled_empty_support_is_reported_as_zero_not_as_the_change_count():
    configure_dtypes()
    labels = np.asarray(
        [PlasmaDomain.COMMON_SOL, PlasmaDomain.EXCLUDED_MATERIAL],
        dtype=np.int8,
    )
    result = _settling_solver(labels=labels)

    support = _executed(result.active_set_support_cells)
    changes = _executed(result.active_set_mask_differences)

    np.testing.assert_array_equal(support, np.zeros_like(support))
    assert changes[0] > 0
    assert changes[-1] == 0
