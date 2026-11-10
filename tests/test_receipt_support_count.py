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

from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.biot.target import FluxTarget
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.domain import PlasmaDomain
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.fixed_point import newton_krylov
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.solve_request import (
        ExplicitSolveSeed,
        ForwardSolveRequest,
    )
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes


SUPPORT_CELLS = 5


def _zero(value):
    return jnp.zeros_like(value)


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


def _raster_operator():
    """A small structured operator, cheap to build and never solved."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.6, 1.4, 9), np.linspace(-0.4, 0.4, 9))
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    wall_coordinate = np.c_[
        1.0 + 1.5 * np.cos(wall_angle),
        1.5 * np.sin(wall_angle),
    ]
    node_count = lattice.node_count
    wall_count = len(wall_coordinate)
    operator = ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((node_count, 1)),
            plasma_target=jnp.eye(node_count),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape), maxsize=5
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((wall_count, 1)),
            plasma_target=jnp.zeros((wall_count, node_count)),
            null=Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64)),
        ),
        source=ForwardSource(core=DomainProfile(p_prime=_zero, ff_prime=_zero)),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        use_linear_moments=False,
    )
    return operator, lattice


class _RecordedHistoryProfile(ForwardProfile):
    """A profile whose equilibrium map returns one recorded fixed-point history.

    Only the map is replaced: the request seam below it, the publication that
    builds the receipt, and the receipt itself are the production ones, so a
    keyword swapped at the publication site reaches the assertions.
    """

    recorded = None

    def solve(self, initial_flux, **kwargs):
        if isinstance(initial_flux, ForwardSolveRequest):
            return super().solve(initial_flux, **kwargs)
        return self.recorded


def _recorded_history(**overrides):
    history = {
        "residual": jnp.asarray(1.0e-9),
        "termination_reason": 0,
        "attempted_newton_promotions": 0,
        "accepted_newton_promotions": 0,
        "inner_iteration_decisions": jnp.zeros(0, dtype=jnp.int32),
        "inner_iteration_applied_factors": jnp.zeros(0),
    }
    history.update(overrides)
    return SimpleNamespace(**history)


def _published_receipt(history, route):
    """Solve one request through the production publication, with a fixed history."""
    operator, lattice = _raster_operator()
    profile = _RecordedHistoryProfile(operator=operator, lattice=lattice)
    profile.recorded = SimpleNamespace(
        fixed_point=history,
        finite=SimpleNamespace(passed=jnp.asarray(True)),
        topology=None,
        normalisation=None,
    )
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="receipt-support-publication",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(jnp.zeros(lattice.node_count)),
        policy_overrides={"route": route},
    )
    return profile.solve(request)


def test_the_published_receipt_keeps_its_support_count_apart_from_its_changes():
    history = _recorded_history(
        active_set_residuals=jnp.asarray((1.0e-2, 1.0e-3)),
        active_set_mask_differences=jnp.asarray((9, 0)),
        active_set_support_cells=jnp.asarray((345, 345)),
    )

    receipt = _published_receipt(history, "newton_krylov")

    assert int(receipt.support_cell_count[-1]) == 345
    assert int(receipt.changed_mask_entries[-1]) == 0
    assert int(receipt.support_cell_count[-1]) != int(receipt.changed_mask_entries[-1])


def test_a_route_without_active_set_trips_publishes_no_support_history():
    history = _recorded_history(trace=jnp.asarray((1.0,)), shadow_mask_changes=7)

    receipt = _published_receipt(history, "picard")

    assert int(receipt.changed_mask_entries) == 7
    assert np.asarray(receipt.support_cell_count).size == 0
