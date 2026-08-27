"""Focused coverage for the two-branch batch-cost analytic carrier."""

import numpy as np

from benchmarks.two_branch_batch_cost import (
    LIMITED_LATTICE_SHAPE,
    _limited_tensor_machine,
)
from nova.equilibrium.topology import TopologyClass
from scripts.analytic_oracle_fixtures.measure import (
    analytic_case,
    exact_state,
    forward_operator,
)


def test_limited_fixture_support_partition_uses_a_tensor_product_grid() -> None:
    """One analytic slice reaches the production support partition on CPU."""
    case = analytic_case()
    machine = _limited_tensor_machine()
    radius = np.unique(machine.node[:, 0])
    height = np.unique(machine.node[:, 1])
    assert (len(radius), len(height)) == LIMITED_LATTICE_SHAPE
    np.testing.assert_array_equal(
        machine.node,
        np.c_[np.repeat(radius, len(height)), np.tile(height, len(radius))],
    )

    state = exact_state(
        case,
        np.vstack((machine.node, machine.wall_node, machine.sample_coordinates)),
    )
    partition = forward_operator(case, machine)._support_partition(
        state, TopologyClass.LIMITED
    )

    assert partition[3].vertex_count.shape == (len(machine.node),)
    assert np.all(np.isfinite(state))
