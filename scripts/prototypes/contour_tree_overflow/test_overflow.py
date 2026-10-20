"""Refuse a carrier wider than the neighbour table a caller declares.

The node and edge capacities are now sized from the carrier, so the tree no
longer refuses a wide mesh for the width of a narrow one.  The refusal that
remains is the one a caller asks for: a declared neighbour width below the
carrier's own maximum degree loses real adjacency, and the receipt must refuse
rather than report a topology built from a partial table.

The carrier that refusal is asserted on is a constructed fan, whose hub degree
is a property of the construction.  ``probe.mastu_inputs`` remains the owner of
the MAST-U carrier inputs, and the MAST-U degrees this probe measures are
recorded in ``MASTU_MEASURED_DEGREES`` as measurement rather than as an
assumption: the refusal record measured 19, 22 and 18 on the pre-repair
carrier, and the wall-coverage repair re-cut it, so the degrees are pinned here
from what the probe reports today.
"""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from nova.equilibrium import contour_tree  # noqa: E402
from scripts.prototypes.contour_tree_overflow.probe import (  # noqa: E402
    mastu_inputs,
)

assert jax.config.jax_enable_x64 is True

# The live maximum degree the MAST-U carrier reports at each lattice
# resolution, as measured by this probe on the wall-complete carrier.
MASTU_MEASURED_DEGREES = ((16, 9), (24, 11), (32, 12))

# A hub of degree 24, so a table 16 wide cannot hold its adjacency.
FAN_LEAVES = 24
FAN_DECLARED_WIDTH = 16


def _fan_carrier(leaves: int):
    """One hub joined to ``leaves`` leaves, plus one unused carrier slot.

    The hub's degree is ``leaves`` by construction, which makes the width a
    table needs a property of the carrier rather than of the field sampled on
    it.  The final slot is left invalid, as a carrier with a virtual outside
    node has one.
    """

    vertex_count = leaves + 2
    values = jnp.asarray(
        [2.0] + [1.0 - index / (2 * leaves) for index in range(leaves)] + [0.0]
    )
    valid = jnp.asarray([True] * (leaves + 1) + [False])
    wall = jnp.asarray([False] * leaves + [True, False])
    edges = jnp.asarray([[0, index] for index in range(1, leaves + 1)], dtype=jnp.int32)
    return vertex_count, values, valid, wall, edges, jnp.ones(leaves, dtype=bool)


def test_small_field_stays_within_capacity():
    tree = contour_tree.build_contour_tree(
        jnp.array((3.0, 2.0, 1.0, 0.0, 0.0)),
        jnp.array((True, True, True, True, False)),
        jnp.array((False, False, True, False, False)),
        jnp.array(((0, 1), (1, 2), (2, 3)), dtype=jnp.int32),
        jnp.ones(3, dtype=bool),
        jnp.array(1.0),
    )
    assert not bool(tree.overflow)


def test_machine_carrier_fits_the_measured_degree_table():
    for resolution, degree in MASTU_MEASURED_DEGREES:
        arguments, _, mesh = mastu_inputs(resolution)
        assert int(mesh.neighbour_capacity) == degree, (
            f"resolution {resolution} measured degree "
            f"{int(mesh.neighbour_capacity)} rather than the recorded {degree}"
        )
        tree = contour_tree.build_contour_tree(
            *arguments, neighbour_capacity=mesh.neighbour_capacity
        )
        assert not bool(tree.overflow), (
            f"tree refused the carrier: nodes={int(tree.node_valid.sum())}/"
            f"{tree.node_vertex.size}, "
            f"edges={int(tree.edge_valid.sum())}/{tree.edges.shape[0]}"
        )


def test_declared_width_below_the_carrier_degree_refuses():
    """A table narrower than the carrier's degree refuses; its own degree fits.

    The width is measured from the constructed fan rather than assumed, so the
    test fails if the refusal stops firing rather than if the carrier is
    re-cut.
    """

    vertex_count, values, valid, wall, edges, edge_valid = _fan_carrier(FAN_LEAVES)
    degree = contour_tree.measured_degree_capacity(edges, edge_valid, vertex_count)
    assert degree == FAN_LEAVES > FAN_DECLARED_WIDTH, (
        f"constructed fan of {FAN_LEAVES} leaves measured degree {degree}"
    )

    refused = contour_tree.build_contour_tree(
        values,
        valid,
        wall,
        edges,
        edge_valid,
        jnp.array(1.0),
        neighbour_capacity=FAN_DECLARED_WIDTH,
    )
    assert bool(refused.overflow), (
        f"a table {FAN_DECLARED_WIDTH} wide did not refuse a hub of degree {degree}"
    )

    measured = contour_tree.build_contour_tree(
        values,
        valid,
        wall,
        edges,
        edge_valid,
        jnp.array(1.0),
        neighbour_capacity=degree,
    )
    assert not bool(measured.overflow), (
        f"the measured degree {degree} did not fit the carrier it was measured from"
    )

    # A traced build cannot count its carrier host-side, so this path takes the
    # safe edge-count bound; the point asserted here is that it refuses nothing
    # the graph permits.
    undeclared = contour_tree.build_contour_tree(
        values, valid, wall, edges, edge_valid, jnp.array(1.0)
    )
    assert not bool(undeclared.overflow), (
        f"an undeclared width refused a carrier whose degree is {degree}"
    )
