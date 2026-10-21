"""The receipt's node, edge and neighbour capacities come from its carrier.

A capacity fixed for one mesh refuses another for no reason the carrier can see.
These tests pin the derivation rule at the three widths the build takes: the
node rows (one per carrier vertex), the edge rows (one per carrier edge slot)
and the neighbour table (the carrier's measured degree, or the safe edge-count
bound when no degree is supplied).  A caller that declares a capacity below what
the carrier produces is refused through the receipt's ``overflow`` flag rather
than handed a truncated tree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import contour_tree
from nova.jax.config import configure_dtypes


configure_dtypes()
assert jax.config.jax_enable_x64 is True


def _path_carrier():
    """A five-slot carrier holding one path of four live vertices."""

    values = jnp.array((3.0, 2.0, 1.0, 0.0, 0.0))
    valid = jnp.array((True, True, True, True, False))
    wall = jnp.array((False, False, True, False, False))
    edges = jnp.array(((0, 1), (1, 2), (2, 3)), dtype=jnp.int32)
    return values, valid, wall, edges, jnp.ones(3, dtype=bool), jnp.array(1.0)


def _complete_carrier(vertex_count: int):
    """A complete graph, so every vertex degree is ``vertex_count - 1``."""

    values = jnp.arange(vertex_count, 0, -1, dtype=jnp.float64)
    valid = jnp.ones(vertex_count, dtype=bool)
    wall = jnp.zeros(vertex_count, dtype=bool).at[vertex_count - 1].set(True)
    pairs = [
        (left, right)
        for left in range(vertex_count)
        for right in range(left + 1, vertex_count)
    ]
    edges = jnp.asarray(pairs, dtype=jnp.int32)
    return values, valid, wall, edges, jnp.ones(len(pairs), dtype=bool), jnp.array(1.0)


def test_undeclared_capacities_are_the_carrier_slot_counts():
    """With nothing declared the receipt is exactly as wide as its carrier."""

    values, valid, wall, edges, edge_valid, sigma = _path_carrier()
    receipt = contour_tree.build_contour_tree(
        values, valid, wall, edges, edge_valid, sigma
    )
    jax.block_until_ready(receipt)
    assert int(receipt.node_vertex.size) == int(values.size)
    assert int(receipt.edges.shape[0]) == int(edges.shape[0])
    assert not bool(receipt.overflow)


def test_declared_capacity_below_the_carrier_needs_is_refused():
    """A declared capacity smaller than the carrier produces refuses visibly."""

    arguments = _path_carrier()
    refused = contour_tree.build_contour_tree(
        *arguments, node_capacity=1, edge_capacity=1
    )
    jax.block_until_ready(refused)
    assert bool(refused.overflow)

    fitting = contour_tree.build_contour_tree(
        *arguments, node_capacity=5, edge_capacity=3
    )
    jax.block_until_ready(fitting)
    assert not bool(fitting.overflow)


def test_dense_graph_of_degree_above_thirty_two_bounds_without_refusal():
    """No measured degree: the table takes the safe edge-count bound."""

    vertex_count = 40
    values, valid, wall, edges, edge_valid, sigma = _complete_carrier(vertex_count)
    degree = int(np.max(np.bincount(np.asarray(edges).ravel(), minlength=vertex_count)))
    assert degree == vertex_count - 1 > 32
    receipt = contour_tree.build_contour_tree(
        values, valid, wall, edges, edge_valid, sigma
    )
    jax.block_until_ready(receipt)
    assert int(receipt.node_vertex.size) == vertex_count
    assert not bool(receipt.overflow), (
        f"the safe edge-count bound refused a degree-{degree} graph on "
        f"{int(edges.shape[0])} edges"
    )
