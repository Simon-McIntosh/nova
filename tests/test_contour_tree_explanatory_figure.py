"""The figure's drawn nodes are exactly the computed tree's valid nodes.

The explanatory figure is only as honest as its agreement with the receipt it
is drawn from, so this test rebuilds the tree independently and asserts the node
set the figure draws matches it in count and carrier vertices.  With
``CONTOUR_TREE_FIGURE_DROP_NODE`` set to a node row, the figure's drawn set
drops that row and the guard must fail; that is the declared mutation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from benchmarks.contour_tree_explanatory_figure import _fixture, build_state
from nova.equilibrium.contour_tree import build_contour_tree
from nova.jax.config import configure_dtypes


configure_dtypes()


def _receipt():
    """Rebuild the certificate tree from the fixture, independently of the figure."""

    fixture = _fixture()
    mesh = fixture.mesh
    tree = build_contour_tree(
        mesh.vertex_psi,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert not bool(tree.overflow)
    return mesh, tree


def test_figure_draws_exactly_the_receipts_valid_nodes():
    """Count and carrier vertices of the drawn set equal the receipt's valid nodes."""

    _, tree = _receipt()
    state = build_state()

    valid = np.asarray(tree.node_valid)
    node_vertex = np.asarray(tree.node_vertex)
    expected = sorted(int(node_vertex[row]) for row in np.flatnonzero(valid))
    drawn = sorted(int(node.carrier_vertex) for node in state.nodes)

    assert len(drawn) == len(expected)
    assert drawn == expected


def test_receipt_tree_has_one_more_node_than_edge():
    """The tree the figure is drawn from keeps the contour-tree invariant."""

    _, tree = _receipt()
    node_count = int(np.count_nonzero(np.asarray(tree.node_valid)))
    edge_count = int(np.count_nonzero(np.asarray(tree.edge_valid)))
    assert node_count - edge_count == 1