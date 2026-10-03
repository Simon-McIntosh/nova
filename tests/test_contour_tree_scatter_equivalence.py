"""Product-level tests for the sweep's duplicate-root combine.

The sweep folds one contribution per carrier neighbour slot into the visited
vertex's parent array. A vertex reached through several edges supplies its root
to that array more than once, so the combine is a scatter with duplicate indices
and resolves last-write-wins. It is deterministic only because every duplicate
slot writes the same value: an active slot writes the visited vertex, and an
inactive slot writes its own root's unchanged parent. The receipt is therefore
determined by the carrier graph, not by the order the carrier stores its edges.

The tests exercise the product sweep, not a re-implemented scatter. Permuting
the carrier edge storage order permutes each vertex's neighbour slots without
changing any vertex's neighbour set, so neither the swept tree nor the receipt's
node fields may move. A mutation that makes the inactive slot write a value
other than the root's parent breaks the invariance, and is the declared negative
control.

The join and split arc sets are compared as sets: the sweep emits arcs in
neighbour-slot order, so a slot permutation permutes the arc rows while leaving
the arc set unchanged. The merged receipt's arc set is deliberately not asserted
here; on the MAST carrier the arc set is not invariant to edge order, and that
dependence is introduced after the sweep, by the split-node subdivision, not by
the duplicate-root combine these tests pin.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.contour_tree_brute_force import (
    certificate_rung_fixtures,
    mast_fixtures,
)
from nova.equilibrium.contour_tree import (
    _carrier_adjacency,
    _sweep_tree,
    build_contour_tree,
)
from nova.jax.config import configure_dtypes


configure_dtypes()
assert jax.config.jax_enable_x64 is True

#: Receipt fields that must be bit-identical under an edge-order permutation.
#: ``edges`` is excluded: it is compared as a set where asserted at all.
_NODE_FIELDS = (
    "node_vertex",
    "node_psi",
    "node_valid",
    "critical_type",
    "overflow",
)


def _receipt(mesh, edges=None, edge_valid=None):
    """Build one receipt, optionally from a permuted copy of the carrier edges."""

    edges = mesh.edges if edges is None else edges
    edge_valid = mesh.edge_valid if edge_valid is None else edge_valid
    return build_contour_tree(
        mesh.vertex_psi,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        jnp.asarray(edges, dtype=jnp.int32),
        jnp.asarray(edge_valid, dtype=bool),
        jnp.asarray(1, dtype=jnp.int32),
    )


def _sweep(mesh, edges=None, edge_valid=None, descending=True):
    """Sweep one merge tree from the carrier adjacency, as the product does."""

    edges = mesh.edges if edges is None else edges
    edge_valid = mesh.edge_valid if edge_valid is None else edge_valid
    vertex_count = int(mesh.vertex_psi.size)
    edge_capacity = int(edges.shape[0])
    neighbours, neighbour_valid, _ = _carrier_adjacency(
        jnp.asarray(edges, dtype=jnp.int32),
        jnp.asarray(edge_valid, dtype=bool),
        vertex_count,
    )
    nodes, node_type, out_edges, out_valid, overflow, _ = _sweep_tree(
        jnp.asarray(mesh.vertex_psi, dtype=jnp.float64),
        jnp.asarray(mesh.vertex_valid, dtype=bool),
        jnp.asarray(mesh.vertex_is_wall, dtype=bool),
        neighbours,
        neighbour_valid,
        descending,
        vertex_count,
        edge_capacity,
    )
    return nodes, node_type, out_edges, out_valid, overflow


def _canonical_arcs(edges, edge_valid) -> np.ndarray:
    """The live arcs as a sorted set, independent of append order."""

    rows = np.asarray(edges)[np.asarray(edge_valid)]
    if rows.size == 0:
        return rows.reshape(0, 2)
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


@pytest.fixture(scope="module")
def carrier_meshes():
    """The certificate rungs and one MAST carrier the brute-force tests build."""

    meshes = [fixture.mesh for fixture in certificate_rung_fixtures()]
    meshes.append(mast_fixtures()[0].mesh)
    return meshes


def _permuted(mesh, seed):
    """A carrier edge order drawn from a fixed seed, with its validity row."""

    order = np.random.default_rng(seed).permutation(int(mesh.edges.shape[0]))
    return mesh.edges[order], mesh.edge_valid[order]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_receipt_nodes_are_invariant_to_carrier_edge_order(carrier_meshes, seed):
    """Reordering the carrier edges leaves every receipt node field unchanged.

    The edge order fed to the carrier adjacency fixes the column order of each
    vertex's neighbour slots, not the neighbour set. Reordering the edges
    reorders those slots, so a receipt assembled from the swept deduplicated
    roots must carry the same nodes: the same vertex identities, the same
    critical classifications, and the same adjacency overflow flag, bit for bit.
    """

    for mesh in carrier_meshes:
        reference = _receipt(mesh)
        edges, edge_valid = _permuted(mesh, seed)
        candidate = _receipt(mesh, edges, edge_valid)
        label = f"seed {seed}, {int(mesh.vertex_psi.size)} vertices"
        for name in _NODE_FIELDS:
            left = np.asarray(getattr(reference, name))
            right = np.asarray(getattr(candidate, name))
            assert left.shape == right.shape, f"{label}: {name} shape moved"
            assert left.tobytes() == right.tobytes(), f"{label}: {name} moved"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_swept_arc_sets_are_invariant_to_carrier_edge_order(carrier_meshes, seed):
    """Reordering the carrier edges leaves both swept trees' arcs unchanged.

    Each sweep emits an arc per distinct ``(component birth, visited vertex)``
    pair, drawn from the roots the duplicate-index scatter resolves. Column order
    permutes the rows those arcs land in but not the pairs themselves, so the
    join and split arc sets are invariant and the arc count is unchanged. This is
    the observable form of the combine's determinism: without it, a tree would
    depend on carrier storage order rather than on the graph.
    """

    for mesh in carrier_meshes:
        edges, edge_valid = _permuted(mesh, seed)
        label = f"seed {seed}, {int(mesh.vertex_psi.size)} vertices"
        for descending, arm in ((True, "join"), (False, "split")):
            reference = _sweep(mesh, descending=descending)
            candidate = _sweep(mesh, edges, edge_valid, descending=descending)
            reference_arcs = _canonical_arcs(reference[2], reference[3])
            candidate_arcs = _canonical_arcs(candidate[2], candidate[3])
            assert reference[0].tobytes() == candidate[0].tobytes(), (
                f"{label}: {arm} node set moved"
            )
            assert reference_arcs.shape == candidate_arcs.shape, (
                f"{label}: {arm} arc count moved"
            )
            assert reference_arcs.tobytes() == candidate_arcs.tobytes(), (
                f"{label}: {arm} arc set moved"
            )


def test_active_neighbours_sharing_one_root_sweep_to_one_arc():
    """Two active neighbours with one root combine without a spurious saddle.

    A triangle whose values fall with the vertex index is visited 0, 1, 2. When
    vertex 2 is visited last, its two neighbours 0 and 1 are already active and
    share the single root 1, so the duplicate-index scatter writes that root
    twice with the same value and the component count stays one. The receipt's
    node types and arc endpoints are the swept births and parents made
    observable: the maximum sits at vertex 0, the minimum at vertex 2, and a
    single arc joins them, with no saddle at the shared-root vertex.
    """

    values = jnp.asarray([2.0, 1.0, 0.0], dtype=jnp.float64)
    edges = jnp.asarray([[0, 1], [0, 2], [1, 2]], dtype=jnp.int32)
    result = build_contour_tree(
        values,
        jnp.ones(3, dtype=bool),
        jnp.zeros(3, dtype=bool),
        edges,
        jnp.ones(3, dtype=bool),
        jnp.asarray(1, dtype=jnp.int32),
    )

    vertex = np.asarray(result.node_vertex)[np.asarray(result.node_valid)]
    kind = np.asarray(result.critical_type)[np.asarray(result.node_valid)]
    arcs = np.asarray(result.edges)[np.asarray(result.edge_valid)]

    assert vertex.tolist() == [0, 2]
    assert kind.tolist() == [2, 0]
    assert arcs.tolist() == [[0, 1]]
