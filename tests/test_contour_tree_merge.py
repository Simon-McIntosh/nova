"""Synthetic topology contracts for the fixed-capacity contour-tree sweep."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.contour_tree import build_contour_tree
from nova.jax.config import configure_dtypes


configure_dtypes()


def _tree(values, edges, wall=()):
    """Build one compact graph with all supplied vertices live."""

    values = jnp.asarray(values, dtype=jnp.float64)
    edge_array = jnp.asarray(edges, dtype=jnp.int32)
    wall_mask = (
        jnp.zeros(values.size, dtype=bool)
        .at[jnp.asarray(wall, dtype=jnp.int32)]
        .set(True)
    )
    return build_contour_tree(
        values,
        jnp.ones(values.size, dtype=bool),
        wall_mask,
        edge_array,
        jnp.ones(edge_array.shape[0], dtype=bool),
        jnp.asarray(1, dtype=jnp.int32),
    )


def _nodes(result):
    """Return carrier vertex to DD critical-type mapping."""

    vertex = np.asarray(result.node_vertex)[np.asarray(result.node_valid)]
    kind = np.asarray(result.critical_type)[np.asarray(result.node_valid)]
    return dict(zip(vertex.tolist(), kind.tolist(), strict=True))


def _superlevel_components(values, edges, level):
    """Count strict superlevel components directly on the carrier graph."""

    active = {index for index, value in enumerate(values) if value > level}
    remaining = set(active)
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            source = frontier.pop()
            for left, right in edges:
                target = right if left == source else left if right == source else None
                if target in remaining:
                    remaining.remove(target)
                    frontier.append(target)
    return count


def test_single_maximum_has_one_contour_arc():
    """A monotone carrier has only its maximum and minimum as critical nodes."""

    result = _tree([3.0, 2.0, 1.0], [(0, 1), (1, 2)])

    assert _nodes(result) == {0: 2, 2: 0}
    assert int(np.sum(result.node_valid) - np.sum(result.edge_valid)) == 1
    assert np.asarray(result.edges)[np.asarray(result.edge_valid)].tolist() == [[0, 2]]


def test_regular_triangle_cycle_does_not_create_a_saddle():
    """A regular triangular carrier stays one arc instead of duplicating its join."""

    result = _tree(
        [4.0, 3.0, 2.0, 1.0],
        [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)],
    )

    assert _nodes(result) == {0: 2, 3: 0}
    assert int(np.sum(result.node_valid) - np.sum(result.edge_valid)) == 1
    assert np.asarray(result.edges)[np.asarray(result.edge_valid)].tolist() == [[0, 3]]


def test_join_saddle_matches_superlevel_component_event():
    """Two maxima join at the only saddle in a Y-shaped carrier."""

    values = [4.0, 3.0, 2.0, 0.0]
    edges = [(0, 2), (1, 2), (2, 3)]
    result = _tree(values, edges)

    assert _nodes(result) == {0: 2, 1: 2, 2: 1, 3: 0}
    assert int(np.sum(result.node_valid) - np.sum(result.edge_valid)) == 1
    assert _superlevel_components(values, edges, 2.0) == 2
    assert _superlevel_components(values, edges, np.nextafter(2.0, -np.inf)) == 1


def test_plateau_uses_vertex_order_as_simulation_of_simplicity():
    """Equal-valued adjacent vertices retain one deterministic maximum."""

    result = _tree([3.0, 3.0, 1.0], [(0, 1), (1, 2)])

    assert _nodes(result) == {0: 2, 1: 1, 2: 0}
    assert int(np.sum(result.node_valid) - np.sum(result.edge_valid)) == 1


def test_first_wall_vertex_is_a_join_with_outside():
    """The highest wall vertex becomes the explicit wall-contact event."""

    result = _tree([3.0, 2.0, 0.0], [(0, 1), (1, 2)], wall=(1,))

    assert _nodes(result) == {0: 2, 1: 1, 2: 0}
    assert int(np.sum(result.node_valid) - np.sum(result.edge_valid)) == 1


def test_each_wall_region_joins_outside_at_its_own_wall_vertex():
    """Two disconnected wall-bearing regions give one outside-join each."""

    result = _tree(
        [3.0, 1.0, 0.0, 3.0, 1.0, 0.0],
        [(0, 1), (1, 2), (3, 4), (4, 5)],
        wall=(1, 4),
    )
    nodes = _nodes(result)

    assert sorted(v for v, kind in nodes.items() if kind == 1) == [1, 4]


def test_batched_tree_is_identical_to_per_field_tree():
    """The fixed-shape receipt is a valid vmap result, not a host fallback."""

    values = jnp.asarray([[4.0, 3.0, 2.0, 0.0], [5.0, 4.0, 1.0, -1.0]])
    edges = jnp.asarray([[0, 2], [1, 2], [2, 3]], dtype=jnp.int32)
    valid = jnp.ones(4, dtype=bool)
    wall = jnp.zeros(4, dtype=bool)
    edge_valid = jnp.ones(3, dtype=bool)
    batched = jax.vmap(build_contour_tree, in_axes=(0, None, None, None, None, None))(
        values, valid, wall, edges, edge_valid, jnp.asarray(1, dtype=jnp.int32)
    )
    for index in range(values.shape[0]):
        direct = build_contour_tree(values[index], valid, wall, edges, edge_valid, 1)
        for name in (
            "node_vertex",
            "node_psi",
            "node_valid",
            "critical_type",
            "edges",
            "edge_valid",
            "overflow",
        ):
            assert np.array_equal(
                np.asarray(getattr(batched, name)[index]),
                np.asarray(getattr(direct, name)),
            )


def test_capacity_overflow_is_reported_not_truncated():
    """A carrier exceeding the DD node capacity carries an explicit refusal bit."""

    count = 257
    values = jnp.arange(count, 0, -1, dtype=jnp.float64)
    edges = jnp.stack((jnp.arange(count - 1), jnp.arange(1, count)), axis=1)
    result = build_contour_tree(
        values,
        jnp.ones(count, dtype=bool),
        jnp.zeros(count, dtype=bool),
        edges.astype(jnp.int32),
        jnp.ones(count - 1, dtype=bool),
        jnp.asarray(1, dtype=jnp.int32),
    )

    assert bool(result.overflow)
