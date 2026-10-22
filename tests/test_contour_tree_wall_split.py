"""Wall-connected sublevel events on the triangulated carrier."""

import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.contour_tree import (
    _carrier_adjacency,
    _sweep_tree,
    build_contour_tree,
)
from nova.jax.config import configure_dtypes


configure_dtypes()


def _carrier(field):
    coordinates = np.linspace(-1.0, 1.0, 21)
    points = np.array([(x, y) for y in coordinates for x in coordinates])
    edges = set()
    for row in range(20):
        for column in range(20):
            lower = row * 21 + column
            for source, target in (
                (lower, lower + 1),
                (lower, lower + 21),
                (lower, lower + 22),
                (lower + 1, lower + 22),
                (lower + 21, lower + 22),
            ):
                edges.add((source, target))
    wall = np.array([abs(x) == 1 or abs(y) == 1 for x, y in points], dtype=bool)
    return np.asarray(field(points), dtype=np.float64), np.asarray(sorted(edges)), wall


def _split(values, edges, wall):
    edge_array = jnp.asarray(edges, dtype=jnp.int32)
    adjacency, valid, overflow = _carrier_adjacency(
        edge_array, jnp.ones(len(edges), dtype=bool), len(values)
    )
    assert not bool(overflow)
    return _sweep_tree(
        jnp.asarray(values),
        jnp.ones(len(values), dtype=bool),
        jnp.asarray(wall),
        adjacency,
        valid,
        False,
        len(values),
        len(edges),
    )


def _sublevel_count(active, edges, wall):
    outside = len(wall)
    graph = [[] for _ in range(outside + 1)]
    for source, target in edges:
        graph[source].append(target)
        graph[target].append(source)
    for vertex in np.flatnonzero(wall):
        graph[outside].append(int(vertex))
        graph[vertex].append(outside)
    remaining = {int(vertex) for vertex in active} | {outside}
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            for target in graph[pending.pop()]:
                if target in remaining:
                    remaining.remove(target)
                    pending.append(target)
    return count


@pytest.mark.parametrize("ripple", [0.0, 0.05])
def test_hill_wall_vertices_are_regular_in_split_sweep(ripple):
    def field(points):
        radius_squared = np.sum(points**2, axis=1)
        angle = np.arctan2(points[:, 1], points[:, 0])
        return 1 - radius_squared + ripple * np.cos(4 * angle)

    values, edges, wall = _carrier(field)
    split = _split(values, edges, wall)
    assert not np.any(np.asarray(split[0]) & wall)
    padded_values = np.append(values, 0.0)
    padded_wall = np.append(wall, False)
    receipt = build_contour_tree(
        jnp.asarray(padded_values),
        jnp.asarray(np.append(np.ones(len(values), dtype=bool), False)),
        jnp.asarray(padded_wall),
        jnp.asarray(edges, dtype=jnp.int32),
        jnp.ones(len(edges), dtype=bool),
        jnp.asarray(1),
    )
    node_vertices = np.asarray(receipt.node_vertex)[np.asarray(receipt.node_valid)]
    node_types = np.asarray(receipt.critical_type)[np.asarray(receipt.node_valid)]
    assert not bool(receipt.overflow)
    assert np.count_nonzero(padded_wall[node_vertices]) == 1
    assert np.count_nonzero((node_types == 0) & padded_wall[node_vertices]) == 0


def test_two_hills_keep_interior_saddle():
    values, edges, wall = _carrier(
        lambda points: np.maximum(
            1 - (points[:, 0] + 0.5) ** 2 - points[:, 1] ** 2,
            1 - (points[:, 0] - 0.5) ** 2 - points[:, 1] ** 2,
        )
    )
    split = _split(values, edges, wall)
    assert not np.any(np.asarray(split[0]) & wall)
    receipt = build_contour_tree(
        jnp.asarray(values),
        jnp.ones(len(values), dtype=bool),
        jnp.asarray(wall),
        jnp.asarray(edges, dtype=jnp.int32),
        jnp.ones(len(edges), dtype=bool),
        jnp.asarray(1),
    )
    vertices = np.asarray(receipt.node_vertex)[np.asarray(receipt.node_valid)]
    types = np.asarray(receipt.critical_type)[np.asarray(receipt.node_valid)]
    assert np.any((types == 1) & ~wall[vertices])


@pytest.mark.parametrize("ripple", [0.0, 0.05])
def test_split_events_match_outside_augmented_sublevel_counts(ripple):
    values, edges, wall = _carrier(
        lambda points: (
            1
            - np.sum(points**2, axis=1)
            + ripple * np.cos(4 * np.arctan2(points[:, 1], points[:, 0]))
        )
    )
    nodes, kinds, _, _, overflow, order = _split(values, edges, wall)
    assert not bool(overflow)
    order = np.asarray(order)
    nodes = np.asarray(nodes)
    kinds = np.asarray(kinds)
    active = set()
    previous = 1
    for vertex in order:
        active.add(int(vertex))
        current = _sublevel_count(active, edges, wall)
        if current > previous:
            assert nodes[vertex] and kinds[vertex] == 0
        elif current < previous:
            assert nodes[vertex] and kinds[vertex] == 1
        elif vertex != order[-1]:
            assert not nodes[vertex]
        previous = current
