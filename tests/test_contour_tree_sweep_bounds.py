"""Keep contour-tree receipts stable and sweep parent walks shallow."""

from __future__ import annotations

from math import ceil, log2

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.contour_tree_brute_force import (
    certificate_fixtures,
    certificate_rung_fixtures,
    compare,
    mast_fixtures,
)
from nova.equilibrium import contour_tree
from nova.jax.config import configure_dtypes
from scripts.prototypes.contour_tree_sweep_profile.profile import (
    inputs,
    instrumented_sweep,
)
from tests.test_contour_tree_wall_split import _sublevel_count


configure_dtypes()
assert jax.config.jax_enable_x64 is True


def test_fixture_events_match_independent_component_counts() -> None:
    """Every carrier event agrees with independent graph connectivity counts."""

    fixtures = (
        *certificate_fixtures(),
        *certificate_rung_fixtures(),
        *mast_fixtures(),
    )
    assert fixtures
    for fixture in fixtures:
        mesh = fixture.mesh
        comparison = compare(mesh)
        rows = comparison["rows"]
        assert rows and any(row["brute_force"] > 0 for row in rows)
        mismatches = [row for row in rows if row["brute_force"] != row["tree"]]
        assert not mismatches, (fixture.name, mismatches[:3])

        values = np.asarray(mesh.vertex_psi)
        valid = np.asarray(mesh.vertex_valid, dtype=bool)
        wall = np.asarray(mesh.vertex_is_wall, dtype=bool) & valid
        edges = np.asarray(mesh.edges)[np.asarray(mesh.edge_valid)]
        neighbours, neighbour_valid, adjacency_overflow = (
            contour_tree._carrier_adjacency(mesh.edges, mesh.edge_valid, len(values))
        )
        assert not bool(adjacency_overflow)
        nodes, kinds, _, _, overflow, order = contour_tree._sweep_tree(
            mesh.vertex_psi,
            mesh.vertex_valid,
            mesh.vertex_is_wall,
            neighbours,
            neighbour_valid,
            False,
            len(values),
            len(mesh.edges),
        )
        assert not bool(overflow)
        nodes = np.asarray(nodes)
        kinds = np.asarray(kinds)
        usable_order = [int(v) for v in np.asarray(order) if valid[v]]
        assert usable_order
        active: set[int] = set()
        previous = 1  # The outside component exists before the first wall visit.
        for vertex in usable_order:
            active.add(vertex)
            current = _sublevel_count(active, edges, wall)
            if current > previous:
                assert nodes[vertex] and kinds[vertex] == 0, (fixture.name, vertex)
            elif current < previous:
                assert nodes[vertex] and kinds[vertex] == 1, (fixture.name, vertex)
            elif vertex != usable_order[-1]:
                assert not nodes[vertex], (fixture.name, vertex)
            previous = current
        print(
            f"ORACLE_EVENTS={fixture.name}:"
            f"superlevels={len(rows)}:sublevels={len(usable_order)}"
        )
    print(f"CHECKED_FIXTURES={len(fixtures)}")


def test_largest_profile_fixture_bounds_root_hops() -> None:
    """Both sweep directions stay within the rank-balanced tree depth."""

    arguments, realised_cells = inputs("limited", 10_000)
    vertex_psi, vertex_valid, vertex_is_wall, edges, edge_valid, _ = arguments
    vertex_count = int(vertex_psi.size)
    neighbours, neighbour_valid, _ = contour_tree._carrier_adjacency(
        edges, edge_valid, vertex_count
    )
    bound = ceil(log2(vertex_count)) + 1

    for name, descending in (("join", True), ("split", False)):
        sweep_arguments = (
            jnp.asarray(vertex_psi),
            jnp.asarray(vertex_valid),
            jnp.asarray(vertex_is_wall),
            neighbours,
            neighbour_valid,
            descending,
            vertex_count,
            int(edges.shape[0]),
        )
        _, depth = instrumented_sweep(contour_tree, sweep_arguments)
        assert depth["visits"] == vertex_count
        assert depth["roots_max"] <= bound, (
            f"{name} sweep root-hop depth {depth['roots_max']} exceeds "
            f"ceil(log2({vertex_count})) + 1 = {bound} "
            f"for {realised_cells} realised cells"
        )
        print(
            f"ROOT_HOPS={name}:{depth['roots_max']}/{bound} "
            f"VERTICES={vertex_count} CELLS={realised_cells}"
        )
