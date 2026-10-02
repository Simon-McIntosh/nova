"""Contour-tree topology conventions for equilibrium states.

Topology orders the signed, raw poloidal flux ``sigma * psi`` in webers, where
``sigma`` is derived from the state's declared COCOS sign tuple and the sign of
its plasma current through :mod:`nova.io.cocos`; the magnetic axis is therefore
a maximum of the signed flux.  Normalised flux is never a topology operand.

The domain is the interior of the multi-unit vessel polygon under true polygon
containment, with that polygon as its boundary.  The carrier is the hex plasma
cells: this read never resamples onto a raster.  Results map to DD 4.1.0 under
COCOS 17 as ``contour_tree.node`` and ``contour_tree.edges`` (with
``critical_type`` on raw ``psi``), ``boundary.type``, ``boundary.psi``,
``global_quantities.psi_magnetic_axis``, and ``boundary.closest_wall_point``.
The virtual outside node is not a critical point of the module contract: it
carries :data:`CRITICAL_TYPE_OUTSIDE`, outside the DD minimum/saddle/maximum
vocabulary, and :func:`dd_emittable_nodes` excludes it from the DD
``contour_tree`` mapping.

The fixed capacities are 256 nodes and 255 edges.  A capacity overflow is a
visible refusal rather than a truncated topology result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Protocol, TypeAlias

import jax
import jax.numpy as jnp

from nova.io.cocos import convention

ContourTreeNode: TypeAlias = tuple[int, float, float, float]
"""One DD node as ``(critical_type, radius, height, psi)`` on raw flux."""

ContourTreeEdge: TypeAlias = tuple[int, int]
"""One pair of indices into :attr:`ContourTreeResult.node`."""

CRITICAL_TYPE_OUTSIDE: int = 3
"""Type of the virtual outside node: not a minimum (0), saddle (1) or maximum (2).

The wall-contact receiver that carries the outside is a bookkeeping node, not a
critical point of the field, so it is typed outside the DD critical vocabulary
and :func:`dd_emittable_nodes` drops it from the ``contour_tree`` mapping.
"""


class FluxCurrentSignError(ValueError):
    """The current sign and raw flux ordering disagree with declared COCOS."""


class ContourTreeState(Protocol):
    """State fields required to orient a contour-tree topology read."""

    cocos: int
    plasma_current: float
    psi_magnetic_axis: float
    boundary_psi: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ContourTreeArrays:
    """Fixed-capacity, accelerator-native contour-tree receipt.

    Node rows retain their carrier-vertex slots: ``node_vertex`` identifies the
    piecewise-linear vertex, ``node_psi`` is raw flux in webers, and
    ``critical_type`` uses the DD convention (minimum 0, saddle 1, maximum 2),
    except for the virtual outside node which carries
    :data:`CRITICAL_TYPE_OUTSIDE` and is dropped from the DD ``contour_tree``
    mapping by :func:`dd_emittable_nodes`.
    ``node_valid`` and ``edge_valid`` make every array fixed shaped.  Edges
    refer to compact node rows; ``node_vertex`` maps each valid node row back to
    its carrier vertex.  ``overflow`` means that a candidate could not fit in
    the declared DD capacities; callers must refuse that receipt instead of
    treating the prefix as a tree.
    """

    node_vertex: jax.Array
    node_psi: jax.Array
    node_valid: jax.Array
    critical_type: jax.Array
    edges: jax.Array
    edge_valid: jax.Array
    overflow: jax.Array


def _neighbour_roots(parents: jax.Array, neighbours: jax.Array) -> jax.Array:
    """Follow only the adjacency entries required by one sweep visit."""

    def unresolved(roots: jax.Array) -> jax.Array:
        return jnp.any(roots != parents[roots])

    return jax.lax.while_loop(unresolved, lambda roots: parents[roots], neighbours)


def _append_edges(
    edges: jax.Array,
    edge_valid: jax.Array,
    overflow: jax.Array,
    sources: jax.Array,
    source_valid: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Append distinct undirected arcs in bounded slots without truncation."""

    capacity = edges.shape[0]

    def append(index, state):
        table, valid, overflow_bit = state
        source = sources[index]
        destination = target if target.ndim == 0 else target[index]
        requested = source_valid[index] & (source != destination)
        arc = jnp.sort(jnp.stack((source, destination)))
        duplicate = jnp.any(valid & jnp.all(table == arc, axis=1))
        slot = jnp.argmin(jnp.where(valid, capacity, jnp.arange(capacity)))
        has_slot = jnp.any(~valid)
        write = requested & ~duplicate & has_slot
        table = table.at[slot].set(jnp.where(write, arc, table[slot]))
        valid = valid.at[slot].set(valid[slot] | write)
        return table, valid, overflow_bit | (requested & ~duplicate & ~has_slot)

    return jax.lax.fori_loop(0, sources.size, append, (edges, edge_valid, overflow))


def _insert_split_nodes(
    node_valid: jax.Array,
    edges: jax.Array,
    edge_valid: jax.Array,
    overflow: jax.Array,
    split_only: jax.Array,
    values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Subdivide join arcs at split-only critical vertices.

    A join tree already preserves every superlevel component event.  The split
    tree contributes critical vertices that lie in the interior of those arcs;
    subdividing an arc preserves its level-crossing count while retaining the
    additional critical node.  In contrast, appending the split arc creates a
    cycle whenever both trees describe the same contour region.
    """

    capacity = edges.shape[0]
    edge_indices = jnp.arange(capacity, dtype=jnp.int32)

    def above(left: jax.Array, right: jax.Array) -> jax.Array:
        return (values[left] > values[right]) | (
            (values[left] == values[right]) & (left < right)
        )

    def insert(vertex: jax.Array, state):
        nodes, table, valid, overflow_bit = state
        source = table[:, 0]
        target = table[:, 1]
        crosses = valid & (above(source, vertex) != above(target, vertex))
        arc_slot = jnp.argmax(crosses).astype(jnp.int32)
        free_slot = jnp.argmin(jnp.where(valid, capacity, edge_indices)).astype(
            jnp.int32
        )
        has_arc = jnp.any(crosses)
        has_slot = jnp.any(~valid)
        requested = split_only[vertex]
        write = requested & has_arc & has_slot
        old = table[arc_slot]
        table = table.at[arc_slot].set(
            jnp.where(write, jnp.stack((old[0], vertex)), old)
        )
        table = table.at[free_slot].set(
            jnp.where(write, jnp.stack((vertex, old[1])), table[free_slot])
        )
        valid = valid.at[free_slot].set(valid[free_slot] | write)
        nodes = nodes.at[vertex].set(nodes[vertex] | write)
        return nodes, table, valid, overflow_bit | (requested & ~write)

    return jax.lax.fori_loop(
        0, values.size, insert, (node_valid, edges, edge_valid, overflow)
    )


def _carrier_adjacency(
    edges: jax.Array, edge_valid: jax.Array, vertex_count: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Index each carrier edge once for constant-degree sweep visits.

    Three-direction carriers have bounded vertex degree; ordinary sparse graphs
    retain their full edge-width neighbourhood so the public graph contract is
    unchanged.  An unexpected carrier degree is a visible capacity refusal.
    """

    edge_count = edges.shape[0]
    neighbour_capacity = 16 if edge_count > 2 * vertex_count else edge_count
    neighbours = jnp.zeros((vertex_count, neighbour_capacity), dtype=jnp.int32)
    neighbour_valid = jnp.zeros((vertex_count, neighbour_capacity), dtype=bool)
    counts = jnp.zeros(vertex_count, dtype=jnp.int32)
    overflow = jnp.asarray(False)

    def append(edge_index: int, state):
        table, valid, counts, overflow = state
        left, right = edges[edge_index]
        write = edge_valid[edge_index]
        left_slot = counts[left]
        right_slot = counts[right]
        left_fits = left_slot < neighbour_capacity
        right_fits = right_slot < neighbour_capacity
        left_slot = jnp.minimum(left_slot, neighbour_capacity - 1)
        right_slot = jnp.minimum(right_slot, neighbour_capacity - 1)
        table = table.at[left, left_slot].set(
            jnp.where(write & left_fits, right, table[left, left_slot])
        )
        table = table.at[right, right_slot].set(
            jnp.where(write & right_fits, left, table[right, right_slot])
        )
        valid = valid.at[left, left_slot].set(
            valid[left, left_slot] | (write & left_fits)
        )
        valid = valid.at[right, right_slot].set(
            valid[right, right_slot] | (write & right_fits)
        )
        counts = counts.at[left].add(write)
        counts = counts.at[right].add(write)
        return table, valid, counts, overflow | (write & ~(left_fits & right_fits))

    neighbours, neighbour_valid, _, overflow = jax.lax.fori_loop(
        0, edge_count, append, (neighbours, neighbour_valid, counts, overflow)
    )
    return neighbours, neighbour_valid, overflow


def _sweep_tree(
    values: jax.Array,
    vertex_valid: jax.Array,
    vertex_is_wall: jax.Array,
    neighbours: jax.Array,
    neighbour_valid: jax.Array,
    descending: bool,
    node_capacity: int,
    edge_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sweep one merge tree and emit extrema, saddles, and bounded arcs.

    A lexicographic ordering of value then carrier index is simulation of
    simplicity: values can be exactly equal without creating an ambiguous
    union.  Every wall vertex is joined to one virtual outside component, so
    each wall-bearing region meets the outside at its own highest wall vertex,
    the first one reached as the level descends.  A wall vertex whose active
    neighbour components already carry a wall is not a new contact, so one
    region is recorded once however many wall vertices it holds as the sweep
    reaches them.  The contact is represented by that real wall slot; the
    virtual node is deliberately not emitted as a DD critical point.
    """

    vertex_count = values.size
    neighbour_count = neighbours.shape[1]
    index = jnp.arange(vertex_count, dtype=jnp.int32)
    order = jnp.lexsort((index, -values if descending else values))
    last_usable_rank = jnp.max(
        jnp.where(vertex_valid[order], index, jnp.asarray(-1, dtype=jnp.int32))
    )
    parents = index
    active = jnp.zeros(vertex_count, dtype=bool)
    births = index
    node_valid = jnp.zeros(node_capacity, dtype=bool)
    node_type = jnp.full(node_capacity, -1, dtype=jnp.int32)
    output_edges = jnp.zeros((edge_capacity, 2), dtype=jnp.int32)
    output_valid = jnp.zeros(edge_capacity, dtype=bool)
    overflow = jnp.asarray(False)
    wall_flag = jnp.zeros(vertex_count, dtype=bool)

    def visit(rank, state):
        (
            parents,
            active,
            births,
            node_valid,
            node_type,
            output_edges,
            output_valid,
            overflow,
            wall_flag,
        ) = state
        vertex = order[rank].astype(jnp.int32)
        usable = vertex_valid[vertex]
        adjacent = neighbour_valid[vertex]
        neighbour = jnp.where(adjacent, neighbours[vertex], vertex)
        neighbour_root = _neighbour_roots(parents, neighbour)
        root_active = adjacent & active[neighbour]
        earlier = jnp.arange(neighbour_count)[:, None] < jnp.arange(neighbour_count)
        repeated = jnp.any(
            earlier
            & root_active[:, None]
            & root_active[None, :]
            & (neighbour_root[:, None] == neighbour_root[None, :]),
            axis=0,
        )
        representative = root_active & ~repeated
        neighbour_wall = jnp.any(representative & wall_flag[neighbour_root])
        component_count = jnp.sum(representative, dtype=jnp.int32)
        first_wall = usable & descending & vertex_is_wall[vertex] & ~neighbour_wall
        final_vertex = usable & (rank == last_usable_rank)
        extremum = component_count == 0
        saddle = component_count >= 2
        terminal = final_vertex & (component_count == 1)
        event = usable & (extremum | saddle | terminal | first_wall)
        signed_maximum = (descending & extremum) | ((not descending) & terminal)
        signed_minimum = ((not descending) & extremum) | (descending & terminal)
        critical = jnp.where(
            saddle | first_wall,
            1,
            jnp.where(signed_maximum, 2, jnp.where(signed_minimum, 0, 1)),
        ).astype(jnp.int32)
        node_valid = node_valid.at[vertex].set(node_valid[vertex] | event)
        node_type = node_type.at[vertex].set(
            jnp.where(event, critical, node_type[vertex])
        )
        connect = usable & (saddle | terminal | first_wall)
        sources = births[neighbour_root]
        output_edges, output_valid, overflow = _append_edges(
            output_edges,
            output_valid,
            overflow,
            sources,
            representative & connect,
            vertex,
        )
        parents = parents.at[neighbour_root].set(
            jnp.where(root_active & usable, vertex, parents[neighbour_root])
        )
        parents = parents.at[vertex].set(vertex)
        first_root = jnp.argmax(representative).astype(jnp.int32)
        inherited = births[neighbour_root[first_root]]
        births = births.at[vertex].set(
            jnp.where(extremum | saddle | first_wall, vertex, inherited)
        )
        active = active.at[vertex].set(usable)
        wall_flag = wall_flag.at[vertex].set(
            usable & (vertex_is_wall[vertex] | neighbour_wall)
        )
        return (
            parents,
            active,
            births,
            node_valid,
            node_type,
            output_edges,
            output_valid,
            overflow,
            wall_flag,
        )

    result = jax.lax.fori_loop(
        0,
        vertex_count,
        visit,
        (
            parents,
            active,
            births,
            node_valid,
            node_type,
            output_edges,
            output_valid,
            overflow,
            wall_flag,
        ),
    )
    return result[3], result[4], result[5], result[6], result[7], order


@jax.jit
def build_contour_tree(
    vertex_psi: jax.Array,
    vertex_valid: jax.Array,
    vertex_is_wall: jax.Array,
    edges: jax.Array,
    edge_valid: jax.Array,
    sigma: jax.Array,
) -> ContourTreeArrays:
    """Build a contour tree from a fixed-capacity piecewise-linear edge graph.

    The join sweep is over descending ``sigma * psi`` and the split sweep is
    ascending.  Their critical slots and arcs are merged into one acyclic
    receipt.  Equal raw values are ordered by vertex index, the required
    simulation-of-simplicity rule, so this function has no value tolerance and
    remains traceable through both :func:`jax.jit` and :func:`jax.vmap`.
    """

    node_capacity = ContourTreeResult.node_capacity
    edge_capacity = ContourTreeResult.edge_capacity
    mesh_node_capacity = vertex_psi.size
    mesh_edge_capacity = edges.shape[0]
    signed = jnp.asarray(sigma, dtype=vertex_psi.dtype) * vertex_psi
    neighbours, neighbour_valid, adjacency_overflow = _carrier_adjacency(
        edges, edge_valid, mesh_node_capacity
    )
    join = _sweep_tree(
        signed,
        vertex_valid,
        vertex_is_wall,
        neighbours,
        neighbour_valid,
        True,
        mesh_node_capacity,
        mesh_edge_capacity,
    )
    split = _sweep_tree(
        signed,
        vertex_valid,
        vertex_is_wall,
        neighbours,
        neighbour_valid,
        False,
        mesh_node_capacity,
        mesh_edge_capacity,
    )
    join_nodes, join_types, join_edges, join_valid, join_overflow, _ = join
    split_nodes, split_types, split_edges, split_valid, split_overflow, _ = split
    node_valid = join_nodes | split_nodes
    critical_type = jnp.where(join_nodes, join_types, split_types)
    critical_type = jnp.where(
        critical_type == 1,
        critical_type,
        jnp.where(sigma > 0, critical_type, 2 - critical_type),
    )
    split_only = split_nodes & ~join_nodes
    node_valid, merged_edges, merged_valid, overflow = _insert_split_nodes(
        node_valid,
        join_edges,
        join_valid,
        join_overflow | split_overflow,
        split_only,
        signed,
    )
    vertex_count = vertex_valid.size
    slots = jnp.arange(node_capacity, dtype=jnp.int32)
    join_order = join[-1]
    terminal_rank = jnp.max(
        jnp.where(
            vertex_valid[join_order],
            jnp.arange(vertex_count, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        )
    )
    terminal = join_order[terminal_rank]
    virtual = jnp.argmin(
        jnp.where(vertex_valid, vertex_count, jnp.arange(vertex_count, dtype=jnp.int32))
    )
    has_virtual = jnp.any(~vertex_valid)
    replace_terminal = has_virtual & join_nodes[terminal]
    node_valid = node_valid.at[terminal].set(
        jnp.where(replace_terminal, False, node_valid[terminal])
    )
    node_valid = node_valid.at[virtual].set(node_valid[virtual] | replace_terminal)
    critical_type = critical_type.at[virtual].set(
        jnp.where(replace_terminal, CRITICAL_TYPE_OUTSIDE, critical_type[virtual])
    )
    merged_edges = jnp.where(
        (merged_edges == terminal) & replace_terminal,
        virtual,
        merged_edges,
    )
    node_count = jnp.sum(node_valid, dtype=jnp.int32)
    node_source = jnp.nonzero(node_valid, size=node_capacity, fill_value=0)[0]
    compact_node_valid = slots < node_count
    edge_count = jnp.sum(merged_valid, dtype=jnp.int32)
    edge_source = jnp.nonzero(merged_valid, size=edge_capacity, fill_value=0)[0]
    compact_edge_valid = jnp.arange(edge_capacity) < edge_count
    carrier_vertex = jnp.arange(vertex_count, dtype=jnp.int32)
    node_matches = (
        node_source[:, None] == carrier_vertex[None, :]
    ) & compact_node_valid[:, None]
    vertex_node = jnp.argmax(node_matches, axis=0).astype(jnp.int32)
    compact_edges = vertex_node[merged_edges[edge_source]]
    edge_nodes_valid = jnp.all(
        jnp.any(node_matches, axis=0)[merged_edges[edge_source]], axis=1
    )
    overflow = (
        overflow
        | adjacency_overflow
        | (node_count > node_capacity)
        | (edge_count > edge_capacity)
    )
    overflow = overflow | jnp.any(compact_edge_valid & ~edge_nodes_valid)
    return ContourTreeArrays(
        node_vertex=node_source,
        node_psi=jnp.where(
            compact_node_valid,
            vertex_psi[node_source],
            jnp.zeros((), vertex_psi.dtype),
        ),
        node_valid=compact_node_valid,
        critical_type=jnp.where(compact_node_valid, critical_type[node_source], -1),
        edges=compact_edges,
        edge_valid=compact_edge_valid,
        overflow=overflow,
    )


def dd_emittable_nodes(result: ContourTreeArrays) -> jax.Array:
    """Select the receipt's node rows that are DD ``contour_tree`` critical points.

    The virtual outside node is not a critical point of the field: it carries
    :data:`CRITICAL_TYPE_OUTSIDE` and is excluded here, so a caller mapping the
    receipt to DD ``equilibrium.time_slice.contour_tree`` never emits it as a
    minimum, saddle or maximum.  The tree keeps the node and its arcs; only the
    DD mapping drops it.
    """

    return result.node_valid & (result.critical_type != CRITICAL_TYPE_OUTSIDE)


def _require_flux_current_consistency(
    state: ContourTreeState, *, sigma_bp: int, current_sign: int
) -> None:
    """Refuse a state whose raw axis-to-boundary ordering contradicts COCOS."""

    axis = float(state.psi_magnetic_axis)
    boundary = float(state.boundary_psi)
    difference = boundary - axis
    if not isfinite(axis) or not isfinite(boundary) or difference == 0.0:
        raise FluxCurrentSignError(
            "axis and boundary flux must be distinct finite webers"
        )
    flux_sign = 1 if difference > 0.0 else -1
    expected = current_sign * sigma_bp
    if flux_sign != expected:
        raise FluxCurrentSignError(
            "plasma current sign and raw flux ordering disagree with declared COCOS"
        )


@dataclass(frozen=True, slots=True)
class ContourTreeResult:
    """DD 4.1.0 topology fields emitted under COCOS 17.

    ``node`` contains raw-psi critical-point rows, and ``edges`` contains their
    index pairs.  The remaining fields map directly to the DD paths named in
    the module contract.
    """

    node: tuple[ContourTreeNode, ...]
    edges: tuple[ContourTreeEdge, ...]
    boundary_type: int
    boundary_psi: float
    psi_magnetic_axis: float
    closest_wall_point: tuple[float, float]

    node_capacity: ClassVar[int] = 256
    edge_capacity: ClassVar[int] = 255

    def __post_init__(self) -> None:
        """Refuse invalid DD values and fixed-capacity overflow visibly."""

        if len(self.node) > self.node_capacity:
            raise ValueError(f"contour tree has more than {self.node_capacity} nodes")
        if len(self.edges) > self.edge_capacity:
            raise ValueError(f"contour tree has more than {self.edge_capacity} edges")
        if self.boundary_type not in (0, 1):
            raise ValueError("boundary type must be 0 (limiter) or 1 (diverted)")
        if not isfinite(self.boundary_psi) or not isfinite(self.psi_magnetic_axis):
            raise ValueError("contour-tree flux values must be finite raw webers")
        if len(self.closest_wall_point) != 2 or not all(
            isfinite(value) for value in self.closest_wall_point
        ):
            raise ValueError("closest wall point must be one finite radius-height pair")
        for critical_type, radius, height, psi in self.node:
            if critical_type not in (0, 1, 2):
                raise ValueError(
                    "critical type must be 0 (minimum), 1 (saddle), or 2 (maximum)"
                )
            if not all(isfinite(value) for value in (radius, height, psi)):
                raise ValueError("contour-tree nodes must carry finite raw values")
        for start, end in self.edges:
            if not (0 <= start < len(self.node) and 0 <= end < len(self.node)):
                raise ValueError("contour-tree edges must reference declared node rows")


def sigma_for_state(state: ContourTreeState) -> int:
    """Return the signed-flux multiplier that makes the state axis a maximum."""

    declared = convention(state.cocos)
    current = float(state.plasma_current)
    if not isfinite(current) or current == 0.0:
        raise FluxCurrentSignError("plasma current must be finite and nonzero")
    current_sign = 1 if current > 0.0 else -1
    _require_flux_current_consistency(
        state, sigma_bp=declared.sigma_bp, current_sign=current_sign
    )
    return -current_sign * declared.sigma_bp


__all__ = [
    "CRITICAL_TYPE_OUTSIDE",
    "ContourTreeEdge",
    "ContourTreeArrays",
    "ContourTreeNode",
    "ContourTreeResult",
    "ContourTreeState",
    "FluxCurrentSignError",
    "build_contour_tree",
    "dd_emittable_nodes",
    "sigma_for_state",
]
