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
    ``critical_type`` uses the DD convention (minimum 0, saddle 1, maximum 2).
    ``node_valid`` and ``edge_valid`` make every array fixed shaped.  Edges
    refer to vertex slots, which are also the node slots.  ``overflow`` means
    that a candidate could not fit in the declared DD capacities; callers must
    refuse that receipt instead of treating the prefix as a tree.
    """

    node_vertex: jax.Array
    node_psi: jax.Array
    node_valid: jax.Array
    critical_type: jax.Array
    edges: jax.Array
    edge_valid: jax.Array
    overflow: jax.Array


def _roots(parents: jax.Array) -> jax.Array:
    """Pointer-jump a fixed-size union-find forest to canonical roots."""

    return jax.lax.fori_loop(0, parents.size, lambda _i, tree: tree[tree], parents)


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


def _sweep_tree(
    values: jax.Array,
    vertex_valid: jax.Array,
    vertex_is_wall: jax.Array,
    edges: jax.Array,
    edge_valid: jax.Array,
    descending: bool,
    node_capacity: int,
    edge_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sweep one merge tree and emit extrema, saddles, and bounded arcs.

    A lexicographic ordering of value then carrier index is simulation of
    simplicity: values can be exactly equal without creating an ambiguous
    union.  The first wall vertex encountered while descending is a join with
    the virtual outside component.  It is represented by that real wall slot;
    the virtual node is deliberately not emitted as a DD critical point.
    """

    vertex_count = values.size
    edge_count = edges.shape[0]
    index = jnp.arange(vertex_count, dtype=jnp.int32)
    order = jnp.lexsort((index, -values if descending else values))
    parents = index
    active = jnp.zeros(vertex_count, dtype=bool)
    births = index
    node_valid = jnp.zeros(node_capacity, dtype=bool)
    node_type = jnp.full(node_capacity, -1, dtype=jnp.int32)
    output_edges = jnp.zeros((edge_capacity, 2), dtype=jnp.int32)
    output_valid = jnp.zeros(edge_capacity, dtype=bool)
    overflow = jnp.asarray(vertex_count > node_capacity, dtype=bool)
    wall_seen = jnp.asarray(False)

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
            wall_seen,
        ) = state
        vertex = order[rank].astype(jnp.int32)
        usable = vertex_valid[vertex]
        roots = _roots(parents)
        first = edges[:, 0]
        second = edges[:, 1]
        is_first = edge_valid & (first == vertex)
        is_second = edge_valid & (second == vertex)
        neighbour = jnp.where(is_first, second, first)
        adjacent = is_first | is_second
        neighbour_root = roots[neighbour]
        root_active = adjacent & active[neighbour]
        earlier = jnp.arange(edge_count)[:, None] < jnp.arange(edge_count)
        repeated = jnp.any(
            earlier
            & root_active[:, None]
            & root_active[None, :]
            & (neighbour_root[:, None] == neighbour_root[None, :]),
            axis=0,
        )
        representative = root_active & ~repeated
        component_count = jnp.sum(representative, dtype=jnp.int32)
        first_wall = usable & descending & vertex_is_wall[vertex] & ~wall_seen
        final_vertex = usable & (rank == vertex_count - 1)
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
        node_slot = vertex
        fits_node = node_slot < node_capacity
        node_valid = node_valid.at[jnp.minimum(node_slot, node_capacity - 1)].set(
            node_valid[jnp.minimum(node_slot, node_capacity - 1)] | (event & fits_node)
        )
        node_type = node_type.at[jnp.minimum(node_slot, node_capacity - 1)].set(
            jnp.where(
                event & fits_node,
                critical,
                node_type[jnp.minimum(node_slot, node_capacity - 1)],
            )
        )
        overflow = overflow | (event & ~fits_node)
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
        parent_update = jnp.where(root_active & usable, vertex, parents[neighbour_root])
        parents = parents.at[neighbour_root].set(
            jnp.where(usable, parent_update, parents[neighbour_root])
        )
        parents = parents.at[vertex].set(vertex)
        first_root = jnp.argmax(representative).astype(jnp.int32)
        inherited = births[neighbour_root[first_root]]
        births = births.at[vertex].set(
            jnp.where(extremum | saddle | first_wall, vertex, inherited)
        )
        active = active.at[vertex].set(usable)
        return (
            parents,
            active,
            births,
            node_valid,
            node_type,
            output_edges,
            output_valid,
            overflow,
            wall_seen | (usable & descending & vertex_is_wall[vertex]),
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
            wall_seen,
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
    signed = jnp.asarray(sigma, dtype=vertex_psi.dtype) * vertex_psi
    join = _sweep_tree(
        signed,
        vertex_valid,
        vertex_is_wall,
        edges,
        edge_valid,
        True,
        node_capacity,
        edge_capacity,
    )
    split = _sweep_tree(
        signed,
        vertex_valid,
        vertex_is_wall,
        edges,
        edge_valid,
        False,
        node_capacity,
        edge_capacity,
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
    merged_edges, merged_valid, overflow = _append_edges(
        join_edges,
        join_valid,
        join_overflow | split_overflow,
        split_edges[:, 0],
        split_valid & ~join_nodes[split_edges[:, 0]],
        split_edges[:, 1],
    )
    slots = jnp.arange(node_capacity, dtype=jnp.int32)
    return ContourTreeArrays(
        node_vertex=slots,
        node_psi=jnp.where(
            node_valid, vertex_psi[slots], jnp.zeros((), vertex_psi.dtype)
        ),
        node_valid=node_valid,
        critical_type=critical_type,
        edges=merged_edges,
        edge_valid=merged_valid,
        overflow=overflow,
    )


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
    "ContourTreeEdge",
    "ContourTreeArrays",
    "ContourTreeNode",
    "ContourTreeResult",
    "ContourTreeState",
    "FluxCurrentSignError",
    "build_contour_tree",
    "sigma_for_state",
]
