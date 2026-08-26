"""Fixed-capacity closed and open separatrix branches from spline contour arcs.

The graph nodes are the canonical shared-edge crossings emitted by
``traced_spline_contour``.  An exact-level saddle segment is split at the
polished saddle before components are classified.  The split deliberately uses
one graph node per incoming segment: the axis-enclosing lobe remains a cycle,
while the two physical divertor legs remain distinct paths even though their
terminal coordinates coincide at the saddle.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from nova.equilibrium.flux_surface_connectivity import traced_spline_contour


__all__ = ["assemble_separatrix_branches"]


def _split_cubic_at_saddle(controls, saddle):
    """Split a cubic near its midpoint and move the shared point to a saddle."""
    first = 0.5 * (controls[..., 0, :] + controls[..., 1, :])
    middle = 0.5 * (controls[..., 1, :] + controls[..., 2, :])
    last = 0.5 * (controls[..., 2, :] + controls[..., 3, :])
    left_inner = 0.5 * (first + middle)
    right_inner = 0.5 * (middle + last)
    midpoint = 0.5 * (left_inner + right_inner)
    correction = saddle - midpoint
    left = jnp.stack(
        (controls[..., 0, :], first, left_inner + correction, saddle), axis=-2
    )
    right = jnp.stack(
        (saddle, right_inner + correction, last, controls[..., 3, :]), axis=-2
    )
    return left, right


def _expanded_graph(contour):
    controls = contour["segment_controls_rz"].reshape(-1, 4, 2)
    nodes = contour["segment_node_indices"].reshape(-1, 2)
    valid = contour["segment_valid"].reshape(-1)
    at_saddle = contour["segment_at_saddle"].reshape(-1) & valid
    saddle = contour["segment_saddle_rz"].reshape(-1, 2)
    segment_count = controls.shape[0]
    edge_node_capacity = contour["edge_node_capacity"]
    cell_count = segment_count // 2
    unique_floor = edge_node_capacity + cell_count
    first_saddle_node = unique_floor + 2 * jnp.arange(segment_count, dtype=jnp.int32)
    second_saddle_node = first_saddle_node + 1
    saddle_group_node = (
        edge_node_capacity + jnp.arange(segment_count, dtype=jnp.int32) // 2
    )

    left, right = _split_cubic_at_saddle(controls, saddle)
    first_controls = jnp.where(at_saddle[:, None, None], left, controls)
    second_controls = right
    first_nodes = jnp.stack(
        (nodes[:, 0], jnp.where(at_saddle, first_saddle_node, nodes[:, 1])), axis=-1
    )
    second_nodes = jnp.stack((second_saddle_node, nodes[:, 1]), axis=-1)
    first_group_nodes = jnp.stack(
        (nodes[:, 0], jnp.where(at_saddle, saddle_group_node, nodes[:, 1])), axis=-1
    )
    second_group_nodes = jnp.stack((saddle_group_node, nodes[:, 1]), axis=-1)

    expanded_controls = jnp.stack((first_controls, second_controls), axis=1).reshape(
        -1, 4, 2
    )
    expanded_nodes = jnp.stack((first_nodes, second_nodes), axis=1).reshape(-1, 2)
    grouped_nodes = jnp.stack((first_group_nodes, second_group_nodes), axis=1).reshape(
        -1, 2
    )
    expanded_valid = jnp.stack((valid, at_saddle), axis=1).reshape(-1)
    expanded_saddle = jnp.stack((at_saddle, at_saddle), axis=1).reshape(-1)
    return (
        expanded_controls,
        expanded_nodes,
        grouped_nodes,
        expanded_valid,
        expanded_saddle,
        unique_floor,
    )


def _join_axis_lobe_paths(nodes, grouped_nodes, valid, labels, degree, unique_floor):
    """Join only saddle-to-saddle paths at their coordinate-identical ends."""
    edge_count = nodes.shape[0]
    endpoint = valid[:, None] & (degree[nodes] == 1)
    endpoint_count = jnp.zeros((edge_count,), dtype=jnp.int32)
    boundary_count = jnp.zeros((edge_count,), dtype=jnp.int32)
    safe_labels = jnp.where(valid, labels, 0)
    endpoint_count = endpoint_count.at[safe_labels].add(
        jnp.sum(endpoint, axis=1, dtype=jnp.int32)
    )
    boundary_count = boundary_count.at[safe_labels].add(
        jnp.sum(endpoint & (nodes < unique_floor), axis=1, dtype=jnp.int32)
    )
    joinable = (endpoint_count == 2) & (boundary_count == 0)
    use_group = valid & joinable[safe_labels]
    return jnp.where(use_group[:, None] & (nodes >= unique_floor), grouped_nodes, nodes)


def _component_labels(nodes, valid, node_capacity):
    edge_count = nodes.shape[0]
    sentinel = jnp.asarray(edge_count, jnp.int32)
    labels = jnp.where(valid, jnp.arange(edge_count, dtype=jnp.int32), sentinel)

    def relax(_iteration, current):
        node_label = jnp.full((node_capacity,), sentinel, dtype=jnp.int32)
        node_label = node_label.at[nodes[:, 0]].min(current)
        node_label = node_label.at[nodes[:, 1]].min(current)
        adjacent = jnp.minimum(node_label[nodes[:, 0]], node_label[nodes[:, 1]])
        return jnp.where(valid, jnp.minimum(current, adjacent), sentinel)

    return jax.lax.fori_loop(0, edge_count, relax, labels)


def _component_properties(controls, nodes, valid, labels, axis_rz, node_capacity):
    edge_count = nodes.shape[0]
    degree = jnp.zeros((node_capacity,), dtype=jnp.int32)
    degree = degree.at[nodes[:, 0]].add(valid.astype(jnp.int32))
    degree = degree.at[nodes[:, 1]].add(valid.astype(jnp.int32))
    edge_degree_ok = (degree[nodes[:, 0]] == 2) & (degree[nodes[:, 1]] == 2)
    safe_labels = jnp.where(valid, labels, 0)
    bad_degree = jnp.zeros((edge_count,), dtype=bool)
    bad_degree = bad_degree.at[safe_labels].max(valid & ~edge_degree_ok)
    component_valid = jnp.zeros((edge_count,), dtype=bool).at[safe_labels].max(valid)

    start = controls[:, 0, :]
    end = controls[:, 3, :]
    axis_r, axis_z = axis_rz
    crosses = ((start[:, 1] > axis_z) != (end[:, 1] > axis_z)) & (
        axis_r
        < start[:, 0]
        + (axis_z - start[:, 1])
        * (end[:, 0] - start[:, 0])
        / jnp.where(end[:, 1] != start[:, 1], end[:, 1] - start[:, 1], 1.0)
    )
    crossing_count = jnp.zeros((edge_count,), dtype=jnp.int32)
    crossing_count = crossing_count.at[safe_labels].add(
        (valid & crosses).astype(jnp.int32)
    )
    encloses_axis = (crossing_count & 1) == 1
    return degree, component_valid, ~bad_degree & component_valid, encloses_axis


def _trace_component(
    controls, nodes, valid, labels, component, degree, saddle_node_floor, capacity
):
    edge_count = nodes.shape[0]
    sentinel_edge = jnp.asarray(edge_count, jnp.int32)
    sentinel_node = jnp.asarray(degree.size, jnp.int32)
    member = valid & (labels == component)
    endpoint = member[:, None] & (degree[nodes] == 1)
    open_start = jnp.min(jnp.where(endpoint, nodes, sentinel_node))
    any_open = jnp.any(endpoint)
    component_node = jnp.min(jnp.where(member[:, None], nodes, sentinel_node))
    saddle_node = jnp.min(
        jnp.where(member[:, None] & (nodes >= saddle_node_floor), nodes, sentinel_node)
    )
    has_saddle = saddle_node < sentinel_node
    cycle_start = jnp.where(has_saddle, saddle_node, component_node)
    start_node = jnp.where(any_open, open_start, cycle_start)
    output = jnp.zeros((capacity, 4, 2), dtype=controls.dtype)
    output_valid = jnp.zeros((capacity,), dtype=bool)

    def visit(slot, state):
        current_node, previous_edge, done, branch, branch_valid = state
        incident = member & ~done & (jnp.any(nodes == current_node, axis=1))
        incident &= jnp.arange(edge_count, dtype=jnp.int32) != previous_edge
        selected = jnp.min(
            jnp.where(incident, jnp.arange(edge_count, dtype=jnp.int32), sentinel_edge)
        )
        found = selected < edge_count
        safe_selected = jnp.minimum(selected, edge_count - 1)
        forward = nodes[safe_selected, 0] == current_node
        oriented = jnp.where(
            forward,
            controls[safe_selected],
            controls[safe_selected, ::-1],
        )
        next_node = jnp.where(forward, nodes[safe_selected, 1], nodes[safe_selected, 0])
        active = found & ~done
        branch = branch.at[slot].set(jnp.where(active, oriented, 0.0))
        branch_valid = branch_valid.at[slot].set(active)
        finished = done | ~found | (active & (next_node == start_node))
        return next_node, selected, finished, branch, branch_valid

    _node, _edge, _done, output, output_valid = jax.lax.fori_loop(
        0,
        capacity,
        visit,
        (start_node, sentinel_edge, jnp.asarray(False), output, output_valid),
    )
    count = jnp.sum(member, dtype=jnp.int32)
    return output, output_valid, count


@partial(
    jax.jit,
    static_argnames=(
        "branch_capacity",
        "open_branch_capacity",
        "bisection_steps",
        "saddle_steps",
    ),
)
def assemble_separatrix_branches(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    level: jnp.ndarray,
    axis_rz: jnp.ndarray,
    *,
    branch_capacity: int = 256,
    open_branch_capacity: int = 4,
    bisection_steps: int = 40,
    saddle_steps: int = 8,
) -> dict[str, jnp.ndarray]:
    """Assemble one closed axis branch and fixed slots of saddle-ended legs.

    Every branch stores ordered cubic controls.  ``*_valid`` masks define the
    active prefix; all remaining controls are exact zero.  Any missing or
    non-unique axis-enclosing cycle, graph junction, branch overflow, or open
    slot overflow invalidates the whole result and returns only zero geometry.
    """
    contour = traced_spline_contour(
        values, radial, vertical, level, bisection_steps, saddle_steps
    )
    (
        controls,
        nodes,
        grouped_nodes,
        valid,
        edge_from_saddle,
        unique_floor,
    ) = _expanded_graph(contour)
    edge_count = controls.shape[0]
    saddle_node_floor = (
        vertical.shape[0] * (radial.shape[0] - 1)
        + (vertical.shape[0] - 1) * radial.shape[0]
    )
    cell_count = edge_count // 4
    node_capacity = saddle_node_floor + cell_count + edge_count
    labels = _component_labels(nodes, valid, node_capacity)
    degree, _component_valid, _cycle, _encloses_axis = _component_properties(
        controls, nodes, valid, labels, axis_rz, node_capacity
    )
    nodes = _join_axis_lobe_paths(
        nodes, grouped_nodes, valid, labels, degree, unique_floor
    )
    labels = _component_labels(nodes, valid, node_capacity)
    degree, component_valid, cycle, encloses_axis = _component_properties(
        controls, nodes, valid, labels, axis_rz, node_capacity
    )
    closed_candidates = component_valid & cycle & encloses_axis
    closed_candidate_count = jnp.sum(closed_candidates, dtype=jnp.int32)
    closed_component = jnp.argmax(closed_candidates).astype(jnp.int32)
    closed_controls, closed_valid, closed_count = _trace_component(
        controls,
        nodes,
        valid,
        labels,
        closed_component,
        degree,
        saddle_node_floor,
        branch_capacity,
    )

    safe_labels = jnp.where(valid, labels, 0)
    component_has_saddle = jnp.zeros((edge_count,), dtype=bool)
    component_has_saddle = component_has_saddle.at[safe_labels].max(
        valid & edge_from_saddle
    )
    open_component = component_valid & ~cycle & component_has_saddle
    open_count = jnp.sum(open_component, dtype=jnp.int32)
    component_order = jnp.nonzero(
        open_component, size=open_branch_capacity, fill_value=0
    )[0]

    def trace_open(component):
        return _trace_component(
            controls,
            nodes,
            valid,
            labels,
            component,
            degree,
            saddle_node_floor,
            branch_capacity,
        )

    open_controls, open_valid, open_segment_count = jax.vmap(trace_open)(
        component_order
    )
    open_slot_valid = jnp.arange(open_branch_capacity) < open_count
    open_valid &= open_slot_valid[:, None]
    open_controls = jnp.where(open_valid[..., None, None], open_controls, 0.0)

    overflow = (closed_count > branch_capacity) | (open_count > open_branch_capacity)
    overflow |= jnp.any(open_slot_valid & (open_segment_count > branch_capacity))
    graph_well_formed = jnp.all(
        ~valid | ((degree[nodes[:, 0]] <= 2) & (degree[nodes[:, 1]] <= 2))
    )
    well_formed = (
        contour["well_formed"]
        & graph_well_formed
        & (closed_candidate_count == 1)
        & ~overflow
    )
    closed_valid &= well_formed
    open_slot_valid &= well_formed
    open_valid &= well_formed
    return {
        "closed_controls_rz": jnp.where(
            closed_valid[:, None, None], closed_controls, 0.0
        ),
        "closed_valid": closed_valid,
        "closed_segment_count": jnp.where(well_formed, closed_count, 0),
        "open_controls_rz": jnp.where(open_valid[..., None, None], open_controls, 0.0),
        "open_valid": open_valid,
        "open_branch_valid": open_slot_valid,
        "open_segment_count": jnp.where(open_slot_valid, open_segment_count, 0),
        "open_branch_count": jnp.where(well_formed, open_count, 0),
        "closed_candidate_count": closed_candidate_count,
        "overflow": overflow,
        "well_formed": well_formed,
    }
