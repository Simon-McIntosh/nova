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

from nova.equilibrium.flux_surface_connectivity import (
    EDGE_CROSSING_CAPACITY,
    traced_spline_contour,
)
from nova.linalg.tensor_spline import fit_tensor_spline


__all__ = [
    "assemble_separatrix_branches",
    "boundary_flux_at_admitted_saddle",
]


def boundary_flux_at_admitted_saddle(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    axis_rz: jnp.ndarray,
    xpoint_rz: jnp.ndarray,
    *,
    bisection_steps: int = 40,
    saddle_steps: int = 8,
) -> jnp.ndarray:
    """Read the traced field's own boundary flux at its admitted saddle.

    The boundary level of a solved field is its flux at the X-point, and the
    level that pairs the crossings of a saddle's own cell is that cell's
    stationary value.  A level read from another grid -- a refit, or the flux
    at a coordinate carried over from a different lattice -- misses the cell's
    stationary value by far more than the decision tolerance, so the
    four-crossing cell falls back on the corner-sign pairing and runs the
    axis-enclosing lobe out along a leg.  Reading the level from the same field
    the crossings are traced on keeps the pairing decision and the saddle
    inside one fit.

    ``xpoint_rz`` only has to land in the saddle's cell: it selects which
    stationary cell is meant, and the returned value is that cell's own
    polished stationary value.  Where the field carries no stationary cell --
    an undiverted state -- the flux at ``xpoint_rz`` is returned unchanged.
    """
    surface = fit_tensor_spline(radial, vertical, values)
    locator = surface(xpoint_rz[0], xpoint_rz[1])
    contour = traced_spline_contour(
        values,
        radial,
        vertical,
        locator,
        bisection_steps,
        saddle_steps,
        surface=surface,
        axis_rz=axis_rz,
    )
    stationary = contour["saddle_stationary"].reshape(-1)
    saddle_value = contour["saddle_value"].reshape(-1)
    saddle_rz = contour["saddle_rz"].reshape(-1, 2)
    distance = jnp.linalg.norm(
        saddle_rz - jnp.asarray(xpoint_rz, dtype=saddle_rz.dtype)[None, :], axis=-1
    )
    distance = jnp.where(stationary, distance, jnp.inf)
    nearest = jnp.argmin(distance)
    return jnp.where(jnp.isfinite(distance[nearest]), saddle_value[nearest], locator)


def _unit(vector):
    length = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / jnp.where(length > 0.0, length, 1.0)


def _arm_to_saddle(crossing, crossing_tangent, saddle, asymptotes):
    """Return the cubic from an edge crossing to the saddle along its arm.

    The crossing keeps the spline tangent the cell's cubic already carries
    there, and the saddle end takes the asymptote ray that faces the crossing:
    the separatrix arm meets the X-point along a null line of the Hessian, not
    along the vertex tangent of the cubic that bows past it.  Both handles use
    the same chord-projected length as every other cell's cubic.
    """
    chord = saddle - crossing
    chord_length = jnp.linalg.norm(chord, axis=-1)
    tangent = _unit(crossing_tangent)
    tangent = jnp.where(
        (jnp.sum(tangent * chord, axis=-1) < 0.0)[..., None], -tangent, tangent
    )
    rays = jnp.concatenate((asymptotes, -asymptotes), axis=-2)
    facing = jnp.argmax(jnp.sum(rays * -chord[..., None, :], axis=-1), axis=-1)
    ray = jnp.take_along_axis(rays, facing[..., None, None], axis=-2)[..., 0, :]
    minimum_scale = 0.25 * chord_length
    start_scale = jnp.maximum(jnp.sum(chord * tangent, axis=-1), minimum_scale)
    end_scale = jnp.maximum(jnp.sum(-chord * ray, axis=-1), minimum_scale)
    return jnp.stack(
        (
            crossing,
            crossing + start_scale[..., None] * tangent / 3.0,
            saddle + end_scale[..., None] * ray / 3.0,
            saddle,
        ),
        axis=-2,
    )


def _split_cubic_at_saddle(controls, saddle, asymptotes):
    """Replace a saddle cell's cubic by its two arms meeting at the saddle."""
    left = _arm_to_saddle(
        controls[..., 0, :],
        controls[..., 1, :] - controls[..., 0, :],
        saddle,
        asymptotes,
    )
    right = _arm_to_saddle(
        controls[..., 3, :],
        controls[..., 2, :] - controls[..., 3, :],
        saddle,
        asymptotes,
    )[..., ::-1, :]
    return left, right


def _expanded_graph(contour):
    controls = contour["segment_controls_rz"].reshape(-1, 4, 2)
    nodes = contour["segment_node_indices"].reshape(-1, 2)
    valid = contour["segment_valid"].reshape(-1)
    at_saddle = contour["segment_at_saddle"].reshape(-1) & valid
    saddle = contour["segment_saddle_rz"].reshape(-1, 2)
    asymptotes = contour["segment_saddle_asymptotes_rz"].reshape(-1, 2, 2)
    segment_count = controls.shape[0]
    edge_node_capacity = contour["edge_node_capacity"]
    cell_count = segment_count // 2
    unique_floor = edge_node_capacity + cell_count
    first_saddle_node = unique_floor + 2 * jnp.arange(segment_count, dtype=jnp.int32)
    second_saddle_node = first_saddle_node + 1
    saddle_group_node = (
        edge_node_capacity + jnp.arange(segment_count, dtype=jnp.int32) // 2
    )

    left, right = _split_cubic_at_saddle(controls, saddle, asymptotes)
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
        values,
        radial,
        vertical,
        level,
        bisection_steps,
        saddle_steps,
        axis_rz=axis_rz,
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
    saddle_node_floor = EDGE_CROSSING_CAPACITY * (
        vertical.shape[0] * (radial.shape[0] - 1)
        + (vertical.shape[0] - 1) * radial.shape[0]
    )
    # The synthetic saddle nodes are numbered from the first id past the real
    # crossings and grouped nodes, two per unsplit segment, so the node arrays
    # must reach that far.  A capacity below it clamps every scatter past its
    # end, and the degrees read back are then not the graph's degrees.
    node_capacity = saddle_node_floor + edge_count // 4 + edge_count
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
    # The candidate count is a conjunction: a component must be closed AND
    # enclose the axis.  Counting the halves separately says which of the two
    # failed -- an open lobe that lost its cycle, or a closed lobe the parity
    # read places outside the axis -- and the two need different repairs.
    representative = valid & (safe_labels == jnp.arange(edge_count, dtype=jnp.int32))
    cycle_component_count = jnp.sum(representative & cycle, dtype=jnp.int32)
    axis_enclosing_component_count = jnp.sum(
        representative & encloses_axis, dtype=jnp.int32
    )
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

    branch_overflow = (closed_count > branch_capacity) | jnp.any(
        open_slot_valid & (open_segment_count > branch_capacity)
    )
    open_slot_overflow = open_count > open_branch_capacity
    overflow = branch_overflow | open_slot_overflow
    graph_well_formed = jnp.all(
        ~valid | ((degree[nodes[:, 0]] <= 2) & (degree[nodes[:, 1]] <= 2))
    )
    contour_well_formed = contour["well_formed"]
    well_formed = (
        contour_well_formed
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
        # The invalidation is reported term by term because the geometry alone
        # cannot distinguish a missing cycle from a non-unique one, a graph
        # junction from a capacity overflow, or a branch overflow from an open
        # slot overflow: all five return the same zeros, and each needs a
        # different repair.
        "contour_well_formed": contour_well_formed,
        "graph_well_formed": graph_well_formed,
        "branch_overflow": branch_overflow,
        "open_slot_overflow": open_slot_overflow,
        "cycle_component_count": cycle_component_count,
        "axis_enclosing_component_count": axis_enclosing_component_count,
    }
