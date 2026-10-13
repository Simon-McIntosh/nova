"""Bell-sampled connected fragments with cubic-flux separatrix geometry."""

import jax
import jax.numpy as jnp

from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.read_route.bell import field_from_jets


def neighbour_mask(geometry, point):
    """Include the null's owner and each carrier sharing one of its edges."""
    owner = topology._inside_cells(geometry, point)
    adjacent = owner[jnp.maximum(geometry.neighbour, 0)]
    return owner | jnp.any(adjacent & (geometry.neighbour >= 0), axis=1)


def connected_support(geometry, coefficient, sigma, level, axis, form, represented):
    """Reuse the read's conic fragments, curved sectors and connectivity owner."""
    policy = TopologyPolicy()
    vertices = (geometry.vertices - geometry.centre[:, None]) / geometry.pitch[
        :, None, None
    ]
    signed = (sigma * coefficient).at[:, 0].add(-sigma * level)
    fragments = jax.vmap(topology.quadratic_cell_fragments, in_axes=(0, 0, 0, None))(
        vertices, geometry.vertex_count, signed, policy.fragment_capacity
    )
    saddle = jax.vmap(topology._normal_form_cell_fragments, in_axes=(0, 0, None, None))(
        geometry.vertices, geometry.vertex_count, form, policy.edge_tolerance
    )
    sectors = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    areas = jnp.where(
        represented[:, None],
        saddle.area[:, sectors],
        fragments.area * geometry.pitch[:, None] ** 2,
    )
    intervals = jnp.where(
        represented[:, None, None, None, None],
        saddle.edge_interval[:, sectors],
        jnp.pad(fragments.edge_interval, ((0, 0), (0, 0), (0, 0), (0, 10), (0, 0))),
    )
    labels, _ = topology._connected_fragment_labels(
        areas, intervals, geometry, policy.edge_tolerance
    )
    axis_cells = topology._inside_cells(geometry, axis)
    cell = jnp.argmax(axis_cells)
    local_axis = (axis - geometry.centre[cell]) / geometry.pitch[cell]
    strip = jnp.clip(
        jnp.searchsorted(fragments.slice_breaks[cell], local_axis[0], side="right") - 1,
        0,
        fragments.slice_labels.shape[1] - 1,
    )
    lower, upper, positive = topology._positive_vertical_intervals(
        vertices[cell], geometry.vertex_count[cell], signed[cell], local_axis[0]
    )
    interval = jnp.argmax(
        positive & (local_axis[1] >= lower) & (local_axis[1] <= upper)
    )
    fragment = fragments.slice_labels[cell, strip, interval]
    selected = (labels == labels[cell, jnp.maximum(fragment, 0)]) & (areas > 0)
    valid = jnp.all(jnp.where(represented, saddle.valid, fragments.valid))
    valid &= jnp.any(axis_cells) & (fragment >= 0)
    membership = jnp.sum(jnp.where(selected, areas, 0), axis=1) / geometry.full_area
    return membership, valid, selected


def membership(
    jets, indices, mesh, geometry, fixed, level, sigma, diverted, third_factor=1.0
):
    """One compiled membership step, including Bell reconstruction and curvature.

    Smooth fragments use the read's own-node quadratic sampled from the Bell
    field. At the null and its edge neighbours, curve geometry replaces that
    fit. Only the Hessian and third derivatives enter the local flux model.
    """
    field = field_from_jets(jets, indices, mesh)
    values = jets[indices, 0]
    values = values.at[:, 0].set(
        jax.vmap(field.cell_value)(jnp.arange(len(values)), geometry.centre)
    )
    coefficient = jnp.einsum("nij,nj->ni", geometry.fit_inverse, values)
    point = fixed.position[-1]
    third = jax.jacfwd(lambda target: field.evaluate(target).hessian)(point)
    form = topology.saddle_normal_form(
        point, sigma * fixed.jet.hessian[-1], sigma * third * third_factor, 1e-12
    )
    represented = neighbour_mask(geometry, point) & diverted
    fractions, valid, selected = connected_support(
        geometry, coefficient, sigma, level, fixed.position[0], form, represented
    )
    return fractions, valid, form, represented, selected


def crossed_within_radius(vertices, vertex_count, form, radius):
    """Select a convex cell hit by a separatrix leg inside a physical disc."""
    slot = jnp.arange(vertices.shape[0])
    following = jnp.where(slot + 1 < vertex_count, slot + 1, 0)
    relative = vertices - form.position
    edge = relative[following] - relative
    cross = topology._cross_plane
    roots = topology._cubic_parameters(
        cross(form.cubic[:, None], edge[None]),
        cross(form.curvature[:, None], edge[None]),
        cross(form.direction[:, None], edge[None]),
        -cross(relative[None], edge[None]),
    )
    parameter = jnp.where(jnp.isfinite(roots), roots, 0)
    points = (
        parameter[..., None] * form.direction[:, None, None]
        + parameter[..., None] ** 2 * form.curvature[:, None, None]
        + parameter[..., None] ** 3 * form.cubic[:, None, None]
    )
    square = jnp.sum(edge * edge, axis=-1)
    fraction = (
        jnp.sum((points - relative[None, :, None]) * edge[None, :, None], axis=-1)
        / jnp.where(square > 0, square, 1)[None, :, None]
    )
    hit = (
        (slot < vertex_count)[None, :, None]
        & jnp.isfinite(roots)
        & (roots >= 0)
        & (roots <= 2 * radius)
        & (fraction >= -1e-12)
        & (fraction <= 1 + 1e-12)
        & (jnp.sum(points * points, axis=-1) <= radius**2)
    )
    contains = jnp.all(
        jnp.where(slot < vertex_count, cross(edge, -relative) >= -1e-12, True)
    )
    return form.valid & (contains | jnp.any(hit))


def patch_membership(
    jets,
    indices,
    mesh,
    geometry,
    fixed,
    bell,
    sigma,
    diverted,
    physical_radius,
    use_crossed_cells,
    third_factor=1.0,
):
    """Replace local fragments while preserving the Bell receipt elsewhere.

    Bell fragments are explicit operands, including their selected components.
    Connectivity is recomputed over the merged fragment graph; a changed
    positive-area selection outside the patch is refused. Membership outside
    the patch is copied from the Bell receipt without another reduction.
    """
    field = field_from_jets(jets, indices, mesh)
    point = fixed.position[-1]
    third = jax.jacfwd(lambda target: field.evaluate(target).hessian)(point)
    form = topology.saddle_normal_form(
        point, sigma * fixed.jet.hessian[-1], sigma * third * third_factor, 1e-12
    )
    crossed = jax.vmap(crossed_within_radius, in_axes=(0, 0, None, None))(
        geometry.vertices, geometry.vertex_count, form, physical_radius
    )
    represented = jnp.where(use_crossed_cells, crossed, neighbour_mask(geometry, point))
    represented &= diverted
    saddle = jax.vmap(topology._normal_form_cell_fragments, in_axes=(0, 0, None, None))(
        geometry.vertices, geometry.vertex_count, form, 1e-12
    )
    sectors = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    area = jnp.where(represented[:, None], saddle.area[:, sectors], bell.fragment_area)
    intervals = jnp.where(
        represented[:, None, None, None, None],
        saddle.edge_interval[:, sectors],
        bell.edge_interval,
    )
    labels, _ = topology._connected_fragment_labels(
        area, intervals, geometry, TopologyPolicy().edge_tolerance
    )
    axis_cell = jnp.argmax(topology._inside_cells(geometry, fixed.position[0]))
    axis_fragment = jnp.argmax(bell.fragment_selected[axis_cell])
    selected = (labels == labels[axis_cell, axis_fragment]) & (area > 0)
    changed = ~represented[:, None] & (area > 0) & (selected != bell.fragment_selected)
    fractions = jnp.sum(jnp.where(selected, area, 0), axis=1) / geometry.full_area
    fractions = jnp.where(represented, fractions, bell.membership)
    selected = jnp.where(represented[:, None], selected, bell.fragment_selected)
    valid = bell.valid & jnp.all(jnp.where(represented, saddle.valid, True))
    valid &= ~jnp.any(changed)
    return fractions, valid, form, represented, selected, area, intervals, changed
