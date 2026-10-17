"""Vectorised fixed-shape topology read for the coarse moment map."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium import topology
from scripts.prototypes.fast_read.ordinary_cut import _split_polygons
from scripts.prototypes.sub_millisecond.measure_bell_cut import (
    bell_polygon as single_polygon,
)
from scripts.prototypes.sub_millisecond.measure_wall_march import polygon_area


class FastRead(NamedTuple):
    axis: jax.Array
    axis_flux: jax.Array
    boundary: jax.Array
    boundary_flux: jax.Array
    boundary_class: jax.Array
    wall_candidate: jax.Array
    saddle_candidate: jax.Array
    membership: jax.Array
    geometric_membership: jax.Array
    path_weight: jax.Array
    selected_cells: jax.Array
    field_coefficients: jax.Array
    saddle_form: topology.SaddleNormalForm
    normal_form_cells: jax.Array
    booking_cells: jax.Array
    fragment_selected: jax.Array
    edge_interval: jax.Array
    qualified: jax.Array
    valid: jax.Array


def bell_fraction_with_crossings(geometry, reading, sigma):
    """Return the ordinary area and the crossings that select saddle booking."""
    polygon, count, valid = single_polygon(geometry, reading, sigma, 4)
    default = polygon_area(polygon, count)
    split, split_count, split_valid, four = _split_polygons(geometry, reading, sigma)
    split_area = jnp.sum(
        jax.vmap(polygon_area, in_axes=(1, 1), out_axes=1)(split, split_count), axis=1
    )
    area = jnp.where(four, split_area, default)
    fraction = jnp.clip(area / geometry.full_area, 0.0, 1.0)
    return fraction, jnp.where(four, split_valid, valid), four


def _local_axis_sector(geometry, axis, form):
    """Select the axis-facing branch inside the saddle form's local domain."""
    direction = axis - form.position
    distance = jnp.linalg.norm(direction)
    reach = 0.5 * jnp.median(geometry.pitch)
    nearby = form.position + direction * jnp.minimum(
        1.0, reach / jnp.maximum(distance, 1e-30)
    )
    return jnp.argmax(topology._normal_form_sectors(nearby[None], form)[0])


def _sector_sidedness(midpoint, form, axis_sector):
    """Continuous signed distance from an edge midpoint to the axis-facing sector.

    The positive branches of the saddle normal form divide the local domain by
    curved rays. A boolean side test snaps that partition onto the mesh; this
    returns the margin to the nearest curved ray so the frame treatment can vary
    smoothly across the boundary instead of switching between cells.
    """
    offset = midpoint - form.position
    parameter = jnp.einsum("...i,ai->...a", offset, form.direction)
    residual = (
        offset[..., None, :]
        - parameter[..., None] ** 2 * form.curvature
        - parameter[..., None] ** 3 * form.cubic
    )
    side = (
        form.direction[:, 0] * residual[..., 1]
        - form.direction[:, 1] * residual[..., 0]
    )
    next_side = jnp.roll(side, -1, axis=-1)
    return jnp.minimum(side[..., axis_sector], -next_side[..., axis_sector])


def _wall_path_ceiling(
    geometry,
    coefficients,
    form,
    axis,
    radius,
    saddle_live,
    saddle_score,
    seed,
    sigma,
    frame_gate,
):
    """Highest signed level at which each cell is reachable from the axis.

    The maximin bottleneck of the edge passages, capped near the saddle by a
    continuous frame attenuation so a path that crosses the saddle along a
    non-positive branch is limited toward the saddle level without any boolean
    sector test.
    """
    vertices = geometry.vertices
    count = geometry.vertex_count
    slot = jnp.arange(vertices.shape[1])
    following = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=1)
    midpoint = (vertices + end) / 2

    def signed(points):
        local = (points - geometry.centre[:, None]) / geometry.pitch[:, None, None]
        x, y = local[..., 0], local[..., 1]
        return sigma * (
            coefficients[:, 0, None]
            + coefficients[:, 1, None] * x
            + coefficients[:, 2, None] * y
            + coefficients[:, 3, None] * x * x
            + coefficients[:, 4, None] * x * y
            + coefficients[:, 5, None] * y * y
        )

    first, middle, last = signed(vertices), signed(midpoint), signed(end)
    curvature = 2 * (first + last - 2 * middle)
    slope = last - first - curvature
    fraction = jnp.clip(-slope / jnp.where(curvature < 0, 2 * curvature, 1), 0, 1)
    peak = jnp.maximum(
        jnp.maximum(first, last),
        signed(vertices + fraction[..., None] * (end - vertices)),
    )
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_edge = jnp.maximum(geometry.neighbour_edge, 0)
    ceiling = jnp.minimum(peak, peak[peer, peer_edge])
    if frame_gate:
        axis_sector = _local_axis_sector(geometry, axis, form)
        length_width = jnp.maximum(1e-7, 1e-4 * jnp.median(geometry.pitch))
        sector_weight = jnp.clip(
            0.5 + _sector_sidedness(midpoint, form, axis_sector) / length_width, 0, 1
        )
        distance = jnp.linalg.norm(geometry.centre - form.position, axis=1)
        near_weight = jnp.clip(0.5 + (radius - distance) / length_width, 0, 1)
        near_weight = jnp.maximum(
            near_weight,
            topology._inside_cells(geometry, form.position).astype(near_weight.dtype),
        )
        near_weight = jnp.where(saddle_live, near_weight, 0)
        leak = near_weight[:, None] * (1 - sector_weight)
        leak = jnp.maximum(leak, leak[peer, peer_edge])
        over = jnp.where(saddle_live, jnp.maximum(ceiling - saddle_score, 0.0), 0.0)
        # Written as a min against the interpolated level, not as a subtraction:
        # subtracting the excess computes ceiling - (ceiling - saddle_score),
        # which in binary64 returns the saddle level to within a few ulp rather
        # than exactly, and a wall contact that ties the saddle level to the last
        # bit then wins the boundary on the tie. Where the attenuation saturates
        # this is exactly min(ceiling, saddle_score).
        ceiling = jnp.minimum(ceiling, saddle_score + (1.0 - leak) * over)
    # The cell quadratic is a fitted surrogate for the field, so an edge peak
    # inside the cell that owns the X-point can land on the wrong side of the
    # true saddle level. The read already locates that level, so the passage
    # through the owner cell is bounded by it before any path can carry it into
    # the private-flux region. A plain min of two located levels, so nothing
    # boolean returns and the bound is continuous in every fitted quantity.
    if form is not None:
        owner = topology._inside_cells(geometry, form.position)
        seat = jnp.where(saddle_live, owner, False)
        ceiling = jnp.where(seat[:, None], jnp.minimum(ceiling, saddle_score), ceiling)
    ceiling = jnp.where(geometry.neighbour >= 0, ceiling, -jnp.inf)
    best = jnp.full((len(geometry.centre),), -jnp.inf).at[seed].set(jnp.inf)

    def relax(_, current):
        passage = jnp.minimum(current[peer], ceiling)
        return jnp.maximum(current, jnp.max(passage, axis=1))

    return jax.lax.fori_loop(0, len(geometry.centre), relax, best)


def _wall_effective_level(tangency, bottleneck):
    """Effective level of a wall contact: the more scrape-off-side of its own
    tangency level and the best axis path's bottleneck. Continuous in both
    operands and free of any boolean reachability gate."""
    return jnp.minimum(tangency, bottleneck)


def read(field, roots, geometry, convention, policy, *, frame_gate=True):
    """Read both boundary candidates and axis-connected Bell support at fixed shape."""
    sigma = convention.sigma
    axis = roots.position[0]
    axis_flux = roots.jet.value[0]
    wall_start = geometry.wall_points
    wall_end = wall_start[geometry.wall_following]
    wall_midpoint = (wall_start + wall_end) / 2
    node_flux = jax.vmap(lambda point: field.evaluate(point).value)(wall_start)
    middle_flux = jax.vmap(lambda point: field.evaluate(point).value)(wall_midpoint)
    signed_start = sigma * node_flux
    signed_end = signed_start[geometry.wall_following]
    signed_middle = sigma * middle_flux
    curvature = 2 * (signed_start + signed_end - 2 * signed_middle)
    slope = signed_end - signed_start - curvature
    peak_fraction = jnp.clip(
        -slope / jnp.where(curvature < 0, 2 * curvature, 1), 0.0, 1.0
    )
    peak = wall_start + peak_fraction[:, None] * (wall_end - wall_start)
    peak_flux = jax.vmap(lambda point: field.evaluate(point).value)(peak)
    candidate_points = jnp.stack((wall_start, wall_end, peak), axis=1)
    candidate_values = jnp.stack(
        (node_flux, node_flux[geometry.wall_following], peak_flux), axis=1
    )
    best_on_edge = jnp.argmax(sigma * candidate_values, axis=1)
    edge_row = jnp.arange(len(wall_start))
    wall_points = candidate_points[edge_row, best_on_edge]
    wall_flux = candidate_values[edge_row, best_on_edge]
    count = roots.position.shape[0]
    if count == 1:
        roots = jax.tree.map(
            lambda leaf: jnp.concatenate((leaf, jnp.zeros_like(leaf)), axis=0), roots
        )
    eigenvalues = jnp.linalg.eigvalsh(sigma * roots.jet.hessian[1:])
    saddles = roots.valid[1:] & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
    if count == 1:
        saddles = jnp.zeros_like(saddles)
    saddle_score_rows = jnp.where(saddles, sigma * roots.jet.value[1:], -jnp.inf)
    saddle_index = 1 + jnp.argmax(saddle_score_rows)
    saddle_score = jnp.max(saddle_score_rows)
    saddle_live = jnp.any(saddles)
    third, fourth = topology._curvature_derivatives(
        field, roots.position[saddle_index], jnp.median(geometry.pitch)
    )
    form = topology.saddle_normal_form(
        roots.position[saddle_index],
        sigma * roots.jet.hessian[saddle_index],
        sigma * third,
        policy.hessian_tolerance,
        sigma * fourth,
    )
    axis_sector = _local_axis_sector(geometry, axis, form)
    radius = jnp.maximum(
        policy.normal_form_radius,
        policy.normal_form_pitch_floor * jnp.median(geometry.pitch),
    )
    near_saddle = topology._inside_cells(geometry, form.position) | (
        jnp.linalg.norm(geometry.centre - form.position, axis=1) < radius
    )
    raw_wall_index = jnp.argmax(sigma * wall_flux)
    raw_wall_score = sigma * wall_flux[raw_wall_index]
    initial_score = jnp.where(saddle_live, saddle_score, raw_wall_score)
    approximate = FastRead(
        axis,
        axis_flux,
        roots.position[saddle_index],
        sigma * initial_score,
        jnp.asarray(0, jnp.int32),
        sigma * raw_wall_score,
        sigma * saddle_score,
        jnp.zeros(len(geometry.centre)),
        jnp.zeros(len(geometry.centre)),
        jnp.ones(len(geometry.centre)),
        jnp.zeros(len(geometry.centre), bool),
        field.sampled_coefficient(geometry),
        form,
        saddle_live & near_saddle,
        jnp.zeros(len(geometry.centre), bool),
        jnp.zeros((len(geometry.centre), 2), bool),
        jnp.zeros((len(geometry.centre), 2, geometry.vertices.shape[1], 13, 2)),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    seed = jnp.argmin(jnp.sum((geometry.centre - axis) ** 2, axis=1))
    wall_cell = jax.vmap(
        lambda point: jnp.argmax(topology._inside_cells(geometry, point))
    )(wall_points)
    ceiling = _wall_path_ceiling(
        geometry,
        approximate.field_coefficients,
        form,
        axis,
        radius,
        saddle_live,
        saddle_score,
        seed,
        sigma,
        frame_gate,
    )
    wall_scores = _wall_effective_level(sigma * wall_flux, ceiling[wall_cell])
    wall_index = jnp.argmax(wall_scores)
    wall_score = wall_scores[wall_index]
    diverted = saddle_live & (saddle_score >= wall_score)
    level_signed = jnp.maximum(wall_score, saddle_score)
    level = sigma * level_signed
    boundary = jnp.where(
        diverted, roots.position[saddle_index], wall_points[wall_index]
    )
    normal = saddle_live & near_saddle
    approximate = approximate._replace(
        boundary=boundary,
        boundary_flux=level,
        boundary_class=diverted.astype(jnp.int32),
        wall_candidate=sigma * wall_score,
        normal_form_cells=normal,
    )
    fraction, ordinary_valid, four_crossings = bell_fraction_with_crossings(
        geometry, approximate, sigma
    )
    saddle_owner = topology._inside_cells(geometry, form.position)
    booking = normal & (saddle_owner | four_crossings | ~ordinary_valid)
    capacity = min(16, len(geometry.centre))
    chosen = jnp.nonzero(booking, size=capacity, fill_value=0)[0]
    fragments = jax.vmap(
        topology._normal_form_cell_fragments, in_axes=(0, 0, None, None)
    )(
        geometry.vertices[chosen],
        geometry.vertex_count[chosen],
        form,
        policy.edge_tolerance,
    )
    positive = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    core = positive == axis_sector
    sector_area = fragments.area[:, positive]
    core_area = jnp.sum(jnp.where(core[None], sector_area, 0.0), axis=1)
    sector_interval = fragments.edge_interval[:, positive]
    row = jnp.arange(capacity) < jnp.sum(booking)
    saddle_fraction = (
        jnp.zeros(len(geometry.centre), dtype=fraction.dtype)
        .at[chosen]
        .add(jnp.where(row, core_area / geometry.full_area[chosen], 0.0))
    )
    intervals = (
        jnp.zeros(
            (len(geometry.centre), 2, geometry.vertices.shape[1], 13, 2),
            dtype=sector_interval.dtype,
        )
        .at[chosen]
        .add(jnp.where(row[:, None, None, None, None], sector_interval, 0))
    )
    selected_sector = (
        jnp.broadcast_to(core, (len(geometry.centre), 2)) & booking[:, None]
    )
    provisional = approximate._replace(
        membership=jnp.where(booking, saddle_fraction, fraction),
        booking_cells=booking,
        fragment_selected=selected_sector,
        edge_interval=intervals,
    )
    if frame_gate:
        connected = ceiling >= level_signed
        path_weight = connected.astype(fraction.dtype)
    else:
        from scripts.prototypes.sub_millisecond.measure_wall_march import (
            connected_cells as ungated,
        )

        connected = ungated(geometry, provisional, sigma, seed)
        path_weight = connected.astype(fraction.dtype)
    membership = provisional.membership * path_weight
    valid = (
        roots.valid[0]
        & jnp.all(jnp.isfinite(approximate.field_coefficients))
        & jnp.all(jnp.isfinite(membership))
        & (jnp.sum(booking) <= capacity)
        & jnp.all(jnp.where(row, fragments.valid, True))
        & jnp.all(
            jnp.where(connected & ~booking & (fraction > 0), ordinary_valid, True)
        )
    )
    return provisional._replace(
        membership=membership,
        geometric_membership=provisional.membership,
        path_weight=path_weight,
        selected_cells=connected,
        fragment_selected=selected_sector & connected[:, None],
        qualified=valid & (~diverted | form.valid),
        valid=valid,
    )
