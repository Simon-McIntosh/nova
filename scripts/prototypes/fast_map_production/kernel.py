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
    closed_loop: jax.Array | None = None
    closed_live: jax.Array | None = None


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


def _wall_path_ceiling(
    geometry, coefficients, form, axis, normal, saddle_score, seed, sigma, frame_gate
):
    """Find the highest level that can reach each cell from the axis."""
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
        sector = topology._normal_form_sectors(midpoint, form)[..., axis_sector]
        barred = (normal[:, None] & ~sector) | (normal[peer] & ~sector[peer, peer_edge])
        ceiling = jnp.where(barred, jnp.minimum(ceiling, saddle_score), ceiling)
    ceiling = jnp.where(geometry.neighbour >= 0, ceiling, -jnp.inf)
    best = jnp.full((len(geometry.centre),), -jnp.inf).at[seed].set(jnp.inf)

    def relax(_, current):
        passage = jnp.minimum(current[peer], ceiling)
        return jnp.maximum(current, jnp.max(passage, axis=1))

    return jax.lax.fori_loop(0, len(geometry.centre), relax, best)


def _wall_effective_level(tangency, bottleneck):
    """Effective level of a wall contact.

    The more scrape-off-side of the contact's own tangency level and the
    bottleneck of the best axis path that reaches it. A private-flux contact
    whose tangency is more axis-ward than the X-point, but which the axis
    reaches only through the X-point, therefore takes the saddle's level rather
    than its own. Continuous in both operands and free of any reachability gate.
    """
    return jnp.minimum(tangency, bottleneck)


def _flux_past_level_weight(peak, trough, level):
    """Share of a cell's own flux excursion that lies past ``level``.

    ``peak`` and ``trough`` bracket the flux the read's quadratic covers inside
    one cell, so their difference is the cell's own resolution of the level. The
    weight is one where the whole excursion is past the level and zero where
    none of it is, with the linear share between. Its slope is one over an
    excursion the map itself carries, which is what keeps the reported
    derivative on the scale of the values it differentiates. A width fixed in
    absolute flux is unresolvable at any step a solve can take, and the
    derivative handed to the solver then disagrees with the values by whatever
    that step happens to straddle.
    """
    span = peak - trough
    safe = jnp.where(span > 0, span, jnp.ones_like(span))
    return jnp.where(
        span > 0,
        jnp.clip((peak - level) / safe, 0, 1),
        (peak >= level).astype(peak.dtype),
    )


def _continuous_path_weight(
    geometry, coefficients, form, axis, seed, sigma, level, radius, saddle_live
):
    """Propagate the strongest smooth edge passage from the magnetic axis."""
    vertices = geometry.vertices
    count = geometry.vertex_count
    slot = jnp.arange(vertices.shape[1])
    following = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=1)
    midpoint = (vertices + end) / 2
    local = (vertices - geometry.centre[:, None]) / geometry.pitch[:, None, None]
    local_middle = (midpoint - geometry.centre[:, None]) / geometry.pitch[:, None, None]
    local_end = (end - geometry.centre[:, None]) / geometry.pitch[:, None, None]

    def signed(position):
        x, y = position[..., 0], position[..., 1]
        return sigma * (
            coefficients[:, 0, None]
            + coefficients[:, 1, None] * x
            + coefficients[:, 2, None] * y
            + coefficients[:, 3, None] * x * x
            + coefficients[:, 4, None] * x * y
            + coefficients[:, 5, None] * y * y
        )

    first, middle, last = signed(local), signed(local_middle), signed(local_end)
    curvature = 2 * (first + last - 2 * middle)
    slope = last - first - curvature
    fraction = jnp.clip(-slope / jnp.where(curvature < 0, 2 * curvature, 1), 0, 1)
    turning = signed(local + fraction[..., None] * (local_end - local))
    peak = jnp.maximum(jnp.maximum(first, last), turning)
    trough = jnp.minimum(jnp.minimum(first, last), turning)
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_edge = jnp.maximum(geometry.neighbour_edge, 0)
    flux_weight = jnp.minimum(
        _flux_past_level_weight(peak, trough, level),
        _flux_past_level_weight(peak[peer, peer_edge], trough[peer, peer_edge], level),
    )

    axis_sector = _local_axis_sector(geometry, axis, form)
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
    sector_margin = jnp.minimum(side[..., axis_sector], -next_side[..., axis_sector])
    length_width = jnp.maximum(1e-7, 1e-4 * jnp.median(geometry.pitch))
    sector_weight = jnp.clip(0.5 + sector_margin / length_width, 0, 1)
    distance = jnp.linalg.norm(geometry.centre - form.position, axis=1)
    near_weight = jnp.clip(0.5 + (radius - distance) / length_width, 0, 1)
    near_weight = jnp.maximum(
        near_weight,
        topology._inside_cells(geometry, form.position).astype(near_weight.dtype),
    )
    near_weight = jnp.where(saddle_live, near_weight, 0)
    frame_weight = 1 - near_weight[:, None] * (1 - sector_weight)
    frame_weight = jnp.minimum(frame_weight, frame_weight[peer, peer_edge])
    edge_weight = jnp.where(
        geometry.neighbour >= 0, jnp.minimum(flux_weight, frame_weight), 0
    )
    strength = (
        jnp.zeros((len(geometry.centre),), dtype=edge_weight.dtype).at[seed].set(1)
    )

    def relax(_, current):
        incoming = jnp.minimum(current[peer], edge_weight)
        return jnp.maximum(current, jnp.max(incoming, axis=1))

    return jax.lax.fori_loop(0, len(geometry.centre), relax, strength)


def psi_rounding_enclosure(coefficients):
    """Binary64 enclosure of the fitted psi at a cell centre.

    The Bell coefficients are one linear fit, but the fit's absolute row sum is
    set by every star in the cell, so this bound is a property of the cell's
    representation rather than of any particular level. It bounds the fitted
    level's arithmetic error; it does not decide whether two stationary points
    are separated features.
    """
    row_sum = jnp.abs(coefficients).sum(axis=1)
    return 2.0 * jnp.finfo(coefficients.dtype).eps * jnp.max(row_sum)


# A root-saddle pair resolved to fewer than this many units in the last place of
# their own levels is one stationary feature, not two. The read refuses such a
# pair rather than publishing its upper member as the magnetic axis.
PERSISTENCE_ULP_FLOOR = 8.0


def _first_tree_events(field, geometry, axis, sigma, policy):
    """Follow descending authority arcs from the axis to its first events."""
    from nova.equilibrium.contour_tree import build_contour_tree

    mesh = geometry.contour_mesh
    values = jax.vmap(field.value)(geometry.contour_sample_points)
    vertex_flux = jnp.sum(
        values[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    tree = build_contour_tree(
        vertex_flux,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        sigma,
    )
    vertices = jnp.maximum(tree.node_vertex, 0)
    points = mesh.vertex_rz[vertices]
    maximum = tree.node_valid & (tree.critical_type == jnp.where(sigma > 0, 2, 0))
    start = jnp.argmin(
        jnp.where(maximum, jnp.sum((points - axis) ** 2, axis=1), jnp.inf)
    )
    levels = sigma * tree.node_psi
    wall = mesh.vertex_is_wall[vertices] & tree.node_valid
    left, right = tree.edges.T

    def advance(_, state):
        at, saddle_node, wall_node, active = state
        other = jnp.where(left == at, right, left)
        adjacent = tree.edge_valid & ((left == at) | (right == at))
        descending = (levels[other] < levels[at]) | (
            (levels[other] == levels[at]) & (other < at)
        )
        following = adjacent & descending & tree.node_valid[other]
        unique = jnp.sum(following) == 1
        next_node = other[jnp.argmax(following)]
        move = active & unique
        is_wall = wall[next_node]
        is_saddle = (tree.critical_type[next_node] == 1) & ~is_wall
        saddle_node = jnp.where(
            move & is_saddle & (saddle_node < 0), next_node, saddle_node
        )
        wall_node = jnp.where(move & is_wall & (wall_node < 0), next_node, wall_node)
        return (
            jnp.where(move, next_node, at),
            saddle_node,
            wall_node,
            move & ((saddle_node < 0) | (wall_node < 0)),
        )

    _, saddle_node, wall_node, _ = jax.lax.fori_loop(
        0,
        tree.node_valid.size,
        advance,
        (start, jnp.asarray(-1), jnp.asarray(-1), ~tree.overflow & jnp.any(maximum)),
    )
    saddle_seed = points[jnp.maximum(saddle_node, 0)]
    polished = topology.stationary_read(
        field, saddle_seed, jnp.median(geometry.pitch), policy
    )
    eigen = jnp.linalg.eigvalsh(sigma * polished.jet.hessian)
    saddle_live = (saddle_node >= 0) & polished.valid & (eigen[0] < 0) & (eigen[1] > 0)
    valid = ~tree.overflow & jnp.any(maximum) & ((wall_node >= 0) | saddle_live)
    return (
        polished,
        saddle_live,
        points[jnp.maximum(wall_node, 0)],
        wall_node >= 0,
        valid,
    )


def read(field, roots, geometry, convention, policy, *, frame_gate=True):
    """Read both boundary candidates and axis-connected Bell support at fixed shape."""
    sigma = convention.sigma
    coefficients = field.sampled_coefficient(geometry)
    eigenvalues = jnp.linalg.eigvalsh(sigma * roots.jet.hessian)
    owner_cells = jax.vmap(lambda point: topology._inside_cells(geometry, point))(
        roots.position
    )
    inside = jnp.any(owner_cells, axis=1)
    extrema = (
        roots.valid
        & inside
        & jnp.all(jnp.isfinite(roots.position), axis=1)
        & jnp.isfinite(roots.jet.value)
        & jnp.all(eigenvalues < 0, axis=1)
    )
    saddle = roots.valid & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
    separation = jnp.linalg.norm(
        roots.position[:, None] - roots.position[None], axis=-1
    )
    nearest = jnp.argmin(jnp.where(saddle[None], separation, jnp.inf), axis=1)
    persistence = sigma * (roots.jet.value - roots.jet.value[nearest])
    # Persistence is the difference of two stationary levels, so what decides
    # whether the pair is resolved is the binary64 resolution of those levels
    # themselves: a few units in the last place of the larger one. A pair whose
    # signed separation sits inside that floor is one number rather than two
    # features, however far apart the two roots lie on the plane.
    level_resolution = (
        PERSISTENCE_ULP_FLOOR
        * jnp.finfo(roots.jet.value.dtype).eps
        * jnp.maximum(jnp.abs(roots.jet.value), jnp.abs(roots.jet.value[nearest]))
    )
    persistent = ~jnp.any(saddle) | (persistence > level_resolution)
    eligible = extrema & persistent
    axis_index = jnp.argmax(jnp.where(eligible, sigma * roots.jet.value, -jnp.inf))
    axis_live = jnp.any(eligible)
    axis = roots.position[axis_index]
    axis_flux = roots.jet.value[axis_index]
    wall_points, wall_flux = topology._wall_events(field, geometry, policy, sigma)
    count = roots.position.shape[0]
    if count == 1:
        roots = jax.tree.map(
            lambda leaf: jnp.concatenate((leaf, jnp.zeros_like(leaf)), axis=0), roots
        )
        saddle = jnp.concatenate((saddle, jnp.zeros_like(saddle)))
        inside = jnp.concatenate((inside, jnp.zeros_like(inside)))
    saddle = saddle & inside
    polished, saddle_live, wall_seed, wall_live, tree_valid = _first_tree_events(
        field, geometry, axis, sigma, policy
    )
    saddle_index = jnp.argmin(
        jnp.where(
            saddle, jnp.sum((roots.position - polished.position) ** 2, axis=1), jnp.inf
        )
    )
    saddle_point = jnp.where(
        saddle_live, polished.position, roots.position[saddle_index]
    )
    saddle_hessian = jnp.where(
        saddle_live, polished.jet.hessian, roots.jet.hessian[saddle_index]
    )
    saddle_score = sigma * polished.jet.value
    wall_distance = jnp.sum((wall_points - wall_seed) ** 2, axis=1)
    tree_wall_index = jnp.argmin(
        jnp.where(jnp.isfinite(wall_flux), wall_distance, jnp.inf)
    )
    wall_passage = jnp.where(
        wall_live & (jnp.arange(wall_flux.size) == tree_wall_index), jnp.inf, -jnp.inf
    )
    third, fourth = topology._curvature_derivatives(
        field, saddle_point, jnp.median(geometry.pitch)
    )
    form = topology.saddle_normal_form(
        saddle_point,
        sigma * saddle_hessian,
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
        saddle_point,
        sigma * initial_score,
        jnp.asarray(0, jnp.int32),
        sigma * raw_wall_score,
        sigma * saddle_score,
        jnp.zeros(len(geometry.centre)),
        jnp.zeros(len(geometry.centre)),
        jnp.ones(len(geometry.centre)),
        jnp.zeros(len(geometry.centre), bool),
        coefficients,
        form,
        saddle_live & near_saddle,
        jnp.zeros(len(geometry.centre), bool),
        jnp.zeros((len(geometry.centre), 2), bool),
        jnp.zeros((len(geometry.centre), 2, geometry.vertices.shape[1], 13, 2)),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    seed = jnp.argmin(jnp.sum((geometry.centre - axis) ** 2, axis=1))
    wall_scores = _wall_effective_level(sigma * wall_flux, wall_passage)
    wall_index = jnp.argmax(wall_scores)
    wall_score = wall_scores[wall_index]
    diverted = saddle_live & (saddle_score >= wall_score)
    level = sigma * jnp.maximum(
        jnp.where(saddle_live, saddle_score, -jnp.inf), wall_score
    )
    boundary = jnp.where(diverted, saddle_point, wall_points[wall_index])
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
        path_weight = _continuous_path_weight(
            geometry,
            provisional.field_coefficients,
            form,
            axis,
            seed,
            sigma,
            sigma * level,
            radius,
            saddle_live,
        )
        connected = path_weight > 0
    else:
        from scripts.prototypes.sub_millisecond.measure_wall_march import (
            connected_cells as ungated,
        )

        connected = ungated(geometry, provisional, sigma, seed)
        path_weight = connected.astype(fraction.dtype)
    closed_loop, closed_live = topology.normal_form_fold_loop(
        form.position,
        sigma * saddle_hessian,
        sigma * third,
        sigma * fourth,
        axis,
        sigma * level - saddle_score,
        radius,
    )
    closed_live &= saddle_live

    def closed_fractions(loop):
        polygons, counts = jax.vmap(
            topology.clip_normal_form_loop, in_axes=(None, 0, 0)
        )(loop, geometry.vertices, geometry.vertex_count)
        return polygon_area(polygons, counts) / geometry.full_area

    closed_fraction = jax.lax.cond(
        closed_live, closed_fractions, lambda _: jnp.zeros_like(fraction), closed_loop
    )
    closed_cells = closed_fraction > 0
    provisional = provisional._replace(
        membership=jnp.where(closed_live, closed_fraction, provisional.membership),
        booking_cells=jnp.where(closed_live, closed_cells, provisional.booking_cells),
        closed_loop=closed_loop,
        closed_live=closed_live,
    )
    # One explicitly closed lobe owns every clipped piece; its graph cannot
    # cross to the opposing saddle branch. Keep the ordinary path weight for
    # components represented by the open normal form.
    path_weight = jnp.where(
        closed_live, closed_cells.astype(path_weight.dtype), path_weight
    )
    connected = jnp.where(closed_live, closed_cells, connected)
    membership = provisional.membership * path_weight
    attained = field.value(boundary)
    boundary_enclosure = (
        128
        * jnp.finfo(level.dtype).eps
        * jnp.maximum(jnp.abs(level), jnp.abs(attained))
    )
    boundary_attained = jnp.isfinite(level) & (
        jnp.abs(attained - level) <= boundary_enclosure
    )
    valid = (
        axis_live
        & tree_valid
        & boundary_attained
        & jnp.all(jnp.isfinite(approximate.field_coefficients))
        & jnp.all(jnp.isfinite(membership))
        & (jnp.sum(provisional.booking_cells) <= capacity)
        & (closed_live | jnp.all(jnp.where(row, fragments.valid, True)))
        & jnp.all(
            jnp.where(
                ~closed_live & connected & ~booking & (fraction > 0),
                ordinary_valid,
                True,
            )
        )
    )
    return provisional._replace(
        axis=jnp.where(axis_live, provisional.axis, jnp.full_like(axis, jnp.nan)),
        axis_flux=jnp.where(axis_live, provisional.axis_flux, jnp.nan),
        membership=membership,
        geometric_membership=provisional.membership,
        path_weight=path_weight,
        selected_cells=connected,
        fragment_selected=selected_sector & connected[:, None],
        qualified=valid & (~diverted | form.valid),
        valid=valid,
    )
