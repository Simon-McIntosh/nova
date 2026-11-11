"""Book current on the topology read's connected fragments.

The sampled physical flux at ``geometry.sample_points`` and its topology read
must come from the same state. Fragment identity is discrete; the quadratic
coefficients, null levels, geometric intersections and normalization remain
traced operands. Ring winding carries subtraction, including holes.
"""

from dataclasses import dataclass, field as dataclass_field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata

from nova.equilibrium import topology
from nova.equilibrium.clip_quadrature import (
    ClippedCurrentMoments,
    _quadrature_from_arrays,
    clipped_support_current_moments,
    fixed_polygon_current_moment_tangent,
    moving_polygon_current_moment_tangent,
    moving_polygon_current_moment_tangents,
)
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial


class FragmentSupport(NamedTuple):
    """Oriented rings, padded independently of their number in each cell."""

    support_vertices: jax.Array
    vertex_count: jax.Array
    centroids: jax.Array
    included: jax.Array
    boundary: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Quadrature:
    """Polygon resolution and optional prescribed total current.

    ``sigma`` is the signed-flux convention supplied to the topology read.
    ``segments`` sets cosine-spaced conic chords and cubic-ray chords, without
    changing fragment ownership. No polygon-count limit is imposed here.
    ``cut_batch_width`` controls the conic tile of ``book_tangent_channels``
    only and defaults to 32. Scalar ``book`` and ``book_tangent`` use fixed
    128-cell cut tiles; a narrower tile repeats conic strip propagation.
    Normal-form buckets use eight cells independently of the conic tile.
    """

    sigma: object = 1.0
    target_current: object = None
    segments: int = dataclass_field(default=512, metadata={"static": True})
    cut_batch_width: int = dataclass_field(default=32, metadata={"static": True})


class Booking(NamedTuple):
    """Current moments and the normalization and qualification ledger."""

    cell_current: jax.Array
    radial_moment: jax.Array
    vertical_moment: jax.Array
    raw_current: jax.Array
    normalization: jax.Array
    valid: jax.Array


_SCALAR_CUT_BATCH_WIDTH = 128
_NORMAL_CUT_BATCH_WIDTH = 8


def _inward_interval_points(x, left, right):
    """Use the strip-interior limit for both booking and its tangent.

    At an event, root ordering can choose the adjoining interval on roundoff.
    Sampling inside the owned strip keeps primal rings and their velocities
    on the same branch. Polygon x coordinates still end on the event.
    """
    distance = jnp.minimum(1e-12, 0.25 * (right - left))[:, None]
    slot = jnp.arange(x.shape[1])[None]
    return jnp.where(
        slot == 0, x + distance, jnp.where(slot == x.shape[1] - 1, x - distance, x)
    )


def _conic_rings(
    vertices,
    count,
    centre,
    pitch,
    coefficient,
    selected,
    segments,
):
    """Trace cosine-spaced rings on each selected algebraic conic strip."""
    local = (vertices - centre) / pitch
    fragments = topology.quadratic_cell_fragments(
        local, count, coefficient, selected.shape[0]
    )
    left, right = fragments.slice_breaks[:-1], fragments.slice_breaks[1:]
    parameter = 0.5 * (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, segments + 1)))
    x = left[:, None] + (right - left)[:, None] * parameter
    interval_x = _inward_interval_points(x, left, right)
    bottom, top, _ = topology._positive_vertical_intervals(
        local, count, coefficient, interval_x
    )
    x = jnp.broadcast_to(x[..., None], bottom.shape)
    lower = jnp.stack((x, bottom), axis=-1)
    upper = jnp.stack((x[:, ::-1], top[:, ::-1]), axis=-1)
    rings = jnp.transpose(jnp.concatenate((lower, upper), axis=1), (0, 2, 1, 3))
    labels = fragments.slice_labels
    live = (labels >= 0) & selected[jnp.maximum(labels, 0)]
    live &= (right > left)[:, None]
    rings = centre + pitch * rings.reshape(-1, 2 * (segments + 1), 2)
    live = live.reshape(-1)
    rings = jnp.where(live[:, None, None], rings, centre)
    return rings, jnp.where(live, rings.shape[1], 0), fragments.valid


def _conic_ring_tangents(
    vertices,
    count,
    centre,
    pitch,
    coefficient,
    tangent,
    selected,
    segments,
):
    """Move each sampled conic ring vertex by the implicit root rule."""
    local = (vertices - centre) / pitch
    fragments = topology.quadratic_cell_fragments(
        local, count, coefficient, selected.shape[0]
    )
    _, break_speed = topology.quadratic_slice_break_tangent(
        local, count, coefficient, tangent
    )
    left, right = fragments.slice_breaks[:-1], fragments.slice_breaks[1:]
    left_speed, right_speed = break_speed[:-1], break_speed[1:]
    parameter = 0.5 * (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, segments + 1)))
    x = left[:, None] + (right - left)[:, None] * parameter
    interval_x = _inward_interval_points(x, left, right)
    x_speed = left_speed[:, None] + (right_speed - left_speed)[:, None] * parameter
    bottom, top, _ = topology._positive_vertical_intervals(
        local, count, coefficient, interval_x
    )
    polygon_lower, polygon_upper = topology._vertical_polygon_interval(
        local, count, interval_x
    )
    slot = jnp.arange(local.shape[0])
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    edge = local[following] - local
    fraction = (interval_x[..., None] - local[:, 0]) / jnp.where(
        edge[:, 0] != 0, edge[:, 0], 1
    )
    live_edge = (
        (slot < count)
        & (edge[:, 0] != 0)
        & (fraction >= -1e-12)
        & (fraction <= 1 + 1e-12)
    )
    edge_y = local[:, 1] + fraction * edge[:, 1]
    edge_slope = edge[:, 1] / jnp.where(edge[:, 0] != 0, edge[:, 0], 1)
    lower_y = jnp.min(jnp.where(live_edge, edge_y, jnp.inf), axis=-1)
    upper_y = jnp.max(jnp.where(live_edge, edge_y, -jnp.inf), axis=-1)
    probe = interval_x + 1e-9 * jnp.sign(x_speed)
    probe_fraction = (probe[..., None] - local[:, 0]) / jnp.where(
        edge[:, 0] != 0, edge[:, 0], 1
    )
    forward_edge = (
        (slot < count)
        & (edge[:, 0] != 0)
        & (probe_fraction >= -1e-12)
        & (probe_fraction <= 1 + 1e-12)
    )

    def owning_edge(bound):
        near = live_edge & (jnp.abs(edge_y - bound[..., None]) < 1e-8)
        moving = near & forward_edge
        candidates = jnp.where(jnp.any(moving, axis=-1, keepdims=True), moving, near)
        return jnp.argmin(jnp.where(candidates, jnp.abs(edge_slope), jnp.inf), axis=-1)

    # At a polygon vertex, use the edge entered as the slice moves. Two edges
    # share the same height there, so a raw argmin/argmax can flip on roundoff.
    lower_edge = owning_edge(lower_y)
    upper_edge = owning_edge(upper_y)
    lower_speed = edge_slope[lower_edge] * x_speed
    upper_speed = edge_slope[upper_edge] * x_speed

    def vertical_speed(y):
        x_cell = interval_x[..., None]
        dx_cell = x_speed[..., None]
        field_speed = (
            tangent[0]
            + tangent[1] * x_cell
            + tangent[2] * y
            + tangent[3] * x_cell**2
            + tangent[4] * x_cell * y
            + tangent[5] * y**2
        )
        radial_gradient = (
            coefficient[1] + 2 * coefficient[3] * x_cell + coefficient[4] * y
        )
        vertical_gradient = (
            coefficient[2] + coefficient[4] * x_cell + 2 * coefficient[5] * y
        )
        root_speed = -(field_speed + radial_gradient * dx_cell) / jnp.where(
            vertical_gradient != 0, vertical_gradient, 1
        )
        # A conic root at a polygon vertex belongs to the polygon edge. The
        # two interval evaluations can differ at this degenerate endpoint.
        on_lower = jnp.abs(y - polygon_lower[..., None]) < 1e-8
        on_upper = jnp.abs(y - polygon_upper[..., None]) < 1e-8
        return jnp.where(
            on_lower,
            lower_speed[..., None],
            jnp.where(on_upper, upper_speed[..., None], root_speed),
        )

    lower_velocity = jnp.stack(
        (jnp.broadcast_to(x_speed[..., None], bottom.shape), vertical_speed(bottom)),
        axis=-1,
    )
    upper_velocity = jnp.stack(
        (jnp.broadcast_to(x_speed[..., None], top.shape), vertical_speed(top)),
        axis=-1,
    )
    velocity = jnp.transpose(
        jnp.concatenate((lower_velocity, upper_velocity[:, ::-1]), axis=1),
        (0, 2, 1, 3),
    ).reshape(-1, 2 * (segments + 1), 2)
    labels = fragments.slice_labels
    live = (labels >= 0) & selected[jnp.maximum(labels, 0)]
    live &= (right > left)[:, None]
    return jnp.where(live.reshape(-1, 1, 1), pitch * velocity, 0.0)


def _closed_normal_rings(loop, vertices, count):
    """Return the clipped ring of the axis-owned closed normal-form lobe."""
    polygon, size = topology.clip_normal_form_loop(loop, vertices, count)
    return polygon[None], size[None]


def _normal_rings(vertices, count, centre, form, selected, intervals, segments):
    """Carry selected sector edges and cubic rays as signed fan rings.

    Straight intervals are owned by the read. Ray intersections use the same
    cubic geometry as its normal-form sectors; no flux level root is involved.
    Opposing fan edges cancel, so holes and disjoint components need no union.
    """
    width = vertices.shape[0]
    slot = jnp.arange(width)
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    edges = vertices[following] - vertices
    live_edge = slot < count
    chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    sector_live = jnp.zeros(4, dtype=bool).at[chosen].set(selected[:2])
    lower, upper = intervals[..., 0], intervals[..., 1]
    live = selected[:, None, None] & (upper > lower) & live_edge[None, :, None]
    first = vertices[None, :, None, :] + lower[..., None] * edges[None, :, None, :]
    last = vertices[None, :, None, :] + upper[..., None] * edges[None, :, None, :]
    first = first.reshape(-1, 2)
    last = last.reshape(-1, 2)
    straight = jnp.stack((jnp.broadcast_to(centre, first.shape), first, last), axis=1)
    straight_count = jnp.where(live.reshape(-1), 3, 0)

    relative = vertices - form.position
    cross = topology._cross_plane
    roots = topology._cubic_parameters(
        cross(form.cubic[:, None, :], edges[None]),
        cross(form.curvature[:, None, :], edges[None]),
        cross(form.direction[:, None, :], edges[None]),
        -cross(relative[None], edges[None]),
    )
    safe = jnp.where(jnp.isfinite(roots), roots, 0.0)
    points = (
        safe[..., None] * form.direction[:, None, None]
        + safe[..., None] ** 2 * form.curvature[:, None, None]
        + safe[..., None] ** 3 * form.cubic[:, None, None]
    )
    square = jnp.sum(edges * edges, axis=-1)
    fraction = (
        jnp.sum((points - relative[None, :, None]) * edges[None, :, None], axis=-1)
        / jnp.where(square > 0, square, 1.0)[None, :, None]
    )
    hit = (
        live_edge[None, :, None]
        & jnp.isfinite(roots)
        & (roots >= -1e-12)
        & (fraction >= -1e-12)
        & (fraction <= 1 + 1e-12)
    )
    bound = 2 * jnp.max(jnp.linalg.norm(relative, axis=-1))
    parameters = jnp.sort(
        jnp.clip(jnp.where(hit, roots, jnp.inf).reshape(4, -1), 0.0, bound), axis=-1
    )
    parameters = jnp.concatenate(
        (jnp.zeros((4, 1)), parameters, jnp.full((4, 1), bound)), axis=-1
    )
    begin, end = parameters[:, :-1], parameters[:, 1:]
    middle = (begin + end) / 2
    curve = (
        middle[..., None] * form.direction[:, None]
        + middle[..., None] ** 2 * form.curvature[:, None]
        + middle[..., None] ** 3 * form.cubic[:, None]
    )
    inside = jnp.all(
        jnp.where(
            live_edge[None, None],
            cross(edges[None, None], curve[:, :, None] - relative[None, None])
            >= -1e-12,
            True,
        ),
        axis=-1,
    )
    sign = sector_live.astype(vertices.dtype) - jnp.roll(sector_live, 1).astype(
        vertices.dtype
    )
    active = inside & (end > begin) & (sign[:, None] != 0)
    parameter = jnp.linspace(0.0, 1.0, segments + 1)
    t = begin[..., None] + (end - begin)[..., None] * parameter
    arcs = (
        form.position
        + t[..., None] * form.direction[:, None, None]
        + t[..., None] ** 2 * form.curvature[:, None, None]
        + t[..., None] ** 3 * form.cubic[:, None, None]
    )
    arcs = jnp.where((sign < 0)[:, None, None, None], arcs[:, :, ::-1], arcs)
    arcs = arcs.reshape(-1, segments + 1, 2)
    arcs = jnp.concatenate((jnp.broadcast_to(centre, (len(arcs), 1, 2)), arcs), axis=1)
    arcs = jnp.where(active.reshape(-1, 1, 1), arcs, centre)
    straight = jnp.pad(straight, ((0, 0), (0, segments - 1), (0, 0)))
    return (
        jnp.concatenate((straight, arcs)),
        jnp.concatenate(
            (straight_count, jnp.where(active.reshape(-1), segments + 2, 0))
        ),
    )


def _integrate_live_rings(rings, counts, evaluate, zero, width=4):
    """Integrate compact live rings in bounded groups without a capacity cutoff."""
    width = min(width, len(counts))
    live = counts >= 3
    count = jnp.sum(live, dtype=jnp.int32)
    indices = jnp.nonzero(live, size=len(counts), fill_value=0)[0]
    indices = jnp.pad(indices, (0, (-len(counts)) % width))

    def add(batch, total):
        start = batch * width
        selected = jax.lax.dynamic_slice(indices, (start,), (width,))
        active = start + jnp.arange(width) < count
        selected_counts = jnp.where(active, counts[selected], 0)
        value = evaluate(rings[selected], selected_counts, selected)
        return jax.tree.map(lambda prior, addition: prior + addition, total, value)

    return jax.lax.fori_loop(0, (count + width - 1) // width, add, zero)


def _integrate_bucket(mask, width, evaluate, zero):
    """Evaluate one geometry population, skipping its empty padded buckets."""
    cells = len(mask)
    count = jnp.sum(mask, dtype=jnp.int32)
    indices = jnp.nonzero(mask, size=cells, fill_value=0)[0]
    indices = jnp.pad(indices, (0, (-cells) % width))

    def add_batch(batch, carried):
        start = batch * width
        selected = jax.lax.dynamic_slice(indices, (start,), (width,))
        active = start + jnp.arange(width) < count
        response = jax.vmap(evaluate)(selected)

        def scatter(total, value):
            shape = (width,) + (1,) * (value.ndim - 1)
            return total.at[selected].add(jnp.where(active.reshape(shape), value, 0.0))

        return jax.tree.map(scatter, carried, response)

    # A conditional under vmap evaluates both branches. The population bounds
    # this loop so an empty geometry bucket never integrates.
    return jax.lax.fori_loop(0, (count + width - 1) // width, add_batch, zero)


def book(psi, read, geometry, profile, quadrature=Quadrature()):
    """Integrate every selected fragment and normalize its current once.

    ``psi`` contains physical flux at the geometry's per-cell sample points.
    The supplied read must be its topology read; using a stale read is a caller
    error. Invalid reads and zero/nonfinite normalization denominators return
    an invalid receipt with nonfinite moments, including under jit and vmap.
    """
    if quadrature.segments < 2:
        raise ValueError("quadrature segments must be at least two")
    psi = jnp.asarray(psi)
    if psi.shape != geometry.sample_points.shape[:-1]:
        raise ValueError("psi must carry flux at every per-cell sample point")
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    span = read.boundary_flux - read.axis_flux
    safe_span = jnp.where(span != 0, span, 1.0)
    normalized = coefficients.at[:, 0].add(-read.axis_flux) / safe_span
    signed = quadrature.sigma * read.field_coefficients
    signed = signed.at[:, 0].add(-quadrature.sigma * read.boundary_flux)

    def cell_field(index):
        centre = geometry.centre[index]
        field = FluxFieldPolynomial(
            normalized[index][None],
            centre[None],
            jnp.full((1, 2), geometry.pitch[index]),
            jnp.ones(1, dtype=bool),
        )
        return centre, field

    def integrate(index, rings, counts):
        centre, field = cell_field(index)
        support = FragmentSupport(
            rings[None],
            counts[None],
            centre[None],
            jnp.ones(1, dtype=bool),
            jnp.ones(1, dtype=bool),
        )
        moments = clipped_support_current_moments(
            support,
            support.included,
            field,
            profile,
            cut_cell_capacity=1,
            boundary_reduction=False,
        )
        return jax.tree.map(lambda value: value[0], moments)

    def whole_cell(index):
        rings = geometry.vertices[index][None]
        counts = geometry.vertex_count[index][None]
        return integrate(index, rings, counts)

    def live_rings(index, rings, counts):
        zero = ClippedCurrentMoments(*(jnp.zeros((), rings.dtype) for _ in range(3)))
        return _integrate_live_rings(
            rings,
            counts,
            lambda selected, sizes, _: integrate(index, selected, sizes),
            zero,
        )

    def conic_cell(index):
        rings, counts, valid = _conic_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            geometry.pitch[index],
            signed[index],
            read.fragment_selected[index],
            quadrature.segments,
        )
        moments = live_rings(index, rings, counts)
        return jax.tree.map(lambda value: jnp.where(valid, value, jnp.nan), moments)

    def normal_cell(index):
        rings, counts = _normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            read.saddle_form,
            read.fragment_selected[index],
            read.edge_interval[index],
            quadrature.segments,
        )
        return live_rings(index, rings, counts)

    cells = len(geometry.centre)
    full = read.membership >= 1 - 1e-14
    cut = (read.membership > 0) & ~full
    whole = jax.vmap(whole_cell)(jnp.arange(cells))
    whole = jax.tree.map(lambda value: jnp.where(full, value, 0.0), whole)

    # Wide conic tiles avoid repeating strip propagation for small populations.
    # Normal-form cells retain their independent bounded tile.
    batch_width = min(_SCALAR_CUT_BATCH_WIDTH, cells)
    normal = cut & read.normal_form_cells
    zero = jax.tree.map(jnp.zeros_like, whole)
    conic = _integrate_bucket(cut & ~normal, batch_width, conic_cell, zero)
    normal = _integrate_bucket(
        normal, min(_NORMAL_CUT_BATCH_WIDTH, cells), normal_cell, zero
    )
    cuts = jax.tree.map(lambda left, right: left + right, conic, normal)
    moments = jax.tree.map(
        lambda full_value, cut_value: full_value + cut_value, whole, cuts
    )
    raw = jnp.sum(moments.cell_current)
    valid = read.valid & read.qualified & (span != 0) & jnp.isfinite(raw)
    if quadrature.target_current is None:
        normalization = jnp.ones_like(raw)
    else:
        valid &= (raw != 0) & jnp.isfinite(quadrature.target_current)
        normalization = quadrature.target_current / jnp.where(raw != 0, raw, 1.0)
    return Booking(
        *(jnp.where(valid, value * normalization, jnp.nan) for value in moments),
        raw,
        normalization,
        valid,
    )


def book_tangent(
    psi,
    psi_tangent,
    read,
    field_tangent,
    axis_tangent,
    boundary_tangent,
    geometry,
    profile,
    density_gradient,
    quadrature,
    booked,
) -> ClippedCurrentMoments:
    """Differentiate a booking with fragment ownership and cut cells fixed.

    The interior term differentiates normalized flux at the existing polygon
    quadrature points. The moving-boundary term integrates current density on
    each selected quadratic level segment. Axis and boundary flux enter both
    the normalized field and the signed level; their tangents are supplied by
    the stationary-point envelope rule. Normal-form sector geometry is held by
    the same tracked read that owns the sector selection.
    """
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    coefficient_tangent = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi_tangent)
    span = read.boundary_flux - read.axis_flux
    span_tangent = boundary_tangent - axis_tangent
    normalized = coefficients.at[:, 0].add(-read.axis_flux) / span
    normalized_tangent = coefficient_tangent.at[:, 0].add(
        -axis_tangent
    ) / span - normalized * (span_tangent / span)
    signed = quadrature.sigma * read.field_coefficients
    signed = signed.at[:, 0].add(-quadrature.sigma * read.boundary_flux)
    signed_tangent = quadrature.sigma * field_tangent
    signed_tangent = signed_tangent.at[:, 0].add(-quadrature.sigma * boundary_tangent)

    def integrate(index, rings, counts):
        centre = geometry.centre[index]
        pitch = geometry.pitch[index]
        values = jax.vmap(
            lambda ring, count: fixed_polygon_current_moment_tangent(
                ring,
                count,
                centre,
                pitch,
                normalized[index],
                normalized_tangent[index],
                density_gradient,
            )
        )(rings, counts)
        return jax.tree.map(jnp.sum, values)

    def whole_cell(index):
        return integrate(
            index,
            geometry.vertices[index][None],
            geometry.vertex_count[index][None],
        )

    def cut_cell(index):
        centre = geometry.centre[index]
        vertices, count = geometry.vertices[index], geometry.vertex_count[index]

        def conic(_):
            rings, counts, valid = _conic_rings(
                vertices,
                count,
                centre,
                geometry.pitch[index],
                signed[index],
                read.fragment_selected[index],
                quadrature.segments,
            )
            interior = integrate(index, rings, counts)
            ring_velocity = _conic_ring_tangents(
                vertices,
                count,
                centre,
                geometry.pitch[index],
                signed[index],
                signed_tangent[index],
                read.fragment_selected[index],
                quadrature.segments,
            )
            boundary = jax.vmap(
                lambda ring, velocity, number: moving_polygon_current_moment_tangent(
                    ring,
                    velocity,
                    number,
                    centre,
                    geometry.pitch[index],
                    normalized[index],
                    profile,
                )
            )(rings, ring_velocity, counts)
            boundary = jax.tree.map(jnp.sum, boundary)
            return jax.tree.map(
                lambda a, b: jnp.where(valid, a + b, jnp.nan), interior, boundary
            )

        def normal(_):
            rings, counts = _normal_rings(
                vertices,
                count,
                centre,
                read.saddle_form,
                read.fragment_selected[index],
                read.edge_interval[index],
                quadrature.segments,
            )
            return integrate(index, rings, counts)

        return jax.lax.cond(read.normal_form_cells[index], normal, conic, None)

    cells = len(geometry.centre)
    full = read.membership >= 1 - 1e-14
    cut = (read.membership > 0) & ~full
    whole = jax.vmap(whole_cell)(jnp.arange(cells))
    whole = jax.tree.map(lambda value: jnp.where(full, value, 0.0), whole)
    batch_width = _SCALAR_CUT_BATCH_WIDTH
    cut_count = jnp.sum(cut, dtype=jnp.int32)
    cut_index = jnp.nonzero(cut, size=cells, fill_value=0)[0]
    cut_index = jnp.pad(cut_index, (0, (-cells) % batch_width))

    def add_batch(batch, carried):
        start = batch * batch_width
        indices = jax.lax.dynamic_slice(cut_index, (start,), (batch_width,))
        active = start + jnp.arange(batch_width) < cut_count
        values = jax.vmap(cut_cell)(indices)
        return jax.tree.map(
            lambda total, value: total.at[indices].add(jnp.where(active, value, 0.0)),
            carried,
            values,
        )

    cuts = jax.tree.map(jnp.zeros_like, whole)
    cuts = jax.lax.fori_loop(
        0, (cut_count + batch_width - 1) // batch_width, add_batch, cuts
    )
    raw_tangent = jax.tree.map(lambda a, b: a + b, whole, cuts)
    total_tangent = jnp.sum(raw_tangent.cell_current)
    scale_tangent = (
        jnp.zeros_like(total_tangent)
        if quadrature.target_current is None
        else -total_tangent / booked.raw_current
    )
    return ClippedCurrentMoments(
        *(
            jnp.where(
                booked.valid,
                booked.normalization * raw + scale_tangent * value,
                jnp.nan,
            )
            for raw, value in zip(raw_tangent, booked[:3], strict=True)
        )
    )


def book_tangent_channels(
    psi,
    psi_tangent,
    read,
    field_tangent,
    axis_tangent,
    boundary_tangent,
    geometry,
    profile,
    density_gradient,
    quadrature,
    booked,
    boundary_channels,
):
    """Evaluate every local booking channel over shared cell polygons."""
    cells, samples = psi.shape
    channels = psi_tangent.shape[0]
    if psi_tangent.shape != (channels, cells, samples):
        raise ValueError("flux tangents must carry a channel and every cell")
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    coefficient_rate = jnp.einsum("nij,cnj->cni", geometry.fit_inverse, psi_tangent)
    span = read.boundary_flux - read.axis_flux
    span_rate = boundary_tangent - axis_tangent
    normalized = coefficients.at[:, 0].add(-read.axis_flux) / span
    normalized_rate = coefficient_rate.at[:, :, 0].add(-axis_tangent[:, None]) / span
    normalized_rate -= normalized[None] * (span_rate / span)[:, None, None]
    signed = quadrature.sigma * read.field_coefficients
    signed = signed.at[:, 0].add(-quadrature.sigma * read.boundary_flux)
    signed_rate = quadrature.sigma * field_tangent
    signed_rate = signed_rate.at[:, :, 0].add(
        -quadrature.sigma * boundary_tangent[:, None]
    )

    def interior(index, rings, counts):
        centre = geometry.centre[index]
        pitch = geometry.pitch[index]
        points, weights = _quadrature_from_arrays(
            rings,
            counts,
            jnp.broadcast_to(centre, (len(rings), 2)),
            counts >= 3,
            signed=True,
        )
        local = (points - centre) / pitch
        radial, vertical = local[..., 0], local[..., 1]
        basis = jnp.stack(
            (
                jnp.ones_like(radial),
                radial,
                vertical,
                radial**2,
                radial * vertical,
                vertical**2,
            ),
            axis=-1,
        )
        flux = jnp.einsum("rpk,k->rp", basis, normalized[index])
        weight = weights * density_gradient(points[..., 0], flux)
        relative = points - centre
        moment_basis = jnp.stack(
            (jnp.ones_like(weight), relative[..., 0], relative[..., 1]), axis=-1
        )
        integral_basis = jnp.sum(
            weight[..., None, None] * basis[..., :, None] * moment_basis[..., None, :],
            axis=(0, 1),
        )
        return normalized_rate[:, index] @ integral_basis

    def whole_cell(index):
        with xla_metadata.set_xla_metadata(tangent_site="interior:whole"):
            return interior(
                index,
                geometry.vertices[index][None],
                geometry.vertex_count[index][None],
            )

    def live_rings(index, rings, counts, velocity=None):
        def evaluate(selected_rings, selected_counts, selected):
            value = interior(index, selected_rings, selected_counts)
            if velocity is not None:
                moved = moving_polygon_current_moment_tangents(
                    selected_rings,
                    velocity[:, selected],
                    selected_counts,
                    geometry.centre[index],
                    geometry.pitch[index],
                    normalized[index],
                    profile,
                )
                moving = jnp.stack(
                    tuple(jnp.sum(component, axis=1) for component in moved), axis=-1
                )
                value = value.at[boundary_channels].add(moving)
            return value

        return _integrate_live_rings(
            rings, counts, evaluate, jnp.zeros((channels, 3), dtype=psi.dtype), width=2
        )

    def conic_cell(index):
        centre = geometry.centre[index]
        vertices = geometry.vertices[index]
        count = geometry.vertex_count[index]

        def evaluate():
            rings, counts, valid = _conic_rings(
                vertices,
                count,
                centre,
                geometry.pitch[index],
                signed[index],
                read.fragment_selected[index],
                quadrature.segments,
            )
            velocity = jax.vmap(
                lambda direction: _conic_ring_tangents(
                    vertices,
                    count,
                    centre,
                    geometry.pitch[index],
                    signed[index],
                    direction,
                    read.fragment_selected[index],
                    quadrature.segments,
                )
            )(signed_rate[boundary_channels, index])
            with xla_metadata.set_xla_metadata(tangent_site="interior:conic"):
                value = live_rings(index, rings, counts, velocity)
            return jnp.where(valid, value, jnp.nan)

        return evaluate()

    def normal_cell(index):
        rings, counts = _normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            read.saddle_form,
            read.fragment_selected[index],
            read.edge_interval[index],
            quadrature.segments,
        )
        with xla_metadata.set_xla_metadata(tangent_site="interior:normal"):
            return live_rings(index, rings, counts)

    full = read.membership >= 1 - 1e-14
    cut = (read.membership > 0) & ~full
    whole = jax.vmap(whole_cell)(jnp.arange(cells))
    whole = jnp.where(full[:, None, None], whole, 0.0)
    if quadrature.cut_batch_width < 1:
        raise ValueError("cut batch width must be positive")
    batch_width = min(quadrature.cut_batch_width, cells)

    normal = cut & read.normal_form_cells
    zero = jnp.zeros_like(whole)
    cuts = _integrate_bucket(cut & ~normal, batch_width, conic_cell, zero)
    cuts += _integrate_bucket(
        normal, min(_NORMAL_CUT_BATCH_WIDTH, cells), normal_cell, zero
    )
    raw = whole + cuts
    total_rate = jnp.sum(raw[:, :, 0], axis=0)
    scale_rate = (
        jnp.zeros_like(total_rate)
        if quadrature.target_current is None
        else -total_rate / booked.raw_current
    )
    moment = jnp.stack(tuple(booked[:3]), axis=-1)
    response = booked.normalization * raw + scale_rate[None, :, None] * moment[:, None]
    return jnp.moveaxis(jnp.where(booked.valid, response, jnp.nan), 1, -1)
