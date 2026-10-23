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

from nova.equilibrium import topology
from nova.equilibrium.clip_quadrature import (
    ClippedCurrentMoments,
    _quadrature_from_arrays,
    clipped_support_current_moments,
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
    """

    sigma: object = 1.0
    target_current: object = None
    segments: int = dataclass_field(default=512, metadata={"static": True})


class Booking(NamedTuple):
    """Current moments and the normalization and qualification ledger."""

    cell_current: jax.Array
    radial_moment: jax.Array
    vertical_moment: jax.Array
    raw_current: jax.Array
    normalization: jax.Array
    valid: jax.Array


def _conic_rings(vertices, count, centre, pitch, coefficient, selected, segments):
    """Trace the same cosine-spaced strips as support.geometry.read_polygons."""
    local = (vertices - centre) / pitch
    fragments = topology.quadratic_cell_fragments(
        local, count, coefficient, selected.shape[0]
    )
    left, right = fragments.slice_breaks[:-1], fragments.slice_breaks[1:]
    parameter = 0.5 * (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, segments + 1)))
    x = left[:, None] + (right - left)[:, None] * parameter
    bottom, top, _ = topology._positive_vertical_intervals(local, count, coefficient, x)
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


def effective_disk(form, geometry, policy_radius, pitch_floor):
    """Return the effective neighbourhood centre and radius about the saddle.

    The radius is ``max(r_nf, pitch_floor * median pitch)``; the centre is the
    selected saddle. Both are runtime policy operands, so the caller supplies
    the configured radius and floor and this owner only composes them.
    """
    radius = jnp.maximum(policy_radius, pitch_floor * jnp.median(geometry.pitch))
    return form.position, radius


def _inside_loop(points, polygon, size):
    """Crossing-number test over the first ``size`` vertices of a padded loop.

    The clip that produces ``polygon`` pads the tail with the origin, so the
    test reads only the live prefix. A ray cast from each point counts the
    edges it crosses; an odd count is the interior. No orientation is assumed,
    so a lobe traced either way is read the same.
    """
    slots = jnp.arange(polygon.shape[0])
    live = slots < size
    following = jnp.where(slots + 1 < size, slots + 1, 0)
    start = jnp.where(live[:, None], polygon, 0.0)
    end = jnp.where(live[:, None], polygon[following], 0.0)
    x, y = points[..., 0][..., None], points[..., 1][..., None]
    x0, y0, x1, y1 = start[:, 0], start[:, 1], end[:, 0], end[:, 1]
    rising = (y0 > y) & (y1 <= y)
    falling = (y1 > y) & (y0 <= y)
    run = y1 - y0
    crossing = x0 + (x1 - x0) * (y - y0) / jnp.where(run != 0, run, 1.0)
    crosses = (rising | falling) & (x < crossing) & live[None, :]
    return jnp.sum(crosses, axis=-1) % 2 == 1


DISK_ARC_SEGMENTS = 32
"""Half-plane count the effective disk is clipped with.

The disk enters the booking as the intersection of this many half-planes
bounding a regular polygon inscribed in the circle, so the boundary between the
normal-form portion inside the disk and the ordinary cut outside it is a set of
explicit support edges that move with the disk centre. Weighting quadrature
nodes by a radial test instead makes the booked integral jump whenever a node
changes side of the boundary, and that jump keeps its size while the step
shrinks, which is why a remainder built on it stops falling with the step.

The inscribed edge lies short of the circle by ``radius * (1 - cos(pi /
segments))``, about 0.48 per cent of the radius at this count. The two pieces
still tile the ring set exactly, because the exterior piece is the whole ring
set with the clipped piece subtracted rather than a second clip.
"""


def _disk_clip_planes(disk_centre, radius, segments):
    """Return the inward normals and offsets bounding the inscribed disk."""
    angle = 2.0 * jnp.pi * (jnp.arange(segments) + 0.5) / segments
    normal = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    return normal, radius * jnp.cos(jnp.pi / segments)


def _disk_clipped_triangles(
    vertices, count, disk_centre, radius, segments=DISK_ARC_SEGMENTS
):
    """Clip every fan triangle of every ring to the disk, as explicit edges.

    Each ring is fanned from its first vertex and each triangle is clipped by
    the ``segments`` half-planes in turn. A clipped triangle is convex and
    carries at most ``3 + segments`` vertices, every one of them a support
    vertex: a ring corner, an intersection of a ring edge with a clip line, or
    one of the endpoints the clip lines themselves contribute. No quadrature
    node is selected or weighted by a radial test anywhere in this route.
    """
    capacity = vertices.shape[1]
    triangle_count = capacity - 2
    slots = jnp.arange(1, capacity - 1)
    first = jnp.broadcast_to(vertices[:, :1], (len(vertices), triangle_count, 2))
    subject = jnp.stack((first, vertices[:, slots], vertices[:, slots + 1]), axis=-2)
    live = slots[None, :] + 1 < count[:, None]
    width = 3 + segments
    order = jnp.arange(width)
    padded = jnp.concatenate(
        (subject, jnp.zeros(subject.shape[:-2] + (segments, 2), subject.dtype)),
        axis=-2,
    )
    live_count = jnp.where(live, 3, 0).astype(jnp.int32)
    normal, bound = _disk_clip_planes(disk_centre, radius, segments)
    for step in range(segments):
        line = normal[step]
        distance = (padded - disk_centre) @ line - bound
        following = (order + 1) % jnp.maximum(live_count[..., None], 1)
        ahead = jnp.take_along_axis(padded, following[..., None], axis=-2)
        next_distance = (ahead - disk_centre) @ line - bound
        live_edge = order < live_count[..., None]
        inside = distance <= 0.0
        keep = live_edge & inside
        entering = live_edge & (inside != (next_distance <= 0.0))
        gap = distance - next_distance
        fraction = distance / jnp.where(gap != 0.0, gap, 1.0)
        meeting = padded + fraction[..., None] * (ahead - padded)
        candidates = jnp.stack((keep, entering), axis=-1).reshape(
            keep.shape[:-1] + (2 * width,)
        )
        values = jnp.stack((padded, meeting), axis=-2).reshape(
            padded.shape[:-2] + (2 * width, 2)
        )
        position = jnp.cumsum(candidates, axis=-1) - candidates
        key = jnp.where(candidates, position, 2 * width)
        chosen = jnp.argsort(key, axis=-1)[..., :width]
        padded = jnp.take_along_axis(values, chosen[..., None], axis=-2)
        live_count = jnp.sum(candidates, axis=-1).astype(jnp.int32)
    return padded, live_count


def _disk_split_moments(
    rings,
    counts,
    centre,
    pitch,
    coefficient,
    axis_flux,
    boundary_flux,
    profile,
    disk_centre,
    radius,
    *,
    keep_inside,
    force_whole=False,
    confined=None,
):
    """Signed fan moments restricted to the disk interior or exterior.

    A cut cell books one binary support, ``(C ∩ D ∩ N) ∪ (C ∩ Dᶜ ∩ Q)``. The
    disk split is taken on the geometry and not on the quadrature: the fan
    triangles of one ring set are clipped by the disk boundary itself and the
    clipped pieces are integrated through the shared clip-quadrature owner. The
    exterior piece is the whole ring set with the clipped piece subtracted, so
    the two tile the ring set exactly whatever the disk does, and both are
    continuous in ``disk_centre`` with a derivative that carries the boundary's
    motion.

    ``confined`` carries ``(polygon, size, live)`` for the closed normal-form
    component the read admits, already clipped to this cell. When it is given
    and live, both halves of the split are additionally restricted to it, so a
    booking cell books only current inside the confined component and never the
    whole ordinary cut. With no confined component the split is used.

    ``force_whole`` drops the disk restriction and integrates the whole ring
    set. It exists only so the continuity check can apply its ``--force-conic
    -right`` mutation -- the right-hand (exterior) support replaced by the whole
    ordinary cut at the same density and weight -- and require that mutation to
    fail the same assertion.
    """
    live = counts >= 3
    centroids = jnp.broadcast_to(centre, (rings.shape[0], 2))
    polygon = None
    if confined is not None:
        polygon, size, confined_live = confined
    else:
        size, confined_live = 0, False

    def book(points, weights):
        mask = jnp.ones(points.shape[:-1], dtype=bool)
        if polygon is not None:
            mask = jnp.where(confined_live, _inside_loop(points, polygon, size), mask)
        local = (points - centre) / pitch
        x, y = local[..., 0], local[..., 1]
        flux = (
            coefficient[0]
            + coefficient[1] * x
            + coefficient[2] * y
            + coefficient[3] * x * x
            + coefficient[4] * x * y
            + coefficient[5] * y * y
        )
        span = boundary_flux - axis_flux
        normalized = (flux - axis_flux) / jnp.where(span != 0, span, 1.0)
        density = profile.current_density(points[..., 0], normalized)
        weight = jnp.where(mask, weights, 0.0)
        displacement = points - centre
        return jnp.stack(
            (
                jnp.sum(weight * density),
                jnp.sum(weight * density * displacement[..., 0]),
                jnp.sum(weight * density * displacement[..., 1]),
            )
        )

    whole_points, whole_weights = _quadrature_from_arrays(
        rings, counts, centroids, live, signed=True
    )
    whole = book(whole_points, whole_weights)
    if force_whole:
        return whole
    clipped, clipped_live = _disk_clipped_triangles(rings, counts, disk_centre, radius)
    flat = clipped.reshape(-1, clipped.shape[-2], 2)
    fan_centre = jnp.repeat(centroids, clipped.shape[1], axis=0)
    inside_points, inside_weights = _quadrature_from_arrays(
        flat,
        clipped_live.reshape(-1),
        fan_centre,
        clipped_live.reshape(-1) > 0,
        signed=True,
    )
    inside = book(inside_points, inside_weights)
    if keep_inside:
        return inside
    return whole - inside


def composite_cut_moments(
    vertices,
    count,
    centre,
    pitch,
    form,
    flux_coefficient,
    conic_coefficient,
    profile,
    selected,
    intervals,
    conic_selected,
    axis_flux,
    boundary_flux,
    disk_centre,
    radius,
    segments,
    force_conic_right=False,
    confined=None,
):
    """Book one cut cell over a single support split at the disk boundary.

    Inside the disk the support is the selected normal-form sector; outside it
    the support is the ordinary quadratic cut. Both portions are integrated on
    the shared clip-quadrature nodes with the same density, and their moments
    add because their interiors are disjoint. ``flux_coefficient`` carries the
    physical local flux used to evaluate the density; ``conic_coefficient``
    carries the same flux in the signed, level-subtracted form the ordinary cut
    traces its positive region with.

    ``confined`` carries the read's own closed normal-form component as
    ``(loop, live)``. It is clipped to this cell once and handed to both halves,
    so the booking follows the connectivity the read already publishes rather
    than the whole ordinary cut; a read with no admitted closed component books
    the split uninterruptedly.
    """
    if confined is None:
        split_confined = None
    else:
        loop, live = confined
        polygon, size = topology.clip_normal_form_loop(loop, vertices, count)
        split_confined = (polygon, size, live)
    normal_rings, normal_counts = _normal_rings(
        vertices, count, centre, form, selected, intervals, segments
    )
    conic_rings, conic_counts, conic_valid = _conic_rings(
        vertices, count, centre, pitch, conic_coefficient, conic_selected, segments
    )
    inside = _disk_split_moments(
        normal_rings,
        normal_counts,
        centre,
        pitch,
        flux_coefficient,
        axis_flux,
        boundary_flux,
        profile,
        disk_centre,
        radius,
        keep_inside=True,
        confined=split_confined,
    )
    outside = _disk_split_moments(
        conic_rings,
        conic_counts,
        centre,
        pitch,
        flux_coefficient,
        axis_flux,
        boundary_flux,
        profile,
        disk_centre,
        radius,
        keep_inside=False,
        force_whole=force_conic_right,
    )
    return inside + outside, conic_valid


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

    def integrate_live_rings(index, rings, counts):
        live = counts >= 3
        live_count = jnp.sum(live, dtype=jnp.int32)
        ring_index = jnp.nonzero(live, size=len(counts), fill_value=0)[0]
        zero = ClippedCurrentMoments(*(jnp.zeros((), rings.dtype) for _ in range(3)))

        def add_ring(slot, total):
            ring = ring_index[slot]
            value = integrate(index, rings[ring][None], counts[ring][None])
            return jax.tree.map(lambda prior, addition: prior + addition, total, value)

        return jax.lax.fori_loop(0, live_count, add_ring, zero)

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
            moments = integrate_live_rings(index, rings, counts)
            return jax.tree.map(lambda value: jnp.where(valid, value, jnp.nan), moments)

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
            return integrate_live_rings(index, rings, counts)

        return jax.lax.cond(read.normal_form_cells[index], normal, conic, None)

    cells = len(geometry.centre)
    full = read.membership >= 1 - 1e-14
    cut = (read.membership > 0) & ~full
    whole = jax.vmap(whole_cell)(jnp.arange(cells))
    whole = jax.tree.map(lambda value: jnp.where(full, value, 0.0), whole)

    # Only the compact cut batch carries sampled rings. The number of batches
    # varies with the read, while the compiled batch width bounds device memory.
    batch_width = 32
    cut_count = jnp.sum(cut, dtype=jnp.int32)
    cut_index = jnp.nonzero(cut, size=cells, fill_value=0)[0]
    cut_index = jnp.pad(cut_index, (0, (-cells) % batch_width))

    def add_batch(batch, carried):
        start = batch * batch_width
        indices = jax.lax.dynamic_slice(cut_index, (start,), (batch_width,))
        active = start + jnp.arange(batch_width) < cut_count
        values = jax.vmap(cut_cell)(indices)

        def scatter(total, value):
            return total.at[indices].add(jnp.where(active, value, 0.0))

        return jax.tree.map(scatter, carried, values)

    cuts = jax.tree.map(jnp.zeros_like, whole)
    cuts = jax.lax.fori_loop(
        0, (cut_count + batch_width - 1) // batch_width, add_batch, cuts
    )
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
