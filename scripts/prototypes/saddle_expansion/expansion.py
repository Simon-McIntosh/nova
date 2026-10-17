"""Book X-point cells from an expansion anchored at the physical saddle.

The field near the separatrix saddle is represented by the exact Biot-Savart
value, gradient and Hessian at the located saddle, not by a per-cell quadratic
fit. One conic therefore anchors every cell in the owned disk, and the same
slice integral returns both the area membership and the booked moments.
"""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_map_production.kernel import read
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.lean_map.quadrature import interior_moments
from scripts.prototypes.read_route.jets import _source_jets
from scripts.prototypes.xpoint_cell_owner.conic_integral import _quadratic


def _pack(jet):
    """Six-jet packing: value, gradient, then the three Hessian entries."""
    return jnp.stack(
        (
            jet.value,
            jet.gradient[0],
            jet.gradient[1],
            jet.hessian[0, 0],
            jet.hessian[0, 1],
            jet.hessian[1, 1],
        )
    )


def exact_jets(total, coupling, reference, point, moments):
    """Exact Biot-Savart value, gradient and Hessian at an arbitrary point."""
    base = _pack(total.evaluate(point))
    source = _source_jets(point, coupling)
    delta = moments - reference
    return base + jnp.einsum("cm,cmq->q", delta, source)


def _saddle_frame(jet6, boundary, sigma):
    """Order-two expansion about the saddle in the sigma-scaled Hessian basis."""
    value = (jet6[0] - boundary) * sigma
    gradient = jet6[1:3] * sigma
    hessian = sigma * jnp.array(((jet6[3], jet6[4]), (jet6[4], jet6[5])))
    eigenvalue, eigenvector = jnp.linalg.eigh(hessian)
    plus = eigenvector[:, 1]
    minus = eigenvector[:, 0]
    return value, gradient, eigenvalue, plus, minus


def _roots(a, b, c):
    """Return the two real roots of a v-quadratic, absent roots padded to inf."""
    radical = jnp.sqrt(jnp.maximum(b * b - 4 * a * c, 0.0))
    safe_a = jnp.where(jnp.abs(a) > 1e-14, a, 1.0)
    safe_b = jnp.where(jnp.abs(b) > 1e-14, b, 1.0)
    quadratic = jnp.stack(
        ((-b - radical) / (2 * safe_a), (-b + radical) / (2 * safe_a))
    )
    linear = -c / safe_b
    roots = jnp.where(jnp.abs(a) > 1e-14, quadratic, linear)
    valid = jnp.where(jnp.abs(a) > 1e-14, b * b - 4 * a * c >= 0, jnp.abs(b) > 1e-14)
    return roots, valid


def integrate_disk_cell(
    vertices,
    count,
    centre,
    pitch,
    expansion_coefficients,
    origin,
    plus,
    minus,
    radius,
    density_coefficients,
    *,
    inner_nodes,
    outer_nodes,
    outer_weights,
):
    """Integrate the expansion's positive region over cell cap the disk.

    ``expansion_coefficients`` are the saddle-anchored conic in this cell's
    pitch-normalised frame, so the integral is shared by membership and
    booking. ``radius`` is the owned disk radius in pitch units and caps the
    slice in both directions, so a cell the disk cuts is split exactly.
    """
    polygon = (vertices - origin) / pitch
    polygon = jnp.stack((polygon @ plus, polygon @ minus), axis=1)
    slot = jnp.arange(vertices.shape[0])
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    end = polygon[following]
    live = slot < count

    def value(u, v):
        return _quadratic(expansion_coefficients, jnp.asarray(u), jnp.asarray(v))

    constant = value(0.0, 0.0)
    plus_value, minus_value = value(1.0, 0.0), value(-1.0, 0.0)
    upper, lower = value(0.0, 1.0), value(0.0, -1.0)
    linear_u = (plus_value - minus_value) / 2
    linear_v = (upper - lower) / 2
    squared_u = (plus_value + minus_value) / 2 - constant
    squared_v = (upper + lower) / 2 - constant
    mixed = value(1.0, 1.0) - plus_value - upper + constant

    vertex_u = jnp.where(live, polygon[:, 0], jnp.inf)
    lower_u = jnp.maximum(0.0, jnp.min(vertex_u))
    upper_u = jnp.max(jnp.where(live, polygon[:, 0], -jnp.inf))
    upper_u = jnp.maximum(lower_u, upper_u)
    # The disk caps the slice in u; the half-plane face defines the inner side.
    upper_u = jnp.minimum(upper_u, radius)
    upper_u = jnp.maximum(lower_u, upper_u)

    start_value = value(polygon[:, 0], polygon[:, 1])
    end_value = value(end[:, 0], end[:, 1])
    middle = (polygon + end) / 2
    middle_value = value(middle[:, 0], middle[:, 1])
    edge_squared = 2 * (start_value + end_value - 2 * middle_value)
    edge_linear = end_value - start_value - edge_squared
    edge_roots, edge_valid = _roots(edge_squared, edge_linear, start_value)
    edge_u = polygon[:, 0, None] + edge_roots.T * (end[:, 0] - polygon[:, 0])[:, None]
    edge_live = (
        live[:, None] & edge_valid[:, None] & (edge_roots.T > 0) & (edge_roots.T < 1)
    )
    discriminant_squared = mixed * mixed - 4 * squared_v * squared_u
    discriminant_linear = 2 * linear_v * mixed - 4 * squared_v * linear_u
    discriminant_constant = linear_v * linear_v - 4 * squared_v * constant
    tangent_roots, tangent_valid = _roots(
        discriminant_squared, discriminant_linear, discriminant_constant
    )
    candidates = jnp.concatenate(
        (
            jnp.array((lower_u, upper_u)),
            jnp.where(live, polygon[:, 0], upper_u),
            jnp.where(edge_live, edge_u, upper_u).reshape(-1),
            jnp.where(tangent_valid, tangent_roots, upper_u),
        )
    )
    candidates = (
        jnp.where((candidates > lower_u) & (candidates < upper_u), candidates, upper_u)
        .at[0]
        .set(lower_u)
    )
    bounds = jnp.sort(candidates)
    left, right = bounds[:-1], bounds[1:]
    theta = (outer_nodes + 1) * (jnp.pi / 4)
    u = left[:, None] + (right - left)[:, None] * jnp.sin(theta) ** 2
    outer = (right - left)[:, None] * jnp.sin(2 * theta) * (jnp.pi / 4) * outer_weights

    def one_slice(u_value):
        span = end[:, 0] - polygon[:, 0]
        safe_span = jnp.where(jnp.abs(span) > 1e-14, span, 1.0)
        edge_fraction = (u_value - polygon[:, 0]) / safe_span
        cross = (
            live & (jnp.abs(span) > 1e-14) & (edge_fraction >= 0) & (edge_fraction < 1)
        )
        crossing_v = polygon[:, 1] + edge_fraction * (end[:, 1] - polygon[:, 1])
        low = jnp.min(jnp.where(cross, crossing_v, jnp.inf))
        high = jnp.max(jnp.where(cross, crossing_v, -jnp.inf))
        valid_span = jnp.isfinite(low) & jnp.isfinite(high)
        low, high = jnp.where(valid_span, low, 0.0), jnp.where(valid_span, high, 0.0)
        half = jnp.sqrt(jnp.maximum(radius * radius - u_value * u_value, 0.0))
        low, high = jnp.maximum(low, -half), jnp.minimum(high, half)
        v_roots, root_valid = _roots(
            squared_v,
            linear_v + mixed * u_value,
            constant + linear_u * u_value + squared_u * u_value * u_value,
        )
        v_roots = jnp.where(root_valid, v_roots, high)
        v_roots = jnp.clip(v_roots, low, high)
        cuts = jnp.sort(jnp.concatenate((jnp.array((low, high)), v_roots)))
        v_left, v_right = cuts[:-1], cuts[1:]
        middle_v = (v_left + v_right) / 2
        positive = value(u_value, middle_v) > 0
        v_nodes = middle_v[:, None] + (v_right - v_left)[:, None] * inner_nodes / 2
        local = (origin - centre) / pitch + u_value * plus + v_nodes[..., None] * minus
        x, y = local[..., 0], local[..., 1]
        density = _quadratic(density_coefficients, x, y)
        terms = jnp.stack(
            (jnp.ones_like(density), density, pitch * density * x, pitch * density * y),
            axis=-1,
        )
        weights = jnp.where(positive, (v_right - v_left) / 2, 0.0)
        return jnp.sum(terms * weights[:, None, None], axis=(0, 1))

    values = jax.vmap(one_slice)(u.reshape(-1)).reshape(*u.shape, 4)
    moments = pitch * pitch * jnp.sum(values * outer[..., None], axis=(0, 1))
    valid = jnp.all(jnp.isfinite(moments))
    return moments, valid


def _polygon_distance(vertices, count, point):
    """Shortest distance from a point to a convex polygon, zero when inside."""
    slot = jnp.arange(vertices.shape[0])
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    end = vertices[following]
    edge = end - vertices
    length = jnp.sum(edge * edge, axis=-1)
    fraction = jnp.sum((point - vertices) * edge, axis=-1) / jnp.where(
        length > 0.0, length, 1.0
    )
    fraction = jnp.clip(fraction, 0.0, 1.0)
    closest = vertices + fraction[:, None] * edge
    return jnp.min(
        jnp.where(slot < count, jnp.linalg.norm(point - closest, axis=-1), jnp.inf)
    )


def _disk_view(
    reading, geometry, coupling, reference, total, moments, sigma, radius, candidates
):
    """Saddle frame and per-cell expansion conics for the cells the disk cuts."""
    saddle = reading.saddle_form.position
    jet6 = exact_jets(total, coupling, reference, saddle, moments)
    value, gradient, eigenvalue, plus, minus = _saddle_frame(
        jet6, reading.boundary_flux, sigma
    )
    distance = jax.vmap(_polygon_distance, in_axes=(0, 0, None))(
        geometry.vertices[candidates], geometry.vertex_count[candidates], saddle
    )
    live = distance <= radius
    capacity = candidates.shape[0]
    index = candidates
    pitch = geometry.pitch[index]
    sigma_expansion = jnp.stack(
        (
            jnp.full(capacity, value),
            pitch * (gradient @ plus),
            pitch * (gradient @ minus),
            0.5 * eigenvalue[1] * pitch * pitch,
            jnp.zeros(capacity),
            0.5 * eigenvalue[0] * pitch * pitch,
        ),
        axis=-1,
    )
    return saddle, plus, minus, radius / pitch, index, live, sigma_expansion


_INNER_NODES = jnp.asarray((-1 / jnp.sqrt(3.0), 1 / jnp.sqrt(3.0)))


def _leggauss(degree):
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(degree)
    return jnp.asarray(nodes), jnp.asarray(weights)


def disk_candidates(geometry, saddle, radius, margin=None):
    """Cells a disk of this radius can cut about a located saddle.

    The owned set depends on the state only through the located saddle, which
    moves by well under a cell pitch between neighbouring states, so the list
    is built once per carrier about a reference saddle and grown by one median
    pitch to cover that motion. The margin is taken in pitches rather than in
    absolute length: a fixed length would admit a larger and larger share of
    the machine as the grid refines, while a pitch-sized margin keeps the
    candidate count fixed along the resolution ladder. It replaces the
    per-state scan over every cell in the machine with a fixed-shape scan over
    the local patch.
    """
    if margin is None:
        margin = jnp.median(geometry.pitch)
    distance = jnp.asarray(
        jax.vmap(_polygon_distance, in_axes=(0, 0, None))(
            geometry.vertices, geometry.vertex_count, saddle
        )
    )
    inside = np.asarray(jax.device_get(distance <= radius + margin))
    return jnp.nonzero(jnp.asarray(inside), size=int(inside.sum()))[0]


def owned_disk_cut(
    reading,
    geometry,
    profile,
    coupling,
    reference,
    total,
    moments,
    sigma,
    radius,
    candidates,
    outer_nodes,
    outer_weights,
    inner_nodes=_INNER_NODES,
):
    """Replace the ordinary saddle-sector booking with the disk expansion."""
    saddle, plus, minus, radius_hat, index, live, expansion = _disk_view(
        reading,
        geometry,
        coupling,
        reference,
        total,
        moments,
        sigma,
        radius,
        candidates,
    )
    density = program._density_coefficients(geometry, reading, profile)
    active = jnp.zeros(len(geometry.centre), dtype=bool).at[index].set(live)

    def integrate(
        vertices, count, centre, pitch, coefficients, density_row, radius_row
    ):
        return integrate_disk_cell(
            vertices,
            count,
            centre,
            pitch,
            coefficients,
            saddle,
            plus,
            minus,
            radius_row,
            density_row,
            inner_nodes=inner_nodes,
            outer_nodes=outer_nodes,
            outer_weights=outer_weights,
        )

    integrals, valid = jax.vmap(integrate)(
        geometry.vertices[index],
        geometry.vertex_count[index],
        geometry.centre[index],
        geometry.pitch[index],
        expansion,
        density[index],
        radius_hat,
    )
    packed = jnp.zeros((len(geometry.centre), 4), dtype=integrals.dtype)
    packed = packed.at[index].add(jnp.where(live[:, None], integrals, 0.0))
    fraction = jnp.clip(packed[:, 0] / geometry.full_area, 0.0, 1.0)
    reading = reading._replace(
        geometric_membership=jnp.where(active, fraction, reading.geometric_membership),
        membership=jnp.where(
            active, fraction * reading.path_weight, reading.membership
        ),
        booking_cells=active,
    )
    return (
        reading,
        packed[:, 1:],
        (jnp.sum(live) <= candidates.shape[0]) & jnp.all(~live | valid),
    )


def map_moments(
    moments,
    values,
    jets,
    window,
    rule,
    fit_inverse,
    sample_design,
    geometry,
    profile,
    seeds,
    convention,
    policy,
    target_current,
    axis_sign,
    *,
    coupling,
    reference,
    total,
    radius,
    candidates,
    outer_nodes,
    outer_weights,
    inner_nodes=_INNER_NODES,
    frame_gate=True,
):
    """Replace fast_map_production.program.map_moments at the booked handoff."""
    cached = fixed_points._CachedField(
        window.mesh, window.lookup, window.rows, moments, window.exterior
    )
    roots = fixed_points.solve(
        window, moments, seeds, seeds, jnp.median(geometry.pitch), axis_sign
    )
    gauss_flux = values.apply(moments).reshape(rule.points.shape[:-1])
    gauss_coefficient = jnp.einsum("nkq,nq->nk", fit_inverse, gauss_flux)
    slots = window.lookup[jets.sample_index]
    complete = jnp.all(slots >= 0, axis=1)
    safe_slots = jnp.maximum(slots, 0)
    band_values = window.exterior[safe_slots, 0] + jnp.einsum(
        "nscm,cm->ns", window.rows[safe_slots, ..., 0], moments
    )
    band_coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
    coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
    field = program.FusedField(cached, coefficient)
    reading = read(field, roots, geometry, convention, policy, frame_gate=frame_gate)
    reading, owned, owner_valid = owned_disk_cut(
        reading,
        geometry,
        profile,
        coupling,
        reference,
        total,
        moments,
        convention.sigma,
        radius,
        candidates,
        outer_nodes,
        outer_weights,
        inner_nodes,
    )
    selected = reading.selected_cells
    full = (
        (reading.geometric_membership >= 1 - 1e-14)
        & (geometry.vertex_count == 6)
        & ~reading.booking_cells
    )
    interior = interior_moments(
        rule, gauss_flux, profile, reading.axis_flux, reading.boundary_flux, full
    )
    ordinary, ordinary_valid, _ = bell_moments(
        geometry, reading, profile, convention.sigma
    )
    cut = (reading.membership > 0) & ~full
    saddle_cut = cut & reading.booking_cells
    ordinary_cut = cut & ~saddle_cut
    raw = (
        interior
        + jnp.where(ordinary_cut[:, None], ordinary, 0.0)
        + jnp.where(saddle_cut[:, None], owned, 0.0)
    ) * reading.path_weight[:, None]
    total_current = jnp.sum(raw[:, 0])
    missing = jnp.any(cut & ~complete)
    valid = (
        jnp.all(roots.valid)
        & reading.valid
        & reading.qualified
        & jnp.all(~ordinary_cut | ordinary_valid)
        & owner_valid
        & ~missing
        & jnp.isfinite(total_current)
        & (total_current != 0)
    )
    normalized = raw * (
        target_current / jnp.where(total_current != 0, total_current, 1)
    )
    normalized = jnp.where(valid, normalized, jnp.nan)
    constraint_scale = jnp.maximum(jnp.abs(target_current), 1.0)
    current_row = (jnp.sum(normalized[:, 0]) - target_current) / constraint_scale
    return program.MapState(
        normalized,
        moments - normalized,
        jnp.atleast_1d(current_row),
        roots.position,
        reading.boundary_class,
        reading.boundary_flux,
        reading.valid & reading.qualified,
        valid,
        jnp.sum(selected),
        jnp.sum(saddle_cut),
    )
