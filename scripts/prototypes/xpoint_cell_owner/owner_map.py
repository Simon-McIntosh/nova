"""Book a single Bell-conic cut for saddle-neighbourhood area and moments."""

from functools import partial

import jax
import jax.numpy as jnp

from nova.equilibrium import current, fixed_points
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_map_production.kernel import read
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.lean_map.quadrature import interior_moments
from scripts.prototypes.xpoint_cell_owner.device_owner import integrate_cell
from scripts.prototypes.xpoint_cell_owner.lobe_booking import lobe_moments


def saddle_conic_coefficients(centres, pitches, saddle, hessian, level):
    """Express the quadratic held at the located saddle in the cell convention.

    The conic is ``0.5 * (x - saddle)^T H (x - saddle)`` with ``H`` the exact
    flux Hessian at the saddle, so its value is the boundary level and its
    gradient vanishes there. The result is returned in the same per-pitch
    coefficient form the fitted Bell quadratic uses, so one integral consumes
    both representations.
    """
    local = (saddle - centres) / pitches[:, None]
    scaled = hessian * pitches[:, None, None] ** 2
    linear = jnp.einsum("nij,nj->ni", scaled, local)
    quadratic = 0.5 * jnp.einsum("ni,nij,nj->n", local, scaled, local)
    return jnp.stack(
        (
            level + quadratic,
            -linear[:, 0],
            -linear[:, 1],
            0.5 * scaled[:, 0, 0],
            scaled[:, 0, 1],
            0.5 * scaled[:, 1, 1],
        ),
        axis=1,
    )


def fitted_hessian(coefficients):
    """Hessian of the cell's fitted quadratic, in the fitted coefficient units."""
    return jnp.stack(
        (
            jnp.stack((2 * coefficients[:, 3], coefficients[:, 4]), axis=1),
            jnp.stack((coefficients[:, 4], 2 * coefficients[:, 5]), axis=1),
        ),
        axis=1,
    )


def fitted_value(coefficients, local):
    """Evaluate a fitted cell quadratic at local pitch-unit coordinates."""
    x, y = local[..., 0], local[..., 1]
    return (
        coefficients[..., 0]
        + coefficients[..., 1] * x
        + coefficients[..., 2] * y
        + coefficients[..., 3] * x * x
        + coefficients[..., 4] * x * y
        + coefficients[..., 5] * y * y
    )


def reference_saddle_moments(geometry, reading, profile, capacity=16):
    """Book owner sectors on the inherited signed ring fans.

    Reproduces the accessor's own saddle booking, so a carrier built through
    this path is what the read books rather than an approximation of it. It
    exists to re-measure the error the diverted map guards were set from.
    """
    active = reading.booking_cells & (reading.membership > 0)
    count = jnp.sum(active, dtype=jnp.int32)
    indices = jnp.nonzero(active, size=capacity, fill_value=0)[0]

    def one(index):
        rings, sizes = current._normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            reading.saddle_form,
            reading.fragment_selected[index],
            reading.edge_interval[index],
            4,
        )
        origin = geometry.centre[index]
        slot = jnp.arange(rings.shape[1])
        following = jnp.where(slot[None] + 1 < sizes[:, None], slot[None] + 1, 0)
        end = jnp.take_along_axis(rings, following[..., None], axis=1)
        first = rings - origin
        second = end - origin
        area = 0.5 * (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
        area = jnp.where(slot[None] < sizes[:, None], area, 0.0)
        barycentric = jnp.asarray(((2 / 3, 1 / 6), (1 / 6, 2 / 3), (1 / 6, 1 / 6)))
        points = (
            origin
            + barycentric[:, 0, None] * first[..., None, :]
            + barycentric[:, 1, None] * second[..., None, :]
        )
        local = (points - origin) / geometry.pitch[index]
        x, y = local[..., 0], local[..., 1]
        coefficient = reading.field_coefficients[index]
        flux = (
            coefficient[0]
            + coefficient[1] * x
            + coefficient[2] * y
            + coefficient[3] * x * x
            + coefficient[4] * x * y
            + coefficient[5] * y * y
        )
        normalized = (flux - reading.axis_flux) / (
            reading.boundary_flux - reading.axis_flux
        )
        density = profile.current_density(points[..., 0], normalized)
        weighted = area[..., None] * density / 3
        displacement = points - origin
        return jnp.stack(
            (
                jnp.sum(weighted),
                jnp.sum(weighted * displacement[..., 0]),
                jnp.sum(weighted * displacement[..., 1]),
            )
        )

    packed = jax.vmap(one)(indices)
    live = jnp.arange(capacity) < count
    scattered = (
        jnp.zeros((len(geometry.centre), 3), dtype=packed.dtype)
        .at[indices]
        .add(jnp.where(live[:, None], packed, 0.0))
    )
    return scattered, count <= capacity


def owned_cut(
    reading,
    geometry,
    profile,
    sigma,
    capacity=16,
    component="halfplane",
    saddle_hessian=None,
    density_mode="fitted",
):
    """Replace the sampled saddle-sector area and moment booking together."""
    if component == "inherited":
        moments, valid = reference_saddle_moments(geometry, reading, profile, capacity)
        return reading, moments, valid
    active = reading.normal_form_cells
    if component == "ordinary":
        active = jnp.zeros_like(active)
    count = jnp.sum(active, dtype=jnp.int32)
    indices = jnp.nonzero(active, size=capacity, fill_value=0)[0]
    density = program._density_coefficients(geometry, reading, profile)
    if component == "lobe":
        selected = reading.booking_cells & (reading.membership > 0)
        count = jnp.sum(selected, dtype=jnp.int32)
        indices = jnp.nonzero(selected, size=capacity, fill_value=0)[0]
        integrals, integral_valid = lobe_moments(
            geometry,
            reading,
            profile,
            indices,
            density,
            reading.boundary_flux - reading.axis_flux,
            sigma,
            density_mode=density_mode,
        )
        live = jnp.arange(capacity) < count
        packed = jnp.zeros((len(geometry.centre), 4), dtype=integrals.dtype)
        packed = packed.at[indices].add(jnp.where(live[:, None], integrals, 0.0))
        return reading, packed[:, 1:], (count <= capacity) & jnp.all(
            ~live | integral_valid
        )
    coefficients = reading.field_coefficients[indices]
    branch = "bell" if component == "bell" else "halfplane"
    orientation = None
    if component == "oriented":
        orientation = jnp.broadcast_to(sigma * saddle_hessian, (indices.shape[0], 2, 2))
    if component == "anchored":
        local = (
            reading.saddle_form.position - geometry.centre[indices]
        ) / geometry.pitch[indices, None]
        offset = reading.boundary_flux - fitted_value(
            reading.field_coefficients[indices], local
        )
        coefficients = reading.field_coefficients[indices].at[:, 0].add(offset)
        branch = "halfplane"
    elif component == "shifted":
        local = (
            reading.saddle_form.position - geometry.centre[indices]
        ) / geometry.pitch[indices, None]
        coefficients = saddle_conic_coefficients(
            geometry.centre[indices],
            geometry.pitch[indices],
            reading.saddle_form.position,
            saddle_hessian,
            fitted_value(reading.field_coefficients[indices], local),
        )
        branch = "halfplane"
    elif component in ("saddle", "pinned"):
        curvature = (
            saddle_hessian
            if component == "saddle"
            else fitted_hessian(reading.field_coefficients[indices])
        )
        coefficients = saddle_conic_coefficients(
            geometry.centre[indices],
            geometry.pitch[indices],
            reading.saddle_form.position,
            curvature,
            reading.boundary_flux,
        )
        branch = "halfplane"
    cut_axes = (0, 0, 0, 0, 0, None, None, None, None, 0)
    orientation_axes = None if orientation is None else 0
    span = reading.boundary_flux - reading.axis_flux
    rule = partial(
        integrate_cell,
        axis_component=branch,
        density_mode=density_mode,
        profile=profile,
        span=span,
    )
    selected = jax.vmap(rule, in_axes=cut_axes + (orientation_axes,))(
        geometry.vertices[indices],
        geometry.vertex_count[indices],
        geometry.centre[indices],
        geometry.pitch[indices],
        coefficients,
        reading.boundary_flux,
        sigma,
        reading.saddle_form.position,
        reading.axis,
        density[indices],
        orientation,
    )
    integrals, valid = selected
    live = jnp.arange(capacity) < count
    packed = jnp.zeros((len(geometry.centre), 4), dtype=integrals.dtype)
    packed = packed.at[indices].add(jnp.where(live[:, None], integrals, 0.0))
    fraction = jnp.clip(packed[:, 0] / geometry.full_area, 0.0, 1.0)
    reading = reading._replace(
        geometric_membership=jnp.where(active, fraction, reading.geometric_membership),
        membership=jnp.where(
            active, fraction * reading.path_weight, reading.membership
        ),
        booking_cells=active,
    )
    return reading, packed[:, 1:], (count <= capacity) & jnp.all(~live | valid)


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
    frame_gate=True,
    component="halfplane",
    density_mode="fitted",
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
    saddle_gap = jnp.linalg.norm(
        roots.position - reading.saddle_form.position[None], axis=-1
    )
    saddle_root = jnp.argmin(jnp.where(roots.valid, saddle_gap, jnp.inf))
    reading, owned, owner_valid = owned_cut(
        reading,
        geometry,
        profile,
        convention.sigma,
        component=component,
        saddle_hessian=roots.jet.hessian[saddle_root],
        density_mode=density_mode,
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
    total = jnp.sum(raw[:, 0])
    missing = jnp.any(cut & ~complete)
    valid = (
        jnp.all(roots.valid)
        & reading.valid
        & reading.qualified
        & jnp.all(~ordinary_cut | ordinary_valid)
        & owner_valid
        & ~missing
        & jnp.isfinite(total)
        & (total != 0)
    )
    normalized = raw * (target_current / jnp.where(total != 0, total, 1))
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
