"""Book a saddle-constrained Bell-conic cut for area and moments."""

import jax
import jax.numpy as jnp

from nova.equilibrium import fixed_points
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_map_production.kernel import read
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.lean_map.quadrature import interior_moments
from scripts.prototypes.xpoint_cell_owner.device_owner import integrate_cell
from scripts.prototypes.xpoint_cell_owner.device_projection import (
    saddle_constrained_coefficients,
)


def owned_cut(reading, geometry, profile, sigma, capacity=16):
    """Replace the sampled saddle-sector area and moment booking together."""
    active = reading.normal_form_cells
    count = jnp.sum(active, dtype=jnp.int32)
    indices = jnp.nonzero(active, size=capacity, fill_value=0)[0]
    density = program._density_coefficients(geometry, reading, profile)
    projected = saddle_constrained_coefficients(
        reading.field_coefficients[indices],
        geometry.centre[indices],
        geometry.pitch[indices],
        reading.saddle_form.position,
        reading.boundary_flux,
    )
    selected = jax.vmap(
        integrate_cell, in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0)
    )(
        geometry.vertices[indices],
        geometry.vertex_count[indices],
        geometry.centre[indices],
        geometry.pitch[indices],
        projected,
        reading.boundary_flux,
        sigma,
        reading.saddle_form.position,
        reading.axis,
        density[indices],
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
    reading, owned, owner_valid = owned_cut(
        reading, geometry, profile, convention.sigma
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
