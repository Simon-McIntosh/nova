"""Blend booked saddle moments where two continuous cuts disagree in area."""

import jax.numpy as jnp

from nova.equilibrium import fixed_points
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_map_production.kernel import (
    bell_fraction_with_crossings,
    read,
)
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.lean_map.quadrature import interior_moments


def booked_blend(area_gap):
    """Damp the ordinary-cut correction outside a narrow agreement region."""
    disagreement = jnp.square(area_gap / 0.043)
    return disagreement * jnp.exp(-disagreement) / 3.0


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
    saddle, saddle_valid = program._saddle_moments(geometry, reading, profile)

    ordinary_fraction, fraction_valid, _ = bell_fraction_with_crossings(
        geometry, reading, convention.sigma
    )
    area_gap = reading.geometric_membership - ordinary_fraction
    blend = booked_blend(area_gap)
    blended_saddle = saddle + jnp.where(
        saddle_cut[:, None], blend[:, None] * (ordinary - saddle), 0.0
    )
    raw = (
        interior + jnp.where(ordinary_cut[:, None], ordinary, 0.0) + blended_saddle
    ) * reading.path_weight[:, None]
    total = jnp.sum(raw[:, 0])
    missing = jnp.any(cut & ~complete)
    valid = (
        jnp.all(roots.valid)
        & reading.valid
        & reading.qualified
        & jnp.all(~ordinary_cut | ordinary_valid)
        & jnp.all(~saddle_cut | (ordinary_valid & fraction_valid))
        & saddle_valid
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
