"""Vectorised fixed-shape topology read for the coarse moment map."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium import topology
from scripts.prototypes.sub_millisecond.measure_saddle_frame import connected_cells
from scripts.prototypes.fast_read.ordinary_cut import bell_fraction


class FastRead(NamedTuple):
    axis: jax.Array
    axis_flux: jax.Array
    boundary: jax.Array
    boundary_flux: jax.Array
    boundary_class: jax.Array
    wall_candidate: jax.Array
    saddle_candidate: jax.Array
    membership: jax.Array
    selected_cells: jax.Array
    field_coefficients: jax.Array
    saddle_form: topology.SaddleNormalForm
    normal_form_cells: jax.Array
    fragment_selected: jax.Array
    edge_interval: jax.Array
    qualified: jax.Array
    valid: jax.Array


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
    axis_sector = jnp.argmax(topology._normal_form_sectors(axis[None], form)[0])
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
        jnp.zeros(len(geometry.centre), bool),
        field.sampled_coefficient(geometry),
        form,
        saddle_live & near_saddle,
        jnp.zeros((len(geometry.centre), 2), bool),
        jnp.zeros((len(geometry.centre), 2, geometry.vertices.shape[1], 13, 2)),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    seed = jnp.argmin(jnp.sum((geometry.centre - axis) ** 2, axis=1))
    if frame_gate:
        provisional_core = connected_cells(geometry, approximate, sigma, seed)
    else:
        from scripts.prototypes.sub_millisecond.measure_wall_march import (
            connected_cells as ungated,
        )

        provisional_core = ungated(geometry, approximate, sigma, seed)
    wall_cell = jax.vmap(
        lambda point: jnp.argmax(topology._inside_cells(geometry, point))
    )(wall_points)
    wall_eligible = provisional_core[wall_cell]
    wall_scores = jnp.where(wall_eligible, sigma * wall_flux, -jnp.inf)
    wall_index = jnp.argmax(wall_scores)
    wall_score = jnp.where(
        jnp.any(wall_eligible),
        wall_scores[wall_index],
        jnp.where(saddle_live, -jnp.inf, raw_wall_score),
    )
    diverted = saddle_live & (saddle_score >= wall_score)
    level = sigma * jnp.maximum(wall_score, saddle_score)
    boundary = jnp.where(
        diverted, roots.position[saddle_index], wall_points[wall_index]
    )
    normal = diverted & near_saddle
    approximate = approximate._replace(
        boundary=boundary,
        boundary_flux=level,
        boundary_class=diverted.astype(jnp.int32),
        wall_candidate=sigma * wall_score,
        normal_form_cells=normal,
    )
    fraction, ordinary_valid = bell_fraction(geometry, approximate, sigma)
    capacity = min(16, len(geometry.centre))
    chosen = jnp.nonzero(normal, size=capacity, fill_value=0)[0]
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
    row = jnp.arange(capacity) < jnp.sum(normal)
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
        jnp.broadcast_to(core, (len(geometry.centre), 2)) & normal[:, None]
    )
    provisional = approximate._replace(
        membership=jnp.where(normal, saddle_fraction, fraction),
        fragment_selected=selected_sector,
        edge_interval=intervals,
    )
    if frame_gate:
        connected = connected_cells(geometry, provisional, sigma, seed)
    else:
        from scripts.prototypes.sub_millisecond.measure_wall_march import (
            connected_cells as ungated,
        )

        connected = ungated(geometry, provisional, sigma, seed)
    membership = jnp.where(connected, provisional.membership, 0.0)
    valid = (
        roots.valid[0]
        & jnp.all(jnp.isfinite(approximate.field_coefficients))
        & jnp.all(jnp.isfinite(membership))
        & (jnp.sum(normal) <= capacity)
        & jnp.all(jnp.where(row, fragments.valid, True))
        & jnp.all(jnp.where(connected & ~normal & (fraction > 0), ordinary_valid, True))
    )
    return provisional._replace(
        membership=membership,
        selected_cells=connected,
        fragment_selected=selected_sector & connected[:, None],
        qualified=valid & (~diverted | form.valid),
        valid=valid,
    )
