"""Vectorised fixed-shape topology read for the coarse moment map."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium import topology
from scripts.prototypes.sub_millisecond.measure_bell_cut import bell_polygon
from scripts.prototypes.sub_millisecond.measure_saddle_frame import connected_cells
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
    wall_flux = jax.vmap(lambda point: field.evaluate(point).value)(
        geometry.wall_points
    )
    wall_index = jnp.argmax(sigma * wall_flux)
    wall_score = sigma * wall_flux[wall_index]
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
    diverted = jnp.any(saddles) & (saddle_score >= wall_score)
    level = sigma * jnp.maximum(wall_score, saddle_score)
    boundary = jnp.where(
        diverted, roots.position[saddle_index], geometry.wall_points[wall_index]
    )
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
    radius = jnp.maximum(
        policy.normal_form_radius,
        policy.normal_form_pitch_floor * jnp.median(geometry.pitch),
    )
    normal = diverted & (
        topology._inside_cells(geometry, form.position)
        | (jnp.linalg.norm(geometry.centre - form.position, axis=1) < radius)
    )
    approximate = FastRead(
        axis,
        axis_flux,
        boundary,
        level,
        diverted.astype(jnp.int32),
        sigma * wall_score,
        sigma * saddle_score,
        jnp.zeros(len(geometry.centre)),
        jnp.zeros(len(geometry.centre), bool),
        field.sampled_coefficient(geometry),
        form,
        normal,
        jnp.zeros((len(geometry.centre), 2), bool),
        jnp.zeros((len(geometry.centre), 2, geometry.vertices.shape[1], 13, 2)),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    polygon, number, ordinary_valid = bell_polygon(geometry, approximate, sigma, 4)
    fraction = jnp.clip(polygon_area(polygon, number) / geometry.full_area, 0.0, 1.0)
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
    axis_sector = jnp.argmax(topology._normal_form_sectors(axis[None], form)[0])
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
    seed = jnp.argmin(jnp.sum((geometry.centre - axis) ** 2, axis=1))
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
