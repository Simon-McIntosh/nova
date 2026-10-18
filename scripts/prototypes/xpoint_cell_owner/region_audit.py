"""Audit which region the owner integral actually integrates.

For every owner cell at one carrier size this reports the area fraction and
current of the region under four independent constructions:

``traced``
    the submitted half-plane: the axis-facing side of the saddle, bounded by
    the cell polygon and the zero level of the fitted cell quadratic,
``dense``
    the same region integrated by a far higher-order rule,
``lobe``
    the accessor's own normal-form booking, which clips the cell against the
    four separatrix rays, and
``grid``
    a direct sample of the analytic flux over the axis-facing positive points.

A fifth reading, ``ordinary``, is the cell's ordinary cut booking: the whole
region the cell would carry if the owner region did not exist, and therefore
the total the owner part and the remainder have to sum to.

The five give the region the traced integral integrates, the region it was
meant to integrate, and the size of the gap between them.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import current
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner.analytic_owner_probe import sample_cell
from scripts.prototypes.xpoint_cell_owner.owner_map import integrate_cell
from scripts.prototypes.xpoint_cell_owner.region_reference import region_moments


class _Unit:
    """Density profile of unit value, so a booking reports its own area."""

    @staticmethod
    def current_density(radial, normalized):
        return jnp.ones_like(jnp.asarray(radial, dtype=float))


def _cut_axes(indices):
    return (0, 0, 0, 0, 0, None, None, None, None, 0)


def _traced_regions(reading, geometry, profile, sigma, indices):
    """Return the traced region moments under the submitted cut, and its area.

    The submitted cut is what ``owned_cut`` books: the fitted cell quadratic
    with its constant moved to the boundary level, restricted to the
    axis-facing side of the saddle.
    """
    from functools import partial

    span = reading.boundary_flux - reading.axis_flux
    rule = partial(
        integrate_cell,
        axis_component="halfplane",
        density_mode="fitted",
        profile=profile,
        span=span,
    )
    selected = jax.vmap(rule, in_axes=_cut_axes(indices) + (None,))(
        geometry.vertices[indices],
        geometry.vertex_count[indices],
        geometry.centre[indices],
        geometry.pitch[indices],
        reading.field_coefficients[indices],
        reading.boundary_flux,
        sigma,
        reading.saddle_form.position,
        reading.axis,
        program._density_coefficients(geometry, reading, profile)[indices],
        None,
    )
    return np.asarray(selected[0])


def _lobe_region(reading, geometry, profile, capacity=16):
    """Area and current booked over the normal-form lobe by the accessor rings."""
    active = reading.booking_cells & (reading.membership > 0)
    count = int(np.sum(np.asarray(active)))
    indices = jnp.nonzero(active, size=capacity, fill_value=0)[0]
    origin = geometry.centre
    pitch = geometry.pitch

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
        slot = jnp.arange(rings.shape[1])
        following = jnp.where(slot[None] + 1 < sizes[:, None], slot[None] + 1, 0)
        end = jnp.take_along_axis(rings, following[..., None], axis=1)
        first = rings - origin[index]
        second = end - origin[index]
        area = 0.5 * (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
        area = jnp.where(slot[None] < sizes[:, None], area, 0.0)
        barycentric = jnp.asarray(((2 / 3, 1 / 6), (1 / 6, 2 / 3), (1 / 6, 1 / 6)))
        points = (
            origin[index]
            + barycentric[:, 0, None] * first[..., None, :]
            + barycentric[:, 1, None] * second[..., None, :]
        )
        local = (points - origin[index]) / pitch[index]
        coefficient = reading.field_coefficients[index]
        flux = (
            coefficient[0]
            + coefficient[1] * local[..., 0]
            + coefficient[2] * local[..., 1]
            + coefficient[3] * local[..., 0] ** 2
            + coefficient[4] * local[..., 0] * local[..., 1]
            + coefficient[5] * local[..., 1] ** 2
        )
        normalized = (flux - reading.axis_flux) / (
            reading.boundary_flux - reading.axis_flux
        )
        density = profile.current_density(points[..., 0], normalized)
        weighted = area[..., None] * density / 3
        displacement = points - origin[index]
        moments = jnp.stack(
            (
                jnp.sum(weighted),
                jnp.sum(weighted * displacement[..., 0]),
                jnp.sum(weighted * displacement[..., 1]),
            )
        )
        return moments

    packed = jax.vmap(one)(indices)
    live = jnp.arange(capacity) < count
    scattered = (
        jnp.zeros((len(geometry.centre), 3), dtype=packed.dtype)
        .at[indices]
        .add(jnp.where(live[:, None], packed, 0.0))
    )
    return np.asarray(scattered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    count = len(context.reference)
    reading = context.reading(context.reference)
    geometry = context.geometry
    profile = context.profile
    sigma = context.convention.sigma
    span = float(reading.boundary_flux - reading.axis_flux)
    owner = np.flatnonzero(np.asarray(reading.booking_cells))
    indices = jnp.asarray(owner)
    traced = _traced_regions(reading, geometry, profile, sigma, indices)
    lobe = _lobe_region(reading, geometry, profile)
    lobe_area = _lobe_region(reading, geometry, _Unit())
    ordinary, ordinary_valid, _ = bell_moments(geometry, reading, profile, sigma)
    ordinary = np.asarray(ordinary)
    result = {
        "kind": "region-audit",
        "cells": count,
        "span": span,
        "owner_cells": [int(cell) for cell in owner],
        "rows": [],
    }
    path = args.output
    path.write_text(json.dumps(result, indent=2) + "\n")
    exact = None
    for position, cell in enumerate(owner):
        vertices = np.asarray(
            geometry.vertices[cell, : int(geometry.vertex_count[cell])]
        )
        full_area = float(geometry.full_area[cell])
        dense, dense_valid = region_moments(
            vertices,
            np.asarray(geometry.centre[cell]),
            float(geometry.pitch[cell]),
            np.asarray(reading.field_coefficients[cell]),
            float(reading.boundary_flux),
            float(sigma),
            np.asarray(reading.saddle_form.position),
            np.asarray(reading.axis),
            np.zeros(6),
            span=span,
            density_mode="fitted",
        )
        if exact is None:
            from nova.equilibrium.analytic_single_null import (
                cerfon_freidberg_single_null,
            )

            exact = cerfon_freidberg_single_null()
        grid = sample_cell(exact, vertices, 1024, float(sigma))
        row = {
            "cell": int(cell),
            "pitch": float(geometry.pitch[cell]),
            "full_area": full_area,
            "traced_area_fraction": float(traced[position][0] / full_area),
            "traced_current": float(traced[position][1]),
            "dense_area_fraction": float(dense[0] / full_area),
            "dense_valid": bool(dense_valid),
            "lobe_area_fraction": float(lobe_area[cell][0] / full_area),
            "lobe_current": float(lobe[cell][1]),
            "ordinary_area_fraction": float(ordinary[cell][0] / full_area),
            "ordinary_current": float(ordinary[cell][1]),
            "ordinary_valid": bool(ordinary_valid[cell]),
            "grid_area_fraction": grid["area_fraction"],
            "grid_selected": grid["selected_count"],
        }
        result["rows"].append(row)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"CELL {json.dumps(row)}", flush=True)
    print(f"SUMMARY cells={count} owners={len(owner)}", flush=True)


if __name__ == "__main__":
    main()
