"""Diagnose how well the located saddle anchors the owner conic at each
carrier size.

From the reference moments alone the probe reports, per resolved owner cell:

* the residual flux level the fitted cell quadratic carries at the located
  saddle, and its slope there, both in units of the carrier span;
* how far the fitted quadratic's own stationary point sits from the located
  saddle, in physical length.

A conic anchored at a true flux stationary point has zero slope there. A large
slope in span units means the anchor is misplaced rather than the conic being a
poor model of the cell.
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
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner.owner_map import fitted_value, map_moments
from scripts.prototypes.xpoint_cell_owner.runtime_solve import without_profile


def pitch_columns(pitch):
    array = jnp.asarray(pitch)
    return array[:, None] if array.ndim == 1 else array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=("halfplane", "bell", "saddle"),
        default="halfplane",
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    operands = without_profile(context.operands)
    profile = context.profile
    geometry = context.geometry
    mapped = jax.jit(
        lambda moments, carrier: map_moments(
            moments, *carrier[:7], profile, *carrier[7:], component=args.component
        )
    )
    reading = context.reading(context.reference)
    state = mapped(context.reference, operands)
    residual = np.asarray(state.moments - context.reference)
    pitch = pitch_columns(geometry.pitch)
    scale = float(jnp.max(jnp.abs(geometry.pitch)))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    saddle = np.asarray(reading.saddle_form.position)
    owner = np.asarray(reading.normal_form_cells)
    booking = np.asarray(reading.booking_cells)
    print(
        f"READ cells={len(geometry.centre)} owner={int(owner.sum())} "
        f"booking={int(booking.sum())} saddle={saddle.tolist()} span={span:.12g}",
        flush=True,
    )

    centre = jnp.asarray(geometry.centre)
    if centre.ndim == 1:
        centre = centre[:, None]
    local = (jnp.asarray(saddle)[None] - centre[owner]) / pitch[owner]
    coefficients = reading.field_coefficients[owner]
    level = fitted_value(coefficients, local)
    slope = coefficients[:, 1:3]
    stationary = -jnp.linalg.solve(
        jnp.stack(
            (
                jnp.stack((2 * coefficients[:, 3], coefficients[:, 4]), axis=1),
                jnp.stack((coefficients[:, 4], 2 * coefficients[:, 5]), axis=1),
            ),
            axis=1,
        ),
        slope[:, :, None],
    )[:, :, 0]
    offset = jnp.linalg.norm(stationary, axis=1) * jnp.linalg.norm(pitch[owner], axis=1)
    value_gap = np.asarray(jnp.abs(level) / span)
    slope_span = np.asarray(
        jnp.linalg.norm(slope, axis=1) / pitch[owner].min(axis=1) / span
    )
    offset = np.asarray(offset)
    magnitudes = np.linalg.norm(residual, axis=1)
    result = {
        "cells": len(context.reference),
        "component": args.component,
        "span": span,
        "saddle": saddle.tolist(),
        "grid_pitch": float(scale),
        "owner_cells": int(owner.sum()),
        "booking_cells": int(booking.sum()),
        "value_gap": {
            "max": float(value_gap.max()),
            "median": float(np.median(value_gap)),
        },
        "slope_over_span": {
            "max": float(slope_span.max()),
            "median": float(np.median(slope_span)),
        },
        "stationary_offset": {
            "max": float(offset.max()),
            "median": float(np.median(offset)),
            "min": float(offset.min()),
        },
        "moment_mass": {
            "owner": float(magnitudes[owner].sum()),
            "non_owner": float(magnitudes[~owner].sum()),
            "owner_share": float(
                magnitudes[owner].sum() / max(magnitudes.sum(), 1e-300)
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PROBE {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
