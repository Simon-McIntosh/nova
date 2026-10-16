"""Locate a stalled moment term and compare its two continuous cut moments."""

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
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_map_production.kernel import bell_fraction_with_crossings
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence.handoff import map_moments
from scripts.prototypes.fine_grid_convergence.runtime_solve import without_profile
from scripts.prototypes.fine_grid_convergence.solution_controls import (
    gauss_newton_gradient,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    receipt = json.loads(args.start_json.read_text())
    source = "terminal" if "terminal_state" in receipt else "reference"
    moments = jnp.asarray(
        receipt["terminal_state"]["moments"]
        if source == "terminal"
        else context.reference
    )
    operands = without_profile(context.operands)
    profile = context.profile
    state = jax.jit(
        lambda point, carrier: map_moments(point, *carrier[:7], profile, *carrier[7:])
    )(moments, operands)
    scaled = np.asarray(state.moment_residual / context.scale)
    magnitude = np.max(np.abs(scaled), axis=1)
    hot = np.argsort(magnitude)[-10:][::-1]
    reading = context.reading(moments)
    normal, normal_valid = program._saddle_moments(context.geometry, reading, profile)
    ordinary, ordinary_valid, _ = bell_moments(
        context.geometry, reading, profile, context.convention.sigma
    )
    ordinary_fraction, fraction_valid, _ = bell_fraction_with_crossings(
        context.geometry, reading, context.convention.sigma
    )
    area_gap = reading.geometric_membership - ordinary_fraction
    blend = -jnp.expm1(-jnp.square(area_gap / 0.043))
    selected = np.union1d(
        hot,
        np.flatnonzero(np.asarray(reading.booking_cells & (reading.membership > 0))),
    )
    rows = []
    for index in selected:
        rows.append(
            {
                "cell": int(index),
                "scaled_residual": scaled[index].tolist(),
                "path_weight": float(reading.path_weight[index]),
                "normal_fraction": float(reading.geometric_membership[index]),
                "ordinary_fraction": float(ordinary_fraction[index]),
                "blend": float(blend[index]),
                "normal_moments": np.asarray(normal[index]).tolist(),
                "ordinary_moments": np.asarray(ordinary[index]).tolist(),
                "booked": bool(reading.booking_cells[index]),
                "normal_valid": bool(normal_valid),
                "ordinary_valid": bool(ordinary_valid[index] & fraction_valid[index]),
            }
        )
    result = {
        "cells": len(context.reference),
        "state_source": source,
        "residual": float(jnp.max(jnp.abs(scaled))),
        "support_cells": int(jnp.sum(reading.selected_cells)),
        "booking_ids": np.flatnonzero(np.asarray(reading.booking_cells)).tolist(),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    flat = (moments / context.scale).reshape(-1)
    _, gradient = jax.jit(
        lambda point, carrier, unit: gauss_newton_gradient(
            point, carrier, unit, profile
        )
    )(flat, operands, context.scale)
    result["gauss_newton_gradient_two"] = float(jnp.linalg.norm(gradient))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"SUMMARY cells={result['cells']} residual={result['residual']:.12g} "
        f"gradient={result['gauss_newton_gradient_two']:.12g} "
        f"peak_cell={int(hot[0])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
