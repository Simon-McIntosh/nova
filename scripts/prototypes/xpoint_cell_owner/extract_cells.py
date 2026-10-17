"""Export the live X-point neighbourhood for an independent conic integral."""

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
from scripts.prototypes.fine_grid_convergence.context import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    context = build(args.cells, args.cache)
    saved = json.loads(args.start_json.read_text())
    moments = (
        context.reference
        if args.reference
        else jnp.asarray(saved["terminal_state"]["moments"])
    )
    reading = context.reading(moments)
    geometry = context.geometry
    density = program._density_coefficients(geometry, reading, context.profile)
    ordinary, _, _ = bell_fraction_with_crossings(
        geometry, reading, context.convention.sigma
    )
    neighbourhood = np.asarray(reading.normal_form_cells)
    selected = np.flatnonzero(neighbourhood)
    rows = []
    for index in selected:
        count = int(geometry.vertex_count[index])
        rows.append(
            {
                "cell": int(index),
                "vertices": np.asarray(geometry.vertices[index, :count]).tolist(),
                "centre": np.asarray(geometry.centre[index]).tolist(),
                "pitch": float(geometry.pitch[index]),
                "full_area": float(geometry.full_area[index]),
                "field_coefficients": np.asarray(
                    reading.field_coefficients[index]
                ).tolist(),
                "density_coefficients": np.asarray(density[index]).tolist(),
                "normal_fraction": float(reading.geometric_membership[index]),
                "ordinary_fraction": float(ordinary[index]),
                "booked": bool(reading.booking_cells[index]),
                "path_weight": float(reading.path_weight[index]),
            }
        )
    result = {
        "cells": len(context.reference),
        "saddle": np.asarray(reading.saddle_form.position).tolist(),
        "axis": np.asarray(reading.axis).tolist(),
        "boundary_level": float(reading.boundary_flux),
        "sigma": float(context.convention.sigma),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"EXTRACT cells={result['cells']} neighbourhood={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
