"""Measure the booked current response to an ordinary-cut handoff."""

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
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.fine_grid_convergence.context import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    context = build(500, args.cache)
    saved = json.loads(args.start_json.read_text())
    moments = jnp.asarray(saved["terminal_state"]["moments"])
    original_saddle = program._saddle_moments
    result = {"cells": len(context.reference), "rows": []}
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):

        def blended(geometry, reading, profile, capacity=16):
            normal, valid = original_saddle(geometry, reading, profile, capacity)
            ordinary, ordinary_valid, _ = bell_moments(
                geometry, reading, profile, context.convention.sigma
            )
            mixed = normal.at[8].set(
                (1 - fraction) * normal[8] + fraction * ordinary[8]
            )
            return mixed, valid & ordinary_valid[8]

        program._saddle_moments = blended
        state = jax.jit(context.map)(moments)
        jax.block_until_ready(state)
        residual = np.asarray(state.moment_residual / context.scale)
        row = {
            "ordinary_fraction": fraction,
            "residual": float(np.max(np.abs(residual))),
            "cell_8_current_residual": float(residual[8, 0]),
            "peak_cell": int(
                np.unravel_index(np.argmax(np.abs(residual)), residual.shape)[0]
            ),
            "valid": bool(state.booking_valid),
        }
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"ROW {json.dumps(row)}", flush=True)
    program._saddle_moments = original_saddle


if __name__ == "__main__":
    main()
