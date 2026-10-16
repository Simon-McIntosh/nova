"""Locate the selected-cut change along a stalled Newton direction."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

import nova
from scripts.prototypes.fine_grid_convergence.context import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    context = build(args.cells, args.cache)
    saved = json.loads(args.start_json.read_text())
    shape = context.reference.shape
    base = jnp.asarray(saved["terminal_state"]["moments"]) / context.scale

    def equation(point):
        state = context.map(point.reshape(shape) * context.scale)
        return (state.moment_residual / context.scale).reshape(-1)

    residual, tangent = jax.linearize(equation, base.reshape(-1))
    residual = np.asarray(residual)
    product = jax.jit(tangent)
    operator = LinearOperator(
        (residual.size, residual.size),
        matvec=lambda vector: np.array(product(jnp.asarray(vector)), copy=True),
        dtype=np.float64,
    )
    direction, code = gmres(
        operator, -residual, restart=48, maxiter=6, rtol=1e-10, atol=0
    )
    result = {
        "cells": len(context.reference),
        "base_residual": float(np.max(np.abs(residual))),
        "linear_code": int(code),
        "linear_residual_ratio": float(
            np.linalg.norm(operator @ direction + residual) / np.linalg.norm(residual)
        ),
        "direction_norm": float(np.linalg.norm(direction)),
        "steps": [],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"BASE residual={result['base_residual']:.12g} "
        f"linear_code={code} direction_norm={result['direction_norm']:.12g}",
        flush=True,
    )
    mapped = jax.jit(context.map)
    inspect = jax.jit(context.reading)
    tangent = jnp.asarray(direction)
    for amount in (0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 8e-4, 1e-3):
        moments = (base.reshape(-1) + amount * tangent).reshape(shape) * context.scale
        state = mapped(moments)
        reading = inspect(moments)
        jax.block_until_ready((state, reading))
        cell_ids = (47, 50)
        row = {
            "step": amount,
            "residual": float(jnp.max(jnp.abs(state.moment_residual / context.scale))),
            "boundary_class": int(reading.boundary_class),
            "booking_ids": np.flatnonzero(np.asarray(reading.booking_cells)).tolist(),
            "supported_cells": int(jnp.sum(reading.selected_cells)),
            "saddle": np.asarray(reading.saddle_form.position).tolist(),
            "cells": [
                {
                    "cell": cell,
                    "booking": bool(reading.booking_cells[cell]),
                    "path_weight": float(reading.path_weight[cell]),
                    "normal_form": bool(reading.normal_form_cells[cell]),
                    "membership": float(reading.membership[cell]),
                    "geometric_membership": float(reading.geometric_membership[cell]),
                    "current_residual": float(
                        (state.moment_residual / context.scale)[cell, 0]
                    ),
                }
                for cell in cell_ids
            ],
        }
        result["steps"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        neighbour = row["cells"][1]
        print(
            f"STEP h={amount:.8g} residual={row['residual']:.12g} "
            f"booking={row['booking_ids']} weight50={neighbour['path_weight']:.6g} "
            f"cell50={neighbour['current_residual']:.12g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
