"""Compare a runtime Newton factor and step with a host dense solve."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve as scipy_solve

import nova
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence.runtime_solve import (
    DIRECTION_BLOCK,
    advance,
    residual,
    without_profile,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    context = build(250, args.cache)
    operands = without_profile(context.operands)
    profile = context.profile
    state = (context.reference / context.scale).reshape(-1)
    size = state.size
    first = jax.jit(
        lambda point, carrier, unit, lu, pivots: advance(
            point, carrier, unit, lu, pivots, True, profile
        )
    )
    state, first_residual, _, _, first_factor = first(
        state, operands, context.scale, jnp.eye(size), jnp.arange(size, dtype=jnp.int32)
    )
    jax.block_until_ready(state)
    result = {
        "cells": len(context.reference),
        "first_residual": float(first_residual),
        "first_factor": float(first_factor),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"FIRST {json.dumps(result)}", flush=True)

    def matrix_and_residual(point, carrier, unit):
        current = residual(point, carrier, unit, profile)
        blocks = (point.size + DIRECTION_BLOCK - 1) // DIRECTION_BLOCK
        rows = jnp.zeros((blocks * DIRECTION_BLOCK, point.size), point.dtype)

        def body(block, packed):
            indices = block * DIRECTION_BLOCK + jnp.arange(DIRECTION_BLOCK)
            directions = (indices[:, None] == jnp.arange(point.size)[None]).astype(
                point.dtype
            )

            def column(direction):
                return jax.jvp(
                    lambda trial: residual(trial, carrier, unit, profile),
                    (point,),
                    (direction,),
                )[1]

            products = jax.vmap(column)(directions)
            return jax.lax.dynamic_update_slice(
                packed, products, (block * DIRECTION_BLOCK, 0)
            )

        rows = jax.lax.fori_loop(0, blocks, body, rows)
        return rows[: point.size].T, current

    matrix, vector = jax.jit(matrix_and_residual)(state, operands, context.scale)
    device_step = jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(matrix), -vector)
    jax.block_until_ready(device_step)
    host_matrix = np.asarray(matrix)
    host_vector = np.asarray(vector)
    result["matrix_nonfinite"] = int(
        np.size(host_matrix) - np.isfinite(host_matrix).sum()
    )
    result["residual_nonfinite"] = int(
        np.size(host_vector) - np.isfinite(host_vector).sum()
    )
    result["matrix_norm_inf"] = float(np.linalg.norm(host_matrix, ord=np.inf))
    result["residual_norm_inf"] = float(np.linalg.norm(host_vector, ord=np.inf))
    result["device_step_nonfinite"] = int(
        size - np.isfinite(np.asarray(device_step)).sum()
    )
    if result["matrix_nonfinite"] or result["residual_nonfinite"]:
        try:
            scipy_solve(host_matrix, -host_vector, assume_a="gen")
        except ValueError as error:
            result["host_solve_refusal"] = f"{type(error).__name__}: {error}"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"SUMMARY {json.dumps(result)}", flush=True)
        return
    host_step = scipy_solve(host_matrix, -host_vector, assume_a="gen")
    device_array = np.asarray(device_step)
    result["host_step_nonfinite"] = int(size - np.isfinite(host_step).sum())
    result["host_linear_residual"] = float(
        np.linalg.norm(host_matrix @ host_step + host_vector, ord=np.inf)
    )
    result["device_linear_residual"] = float(
        np.linalg.norm(host_matrix @ device_array + host_vector, ord=np.inf)
    )
    result["step_relative_difference"] = float(
        np.linalg.norm(device_array - host_step)
        / max(np.linalg.norm(host_step), 1e-300)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SUMMARY {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
