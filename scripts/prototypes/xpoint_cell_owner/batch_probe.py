"""Measure bounded-memory chord batches with one shared Newton factor."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

import nova
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner.runtime_solve import (
    factor,
    residual,
    without_profile,
)


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


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
    saved = json.loads(args.start_json.read_text())
    if not saved.get("converged") or "terminal_state" not in saved:
        raise ValueError("batch probe requires a converged fine cold state")
    context = build(2000, args.cache)
    state = jnp.asarray(saved["terminal_state"]["moments"]).reshape(-1)
    state = state / jnp.tile(context.scale, len(context.reference))
    operands = without_profile(context.operands)
    profile = context.profile
    scale = context.scale
    result = {"cells": len(context.reference), "rows": []}
    save(args.output, result)
    started = perf_counter()
    compiled_factor = (
        jax.jit(lambda point, carrier, unit: factor(point, carrier, unit, profile))
        .lower(state, operands, scale)
        .compile()
    )
    result["factor_compile_seconds"] = perf_counter() - started
    result["factor_executable_bytes"] = (
        compiled_factor.runtime_executable().size_of_generated_code_in_bytes()
    )
    lu, pivots = compiled_factor(state, operands, scale)
    jax.block_until_ready(lu)
    save(args.output, result)
    print(
        f"FACTOR seconds={result['factor_compile_seconds']:.3f} "
        f"bytes={result['factor_executable_bytes']}",
        flush=True,
    )
    coordinates = jnp.arange(state.size, dtype=state.dtype)
    offsets = jnp.arange(128, dtype=state.dtype)[:, None]
    perturbations = jnp.sin(coordinates[None] * 0.17 + offsets * 0.31)
    perturbations /= jnp.linalg.norm(perturbations, axis=1)[:, None]
    neighbours = state[None] + 1e-4 * perturbations

    def one(point, carrier, unit, matrix, pivot):
        for _ in range(2):
            direction = jsp_linalg.lu_solve(
                (matrix, pivot), -residual(point, carrier, unit, profile)
            )
            point = point + direction
        return jnp.max(jnp.abs(residual(point, carrier, unit, profile)))

    def ordinary(batch, carrier, unit, matrix, pivot):
        return jax.vmap(one, in_axes=(0, None, None, None, None))(
            batch, carrier, unit, matrix, pivot
        )

    def bounded(batch, carrier, unit, matrix, pivot):
        chunks = batch.reshape((8, 16, batch.shape[-1]))

        def step(carry, chunk):
            values = ordinary(chunk, carrier, unit, matrix, pivot)
            return carry, values

        _, values = jax.lax.scan(step, None, chunks)
        return values.reshape(-1)

    for width in (1, 16, 128):
        batch = neighbours[:width]
        method = bounded if width == 128 else ordinary
        started = perf_counter()
        program = jax.jit(method).lower(batch, operands, scale, lu, pivots).compile()
        compile_seconds = perf_counter() - started
        executable_bytes = (
            program.runtime_executable().size_of_generated_code_in_bytes()
        )
        residuals = np.asarray(program(batch, operands, scale, lu, pivots))
        times = []
        for _ in range(5):
            started = perf_counter()
            jax.block_until_ready(program(batch, operands, scale, lu, pivots))
            times.append((perf_counter() - started) * 1000 / width)
        row = {
            "width": width,
            "chunked": width == 128,
            "converged": int(np.sum(residuals <= 1e-12)),
            "worst_residual": float(np.max(residuals)),
            "per_state_wall_ms": float(np.median(times)),
            "compile_seconds": compile_seconds,
            "executable_bytes": executable_bytes,
        }
        result["rows"].append(row)
        save(args.output, result)
        print(f"BATCH {json.dumps(row)}", flush=True)
    result["headline_pass"] = all(
        row["converged"] == row["width"]
        and row["per_state_wall_ms"] <= 0.90
        and row["compile_seconds"] <= 60
        and row["executable_bytes"] <= 50_000_000
        for row in result["rows"]
    )
    save(args.output, result)
    print(f"SUMMARY headline_pass={result['headline_pass']}", flush=True)
    if not result["headline_pass"]:
        raise ValueError("bounded batch misses the throughput or convergence gate")


if __name__ == "__main__":
    main()
