"""Measure a converged moment root and runtime-argument chord neighbours."""

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
from scripts.prototypes.fine_grid_convergence.runtime_solve import factor, residual


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def chord_two(states, operands, scale, lu, pivots):
    """Two fixed-factor corrections with every large operand passed at runtime."""

    def one(state):
        for _ in range(2):
            direction = jsp_linalg.lu_solve(
                (lu, pivots), -residual(state, operands, scale)
            )
            state = state + direction
        return jnp.max(jnp.abs(residual(state, operands, scale)))

    return jax.vmap(one)(states)


def gauss_newton_gradient(state, operands, scale):
    """Apply the transposed map Jacobian without capturing carrier arrays."""
    value, pullback = jax.vjp(lambda probe: residual(probe, operands, scale), state)
    return value, pullback(value)[0]


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
    saved = json.loads(args.start_json.read_text())
    if not saved.get("converged"):
        raise ValueError("control requires a converged cold receipt")
    context = build(args.cells, args.cache)
    state = jnp.asarray(saved["terminal_state"]["moments"]).reshape(-1)
    state = state / jnp.tile(context.scale, len(context.reference))
    operands = context.operands
    scale = context.scale
    result = {"cells": len(context.reference), "taylor": [], "batches": []}
    save(args.output, result)

    mapped = jax.jit(residual)
    centre, gradient = jax.jit(gauss_newton_gradient)(state, operands, scale)
    centre_np = np.asarray(centre)
    result["residual"] = float(np.max(np.abs(centre_np)))
    result["gauss_newton_gradient_two"] = float(jnp.linalg.norm(gradient))
    print(
        f"ROOT cells={result['cells']} residual={result['residual']:.12g} "
        f"gradient={result['gauss_newton_gradient_two']:.12g}",
        flush=True,
    )
    save(args.output, result)

    points = jnp.arange(state.size, dtype=state.dtype)
    directions = [
        jnp.sin(points * frequency + phase)
        for frequency, phase in ((0.13, 0.3), (0.29, 1.4), (0.47, 2.7))
    ]
    steps = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    tangent_program = jax.jit(
        lambda point, direction, ops, unit: jax.jvp(
            lambda probe: residual(probe, ops, unit),
            (point,),
            (direction,),
        )
    )
    for index, direction in enumerate(directions):
        direction = direction / jnp.linalg.norm(direction)
        value, tangent = tangent_program(state, direction, operands, scale)
        rows = []
        for step in steps:
            trial = mapped(state + step * direction, operands, scale)
            remainder = np.asarray((trial - value - step * tangent).reshape(-1, 3))
            rows.append(
                [float(np.linalg.norm(remainder[:, block])) for block in range(3)]
            )
        slopes = []
        for first, second in zip(rows, rows[1:]):
            slopes.append(
                [
                    float(np.log(left / right) / np.log(10))
                    if left > 0 and right > 0
                    else None
                    for left, right in zip(first, second)
                ]
            )
        row = {"direction": index, "steps": steps, "remainders": rows, "slopes": slopes}
        result["taylor"].append(row)
        save(args.output, result)
        print(f"TAYLOR direction={index} slopes={slopes}", flush=True)

    started = perf_counter()
    compiled_factor = jax.jit(factor).lower(state, operands, scale).compile()
    result["factor_compile_seconds"] = perf_counter() - started
    result["factor_executable_bytes"] = (
        compiled_factor.runtime_executable().size_of_generated_code_in_bytes()
    )
    lu, pivots = compiled_factor(state, operands, scale)
    jax.block_until_ready(lu)
    save(args.output, result)

    offsets = jnp.arange(128, dtype=state.dtype)[:, None]
    perturbations = jnp.sin(points[None] * 0.17 + offsets * 0.31)
    perturbations = perturbations / jnp.linalg.norm(perturbations, axis=1)[:, None]
    neighbours = state[None] + 1e-4 * perturbations
    for width in (1, 16, 128):
        batch = neighbours[:width]
        started = perf_counter()
        program = jax.jit(chord_two).lower(batch, operands, scale, lu, pivots).compile()
        compile_seconds = perf_counter() - started
        size_bytes = program.runtime_executable().size_of_generated_code_in_bytes()
        values = np.asarray(program(batch, operands, scale, lu, pivots))
        times = []
        for _ in range(5):
            started = perf_counter()
            jax.block_until_ready(program(batch, operands, scale, lu, pivots))
            times.append((perf_counter() - started) * 1000 / width)
        row = {
            "width": width,
            "converged": int(np.sum(values <= 1e-12)),
            "worst_residual": float(np.max(values)),
            "per_state_wall_ms": float(np.median(times)),
            "compile_seconds": compile_seconds,
            "executable_bytes": size_bytes,
        }
        result["batches"].append(row)
        save(args.output, result)
        print(f"BATCH {json.dumps(row)}", flush=True)
    result["taylor_pass"] = all(
        slope is not None and 1.8 <= slope <= 2.2
        for direction in result["taylor"]
        for row in direction["slopes"][1:3]
        for slope in row
    )
    result["batch_pass"] = all(
        row["converged"] == row["width"]
        and row["per_state_wall_ms"] <= 0.90
        and row["compile_seconds"] <= 60
        and row["executable_bytes"] <= 50_000_000
        for row in result["batches"]
    )
    save(args.output, result)
    print(
        f"CONTROL taylor_pass={result['taylor_pass']} "
        f"batch_pass={result['batch_pass']}",
        flush=True,
    )
    if not result["taylor_pass"] or not result["batch_pass"]:
        raise ValueError("solution controls did not meet their measured limits")


if __name__ == "__main__":
    main()
