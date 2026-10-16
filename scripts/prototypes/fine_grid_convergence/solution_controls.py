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
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence.handoff import map_moments
from scripts.prototypes.fine_grid_convergence.runtime_solve import (
    factor,
    residual,
    without_profile,
)
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def panel_data(context, moments, operands, profile, residual_value, converged):
    """Build a shared-level contour receipt for the terminal forward state."""
    machine = context.machine
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state("diverted-single-null", context.exact, coordinates)
    )
    exterior, _ = _exterior(context.cache, context.carrier_key)
    operator = fixture.forward_operator(context.source, machine)
    terminal = jax.jit(
        lambda point, carrier: map_moments(point, *carrier[:7], profile, *carrier[7:])
    )(moments, operands)
    solved_flux = exterior + _image(operator, moments)
    return {
        "coordinates": np.asarray(machine.node).tolist(),
        "wall": np.asarray(machine.wall_node).tolist(),
        "solved_flux": np.asarray(solved_flux)[: len(machine.node)].tolist(),
        "analytic_flux": np.asarray(analytic)[: len(machine.node)].tolist(),
        "solved_roots": np.asarray(terminal.roots).tolist(),
        "analytic_roots": np.asarray(context.seeds).tolist(),
        "terminal_residual": residual_value,
        "converged": converged,
    }


def chord_two(states, operands, scale, lu, pivots, profile):
    """Two fixed-factor corrections with every large operand passed at runtime."""

    def one(state):
        for _ in range(2):
            direction = jsp_linalg.lu_solve(
                (lu, pivots), -residual(state, operands, scale, profile)
            )
            state = state + direction
        return jnp.max(jnp.abs(residual(state, operands, scale, profile)))

    return jax.vmap(one)(states)


def gauss_newton_gradient(state, operands, scale, profile):
    """Form the transposed map Jacobian by forward products through the null solve."""
    value = residual(state, operands, scale, profile)
    count = state.size
    width = 16
    blocks = (count + width - 1) // width
    columns = jnp.zeros((blocks * width, count), state.dtype)

    def body(block, packed):
        first = block * width
        indices = first + jnp.arange(width)
        directions = (indices[:, None] == jnp.arange(count)[None]).astype(state.dtype)

        def column(direction):
            return jax.jvp(
                lambda point: residual(point, operands, scale, profile),
                (state,),
                (direction,),
            )[1]

        return jax.lax.dynamic_update_slice(
            packed, jax.vmap(column)(directions), (first, 0)
        )

    columns = jax.lax.fori_loop(0, blocks, body, columns)
    return value, columns[:count] @ value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-only", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    saved = json.loads(args.start_json.read_text())
    if "terminal_state" not in saved or not saved.get("history"):
        raise ValueError("control requires a cold receipt with a terminal state")
    context = build(args.cells, args.cache)
    state = jnp.asarray(saved["terminal_state"]["moments"]).reshape(-1)
    state = state / jnp.tile(context.scale, len(context.reference))
    operands = without_profile(context.operands)
    scale = context.scale
    profile = context.profile
    result = {"cells": len(context.reference), "taylor": [], "batches": []}
    result["panel"] = panel_data(
        context,
        state.reshape((-1, 3)) * scale,
        operands,
        profile,
        saved["history"][-1]["residual"],
        bool(saved.get("converged")),
    )
    save(args.output, result)
    print(
        f"PANEL cells={result['cells']} converged={result['panel']['converged']}",
        flush=True,
    )
    if args.panel_only:
        return
    if not saved.get("converged"):
        raise ValueError("control requires a converged cold receipt")

    mapped = jax.jit(
        lambda point, carrier, unit: residual(point, carrier, unit, profile)
    )
    centre, gradient = jax.jit(
        lambda point, carrier, unit: gauss_newton_gradient(
            point, carrier, unit, profile
        )
    )(state, operands, scale)
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
            lambda probe: residual(probe, ops, unit, profile),
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

    offsets = jnp.arange(128, dtype=state.dtype)[:, None]
    perturbations = jnp.sin(points[None] * 0.17 + offsets * 0.31)
    perturbations = perturbations / jnp.linalg.norm(perturbations, axis=1)[:, None]
    neighbours = state[None] + 1e-4 * perturbations
    for width in (1, 16, 128):
        batch = neighbours[:width]
        started = perf_counter()
        program = (
            jax.jit(
                lambda states, carrier, unit, matrix, piv: chord_two(
                    states, carrier, unit, matrix, piv, profile
                )
            )
            .lower(batch, operands, scale, lu, pivots)
            .compile()
        )
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
