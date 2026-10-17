"""Measure a diverted map and its cold solve from one carrier assembly."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import nova
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_map_production.qualification import require_map
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments
from scripts.prototypes.xpoint_cell_owner.runtime_solve import (
    advance,
    without_profile,
)
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--component",
        choices=(
            "halfplane",
            "bell",
            "saddle",
            "pinned",
            "anchored",
            "shifted",
            "ordinary",
            "oriented",
        ),
        default="halfplane",
    )
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument(
        "--past-map-guard",
        action="store_true",
        help=(
            "Solve from a map that exceeds its flux guard and record the "
            "excess. Diagnostic only: the iteration then converges to a root "
            "of the map as booked, which is self-consistency and not flux "
            "accuracy, and the run is labelled as such in its receipt."
        ),
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    count = len(context.reference)
    result = {
        "kind": "diverted",
        "component": args.component,
        "cells": count,
        "history": [],
    }
    save(args.output, result)
    operands = without_profile(context.operands)
    profile = context.profile

    mapped = jax.jit(
        lambda moments, carrier: map_moments(
            moments, *carrier[:7], profile, *carrier[7:], component=args.component
        )
    )
    started = perf_counter()
    program = mapped.lower(context.reference, operands).compile()
    result["map_compile_seconds"] = perf_counter() - started
    result["map_executable_bytes"] = (
        program.runtime_executable().size_of_generated_code_in_bytes()
    )
    centre = program(context.reference, operands)
    jax.block_until_ready(centre)
    result["centre_residual"] = float(
        jnp.max(jnp.abs(centre.moment_residual / context.scale))
    )
    result["centre_support"] = int(centre.graph_cells)
    result["centre_class"] = int(centre.boundary_class)
    result["centre_booking_valid"] = bool(centre.booking_valid)

    coordinates = np.vstack(
        (
            context.machine.node,
            context.machine.wall_node,
            context.machine.sample_coordinates,
        )
    )
    analytic = jnp.asarray(
        certificate._exact_state("diverted-single-null", context.exact, coordinates)
    )
    exterior, _ = _exterior(context.cache, context.carrier_key)
    operator = fixture.forward_operator(context.source, context.machine)
    flux = exterior + _image(operator, centre.moments)
    reading = context.reading(context.reference)
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    result["centre_map_flux_error"] = float(jnp.max(jnp.abs(flux - analytic))) / span
    result["centre_map_moment_error"] = float(
        jnp.max(jnp.abs(centre.moments - context.reference))
        / jnp.max(jnp.abs(context.reference))
    )
    save(args.output, result)
    print(
        f"MAP cells={count} error={result['centre_map_flux_error']:.12g} "
        f"support={result['centre_support']} "
        f"compile={result['map_compile_seconds']:.3f}",
        flush=True,
    )
    if args.past_map_guard:
        result["map_guard_exceeded"] = True
        result["diagnostic"] = "solved past the flux guard"
    else:
        require_map([result])
    save(args.output, result)
    if args.map_only:
        return

    state = (context.reference / context.scale).reshape(-1)
    size = state.size
    lu = jnp.eye(size, dtype=state.dtype)
    pivots = jnp.arange(size, dtype=jnp.int32)
    compiled_step = jax.jit(
        lambda point, carrier, unit, matrix, piv, rebuild: advance(
            point, carrier, unit, matrix, piv, rebuild, profile, args.component
        )
    )
    started = perf_counter()
    correction = compiled_step.lower(
        state, operands, context.scale, lu, pivots, True
    ).compile()
    result["correction_compile_seconds"] = perf_counter() - started
    result["correction_executable_bytes"] = (
        correction.runtime_executable().size_of_generated_code_in_bytes()
    )
    save(args.output, result)
    print(
        f"CORRECTION_COMPILED cells={count} "
        f"seconds={result['correction_compile_seconds']:.3f} "
        f"bytes={result['correction_executable_bytes']}",
        flush=True,
    )
    for index in range(args.max_steps):
        started = perf_counter()
        state, residual, lu, pivots, factor_scale = correction(
            state, operands, context.scale, lu, pivots, True
        )
        jax.block_until_ready(residual)
        row = {
            "correction": index + 1,
            "residual": float(residual),
            "factor_scale": float(factor_scale),
            "wall_seconds": perf_counter() - started,
        }
        result["history"].append(row)
        result["terminal_state"] = {
            "moments": np.asarray(state.reshape((-1, 3)) * context.scale).tolist()
        }
        result["converged"] = row["residual"] <= 1e-12
        save(args.output, result)
        print(f"CORRECTION {json.dumps(row)}", flush=True)
        if result["converged"] or row["factor_scale"] == 0:
            break
    print(
        f"COLD cells={count} residual={result['history'][-1]['residual']:.12g} "
        f"converged={result['converged']} "
        f"guard_exceeded={bool(args.past_map_guard)}",
        flush=True,
    )
    if not result["converged"]:
        raise ValueError(
            f"cold solve residual {result['history'][-1]['residual']} exceeds 1e-12"
        )


if __name__ == "__main__":
    main()
