"""Measure the saddle-anchored map against the axis-facing map on one device.

Both arms are lowered with the carrier, the batched state and every operands
array as runtime arguments, so no carrier- or state-scaled array is captured as
an HLO constant; the density profile and the topology policy are closed over
because they carry callables and scalars rather than arrays. The saddle arm is
checked against a reference configuration that reproduces the first build's
integration, and the runner exits nonzero when that check fails, so the gate
carries a correctness guard as well as a cost one.

One width is measured per invocation, so the device memory high-water mark
reported belongs to that configuration alone and is not a running maximum
carried over from a larger one. The caller sweeps widths downward and stops at
the first that fits.
"""

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
from nova.equilibrium import topology
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.saddle_expansion.expansion import _leggauss, disk_candidates
from scripts.prototypes.saddle_expansion.expansion import map_moments as saddle_map
from scripts.prototypes.support.geometry import certificate_field
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments as axis_map
from tests.equilibrium.test_topology_read import _analytic_inputs
from tests.test_equilibrium_solve import _hlo_constants

REFERENCE_OUTER_DEGREE = 128
GATE_OUTER_DEGREE = 8
AGREEMENT_TOLERANCE = 1e-12
DEGREE_LADDER = (8, 16, 32, 64, 128)
EXIT_OOM = 2


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")


def generated_size(program):
    """Generated code size, or -1 where the backend does not report it."""
    try:
        return program.runtime_executable().size_of_generated_code_in_bytes()
    except Exception:  # noqa: BLE001 - the CPU backend has no runtime executable
        return -1


def peak_bytes():
    """Device memory high-water mark, or -1 where the device does not report it."""
    try:
        return int(jax.devices()[0].memory_stats().get("peak_bytes_in_use", -1))
    except Exception:  # noqa: BLE001 - no device statistics on every backend
        return -1


def is_oom(error):
    return "RESOURCE_EXHAUSTED" in str(error) or "Out of memory" in str(error)


def arm_signature(map_fn, profile, policy, *, saddle):
    """One function with every carrier and state array a runtime argument."""

    def one(
        moments,
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        seeds,
        convention,
        target,
        axis_sign,
        coupling,
        reference,
        total,
        radius,
        candidates,
        outer_nodes,
        outer_weights,
    ):
        extras = (
            {
                "coupling": coupling,
                "reference": reference,
                "total": total,
                "radius": radius,
                "candidates": candidates,
                "outer_nodes": outer_nodes,
                "outer_weights": outer_weights,
            }
            if saddle
            else {}
        )
        return map_fn(
            moments,
            values,
            jets,
            window,
            rule,
            fit,
            design,
            geometry,
            profile,
            seeds,
            convention,
            policy,
            target,
            axis_sign,
            **extras,
        )

    return one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=2098)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-cells", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    backend = jax.default_backend()
    if not args.allow_cpu:
        assert backend == "gpu", f"measured on a GPU lane, backend={backend}"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()} "
        f"backend={backend}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    count = len(context.reference)
    _, analytic_field, _, _, _ = _analytic_inputs("diverted")
    total = certificate_field(analytic_field, "diverted")
    coupling = topology.BiotMomentCoupling.from_polygons(context.machine.cell_polygons)
    reference = context.reference
    median_pitch = float(jnp.median(context.geometry.pitch))
    radius = args.radius_cells * median_pitch

    (
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        profile,
        seeds,
        convention,
        policy,
        target,
        axis_sign,
    ) = context.operands
    gate_nodes, gate_weights = _leggauss(GATE_OUTER_DEGREE)

    reading = context.reading(reference)
    candidates = disk_candidates(geometry, reading.saddle_form.position, radius)
    gate_capacity = int(candidates.shape[0])

    size = reference.size
    coordinates = np.arange(size, dtype=float)
    offsets = np.arange(128, dtype=float)[:, None]
    perturbations = np.sin(coordinates[None] * 0.17 + offsets * 0.31)
    perturbations /= np.linalg.norm(perturbations, axis=1)[:, None]
    neighbours = (
        np.asarray(reference).reshape(-1)[None] + 1e-4 * perturbations
    ).reshape(128, *reference.shape)
    neighbours = jnp.asarray(neighbours)

    result = {
        "cells": count,
        "median_pitch_m": median_pitch,
        "radius_m": radius,
        "radius_cells": args.radius_cells,
        "gate_capacity": gate_capacity,
        "gate_outer_degree": GATE_OUTER_DEGREE,
        "reference_outer_degree": REFERENCE_OUTER_DEGREE,
        "width": args.width,
        "arms": {},
    }
    save(args.output, result)

    def arguments_for(active_candidates, nodes, weights):
        return (
            values,
            jets,
            window,
            rule,
            fit,
            design,
            geometry,
            seeds,
            convention,
            target,
            axis_sign,
            coupling,
            reference,
            total,
            radius,
            active_candidates,
            nodes,
            weights,
        )

    if args.agreement:
        run_agreement(
            saddle_map,
            profile,
            policy,
            neighbours,
            candidates,
            gate_nodes,
            gate_weights,
            jnp.arange(len(context.geometry.centre)),
            arguments_for,
            result,
            args.output,
        )
        return

    def compile_and_run(map_fn, saddle):
        one = arm_signature(map_fn, profile, policy, saddle=saddle)
        batch = neighbours[: args.width]
        arguments = arguments_for(candidates, gate_nodes, gate_weights)
        lowered = jax.jit(jax.vmap(one, in_axes=(0,) + (None,) * 18)).lower(
            batch, *arguments
        )
        executable = lowered.compile()
        started = perf_counter()
        state = executable(batch, *arguments)
        jax.block_until_ready(state.moment_residual)
        compile_seconds = perf_counter() - started
        times = []
        for _ in range(5):
            started = perf_counter()
            state = executable(batch, *arguments)
            jax.block_until_ready(state.moment_residual)
            times.append((perf_counter() - started) * 1000 / args.width)
        residual = np.asarray(state.moment_residual)
        reduce_axes = tuple(range(1, residual.ndim))
        constants = _hlo_constants(lowered)
        row = {
            "width": args.width,
            "per_state_wall_ms": float(np.median(times)),
            "compile_seconds": compile_seconds,
            "executable_bytes": generated_size(executable),
            "constant_shapes": [list(shape) for shape in constants],
            "constant_elements": int(sum(np.prod(shape) for shape in constants)),
            "largest_constant_elements": int(
                max((int(np.prod(shape)) for shape in constants), default=0)
            ),
            "peak_bytes_in_use": peak_bytes(),
            "finite": int(np.sum(np.all(np.isfinite(residual), axis=reduce_axes))),
            "owner_cells": int(np.sum(np.asarray(state.graph_cells) > 0)),
        }
        row["per_state_peak_bytes"] = (
            row["peak_bytes_in_use"] // args.width
            if row["peak_bytes_in_use"] > 0
            else -1
        )
        row["per_member_us"] = (
            row["per_state_wall_ms"] * 1000.0 / gate_capacity if gate_capacity else -1.0
        )
        return row

    failed = False
    for label, map_fn, saddle in (
        ("axis", axis_map, False),
        ("saddle", saddle_map, True),
    ):
        try:
            row = compile_and_run(map_fn, saddle)
        except jax.errors.JaxRuntimeError as error:
            if not is_oom(error):
                raise
            result["arms"][label] = {
                "oom": True,
                "width": args.width,
                "error": str(error),
            }
            save(args.output, result)
            print(f"SPEED arm={label} width={args.width} OOM {error}", flush=True)
            failed = True
            break
        result["arms"][label] = row
        save(args.output, result)
        print(f"SPEED arm={label} {json.dumps(row)}", flush=True)

    if not failed:
        axis_row, saddle_row = result["arms"]["axis"], result["arms"]["saddle"]
        ratio = saddle_row["per_state_wall_ms"] / axis_row["per_state_wall_ms"]
        result["saddle_over_axis"] = ratio
        result["bar_pass"] = bool(
            saddle_row["per_state_wall_ms"] <= 0.90
            and saddle_row["compile_seconds"] <= 60
            and saddle_row["executable_bytes"] <= 50_000_000
        )
        save(args.output, result)
        print(
            f"BAR width={args.width} "
            f"saddle_per_state_ms={saddle_row['per_state_wall_ms']:.6f} "
            f"ratio={ratio:.4f} pass={result['bar_pass']} "
            f"capacity={gate_capacity}",
            flush=True,
        )
    save(args.output, result)
    raise SystemExit(EXIT_OOM if failed else 0)


def run_agreement(
    map_fn,
    profile,
    policy,
    batch,
    gate_candidates,
    gate_nodes,
    gate_weights,
    all_candidates,
    arguments_for,
    result,
    output,
):
    """Ladder the outer degree against the 128-node all-cell reference."""
    one = arm_signature(map_fn, profile, policy, saddle=True)
    states = batch[:4]

    def evaluate(active_candidates, nodes, weights):
        arguments = arguments_for(active_candidates, nodes, weights)
        program = (
            jax.jit(jax.vmap(one, in_axes=(0,) + (None,) * 18))
            .lower(states, *arguments)
            .compile()
        )
        state = program(states, *arguments)
        jax.block_until_ready(state.moment_residual)
        return np.asarray(state.moment_residual)

    reference_nodes, reference_weights = _leggauss(REFERENCE_OUTER_DEGREE)
    reference = evaluate(all_candidates, reference_nodes, reference_weights)
    ladder = {}
    for degree in DEGREE_LADDER:
        nodes, weights = _leggauss(degree)
        candidate = evaluate(gate_candidates, nodes, weights)
        difference = float(np.max(np.abs(candidate - reference)))
        ladder[str(degree)] = {
            "max_abs_difference": difference,
            "passed": bool(difference <= AGREEMENT_TOLERANCE),
        }
        print(
            f"AGREEMENT degree={degree} max_abs_difference={difference:.12e}",
            flush=True,
        )
    same_list = evaluate(gate_candidates, reference_nodes, reference_weights)
    result["agreement"] = {
        "states": int(states.shape[0]),
        "gate_capacity": int(gate_candidates.shape[0]),
        "reference_capacity": int(all_candidates.shape[0]),
        "ladder": ladder,
        "candidate_set_effect": float(np.max(np.abs(same_list - reference))),
        "tolerance": AGREEMENT_TOLERANCE,
        "min_over_ladder": min(row["max_abs_difference"] for row in ladder.values()),
        "max_outer_degree": max(int(degree) for degree in ladder),
    }
    result["agreement"]["passed"] = bool(
        result["agreement"]["min_over_ladder"] <= AGREEMENT_TOLERANCE
    )
    save(output, result)
    print(f"AGREEMENT {json.dumps(result['agreement'])}", flush=True)
    if not result["agreement"]["passed"]:
        print("AGREEMENT FAILED", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
