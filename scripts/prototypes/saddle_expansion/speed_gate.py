"""Measure the saddle-anchored map against the axis-facing map on one device.

Both arms are lowered with the carrier, operands and states as runtime
arguments, so no carrier-scaled array is captured as an HLO constant. The
saddle arm is checked against a reference configuration that reproduces the
first build's integration (every cell a candidate, 128-node outer rule, the
realised cut set within the first build's ceiling): the owned-cell area and
moments must agree to 1e-12, so the cost change alters nothing but cost. The
runner exits nonzero when that agreement fails, so the gate carries a
correctness guard as well as a cost one.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import re
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

REFERENCE_OUTER_DEGREE = 128
GATE_OUTER_DEGREE = 8
AGREEMENT_TOLERANCE = 1e-12


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")


def generated_size(program):
    """Generated code size, or -1 where the backend does not report it."""
    try:
        return program.runtime_executable().size_of_generated_code_in_bytes()
    except Exception:  # noqa: BLE001 - CPU backend has no runtime executable
        return -1


def dense_constant_elements(text):
    """Largest element count over the dense constants in one lowered module."""
    largest = 0
    for match in re.finditer(r"dense<([fd]?)([^>]*)>", text):
        product = 1
        for dimension in re.findall(r"(\d+)x", match.group(2)):
            product *= int(dimension)
        largest = max(largest, product)
    return largest


def arm_signature(map_fn, *, saddle):
    """One-function signature with every carrier array a runtime argument."""

    def one(
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
        target,
        axis_sign,
        coupling,
        reference,
        total,
        radius,
        candidates,
        outer_nodes,
        outer_weights,
        policy,
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
    parser.add_argument("--radius-cells", type=float, default=4.0)
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
        _,
        target,
        axis_sign,
    ) = context.operands
    policy = context.policy
    gate_nodes, gate_weights = _leggauss(GATE_OUTER_DEGREE)
    reference_nodes, reference_weights = _leggauss(REFERENCE_OUTER_DEGREE)

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
            profile,
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
            policy,
        )

    def run_arm(map_fn, saddle, active_candidates, nodes, weights, widths):
        one = arm_signature(map_fn, saddle=saddle)
        rows = {}
        for width in widths:
            batch = neighbours[:width]
            arguments = arguments_for(active_candidates, nodes, weights)
            lowered = jax.jit(
                jax.vmap(one, in_axes=(0,) + (None,) * 20), static_argnums=(20,)
            ).lower(batch, *arguments)
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
                times.append((perf_counter() - started) * 1000 / width)
            residual = np.asarray(state.moment_residual)
            reduce_axes = tuple(range(1, residual.ndim))
            rows[str(width)] = {
                "width": width,
                "per_state_wall_ms": float(np.median(times)),
                "compile_seconds": compile_seconds,
                "executable_bytes": generated_size(executable),
                "constant_elements": dense_constant_elements(lowered.as_text()),
                "finite": int(np.sum(np.all(np.isfinite(residual), axis=reduce_axes))),
                "owner_cells": int(np.sum(np.asarray(state.graph_cells) > 0)),
            }
            result["arms"][map_fn.__module__ + "." + map_fn.__name__] = rows
            save(args.output, result)
            print(
                f"SPEED arm={'saddle' if saddle else 'axis'} "
                f"{json.dumps(rows[str(width)])}",
                flush=True,
            )
        return rows

    axis_rows = run_arm(
        axis_map, False, candidates, gate_nodes, gate_weights, (1, 16, 128)
    )
    saddle_rows = run_arm(
        saddle_map, True, candidates, gate_nodes, gate_weights, (1, 16, 128)
    )
    result["arms"] = {"axis": axis_rows, "saddle": saddle_rows}

    agreement = agreement_check(
        context,
        saddle_map,
        neighbours[:4],
        candidates,
        gate_nodes,
        gate_weights,
        reference_nodes,
        reference_weights,
        jnp.arange(len(context.geometry.centre)),
        arguments_for,
    )
    result["agreement"] = agreement
    axis128 = axis_rows["128"]["per_state_wall_ms"]
    saddle128 = saddle_rows["128"]["per_state_wall_ms"]
    ratio = saddle128 / axis128 if axis128 else float("inf")
    result["saddle_over_axis_width128"] = ratio
    result["gate_pass"] = bool(
        ratio <= 1.25
        and saddle_rows["128"]["compile_seconds"] <= 60
        and saddle_rows["128"]["executable_bytes"] <= 50_000_000
        and agreement["max_abs_difference"] <= AGREEMENT_TOLERANCE
    )
    save(args.output, result)
    print(f"GATE saddle_over_axis={ratio:.4f} pass={result['gate_pass']}", flush=True)
    print(f"AGREEMENT {json.dumps(agreement)}", flush=True)
    if agreement["max_abs_difference"] > AGREEMENT_TOLERANCE:
        print("AGREEMENT FAILED", flush=True)
        raise SystemExit(3)


def agreement_check(
    context,
    map_fn,
    batch,
    gate_candidates,
    gate_nodes,
    gate_weights,
    reference_nodes,
    reference_weights,
    all_candidates,
    arguments_for,
):
    """Owned map must agree between the sized disk and the first-build rule."""
    one = arm_signature(map_fn, saddle=True)

    def evaluate(active_candidates, nodes, weights):
        arguments = arguments_for(active_candidates, nodes, weights)
        program = (
            jax.jit(jax.vmap(one, in_axes=(0,) + (None,) * 20), static_argnums=(20,))
            .lower(batch, *arguments)
            .compile()
        )
        state = program(batch, *arguments)
        jax.block_until_ready(state.moment_residual)
        return np.asarray(state.moment_residual)

    gate = evaluate(gate_candidates, gate_nodes, gate_weights)
    reference = evaluate(all_candidates, reference_nodes, reference_weights)
    difference = float(np.max(np.abs(gate - reference)))
    return {
        "states": int(batch.shape[0]),
        "gate_outer_degree": GATE_OUTER_DEGREE,
        "reference_outer_degree": REFERENCE_OUTER_DEGREE,
        "gate_capacity": int(gate_candidates.shape[0]),
        "reference_capacity": int(all_candidates.shape[0]),
        "max_abs_difference": difference,
        "tolerance": AGREEMENT_TOLERANCE,
        "passed": bool(difference <= AGREEMENT_TOLERANCE),
    }


if __name__ == "__main__":
    main()
