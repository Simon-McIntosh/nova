"""Measure the saddle-anchored map against the axis-facing map on one device."""

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
from scripts.prototypes.saddle_expansion.expansion import map_moments as saddle_map
from scripts.prototypes.support.geometry import certificate_field
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments as axis_map
from tests.equilibrium.test_topology_read import _analytic_inputs


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")


def without_profile(operands):
    return (*operands[:7], *operands[8:])


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

    operands = without_profile(context.operands)
    values, jets, window, rule, fit, design, geometry = operands[:7]
    seeds, convention, policy, target, axis_sign = operands[7:]

    def axis_one(moments):
        return axis_map(
            moments,
            values,
            jets,
            window,
            rule,
            fit,
            design,
            geometry,
            context.profile,
            seeds,
            convention,
            policy,
            target,
            axis_sign,
        )

    def saddle_one(moments):
        return saddle_map(
            moments,
            values,
            jets,
            window,
            rule,
            fit,
            design,
            geometry,
            context.profile,
            seeds,
            convention,
            policy,
            target,
            axis_sign,
            coupling=coupling,
            reference=reference,
            total=total,
            radius=radius,
        )

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
        "arms": {},
    }
    save(args.output, result)
    for arm, one in (("axis", axis_one), ("saddle", saddle_one)):
        rows = []
        for width in (1, 16, 128):
            batch = neighbours[:width]
            program = jax.jit(jax.vmap(one)).lower(batch).compile()
            executable = program.runtime_executable().size_of_generated_code_in_bytes()
            started = perf_counter()
            state = program(batch)
            jax.block_until_ready(state.moment_residual)
            compile_seconds = perf_counter() - started
            times = []
            for _ in range(5):
                started = perf_counter()
                state = program(batch)
                jax.block_until_ready(state.moment_residual)
                times.append((perf_counter() - started) * 1000 / width)
            residual = np.asarray(state.moment_residual)
            reduce_axes = tuple(range(1, residual.ndim))
            row = {
                "width": width,
                "per_state_wall_ms": float(np.median(times)),
                "compile_seconds": compile_seconds,
                "executable_bytes": executable,
                "finite": int(np.sum(np.all(np.isfinite(residual), axis=reduce_axes))),
                "owner_cells": int(np.sum(np.asarray(state.graph_cells) > 0)),
            }
            rows.append(row)
            result["arms"].setdefault(arm, {})[str(width)] = row
            save(args.output, result)
            print(f"SPEED arm={arm} {json.dumps(row)}", flush=True)
    axis128 = result["arms"]["axis"]["128"]["per_state_wall_ms"]
    saddle128 = result["arms"]["saddle"]["128"]["per_state_wall_ms"]
    ratio = saddle128 / axis128 if axis128 else float("inf")
    result["saddle_over_axis_width128"] = ratio
    result["gate_pass"] = bool(
        ratio <= 1.25
        and result["arms"]["saddle"]["128"]["compile_seconds"] <= 60
        and result["arms"]["saddle"]["128"]["executable_bytes"] <= 50_000_000
    )
    save(args.output, result)
    print(f"GATE saddle_over_axis={ratio:.4f} pass={result['gate_pass']}", flush=True)


if __name__ == "__main__":
    main()
