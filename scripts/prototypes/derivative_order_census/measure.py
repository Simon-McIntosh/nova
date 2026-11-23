"""CPU lowering and local derivative comparisons, without executing a read."""

# ruff: noqa: E402
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import re
import resource
import runpy
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from scripts.prototypes.derivative_order_census.variants import install


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print("RECEIPT " + json.dumps(value, allow_nan=False), flush=True)


def fixture():
    namespace = runpy.run_path("tests/equilibrium/test_topology_read.py")
    oracle, total, wall, axis, saddle = namespace["_analytic_inputs"]("diverted")
    geometry = namespace["_realised_hex_geometry"](wall, 24)
    field = namespace["_kernel_backed_field"]("diverted", oracle, total, geometry)
    rng = np.random.default_rng(719)
    vertices = np.asarray(geometry.vertices).reshape(-1, 2)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    generic = rng.uniform(lower, upper, (200, 2))
    heights = np.unique(np.asarray(field.coupling.edge)[:, (1, 3), :])
    # Keep generic heights explicitly outside the sampled corner strip.
    for index in range(len(generic)):
        while np.min(np.abs(heights - generic[index, 1])) <= 2e-7:
            generic[index, 1] = rng.uniform(lower[1], upper[1])
    corners = generic[:50].copy()
    corners[:, 1] = rng.choice(heights, 50) + rng.uniform(-9e-8, 9e-8, 50)
    targets = np.concatenate((generic, corners))
    policy = topology._read_policy(geometry, topology.TopologyPolicy())
    info = {
        "requested_cells": 24,
        "cells": int(geometry.centre.shape[0]),
        "edges": int(field.coupling.edge.shape[0]),
        "wall_vertices": int(geometry.wall_points.shape[0]),
        "sample_points": int(np.prod(geometry.sample_points.shape[:-1])),
        "contour_sample_points": int(
            np.prod(geometry.contour_sample_points.shape[:-1])
        ),
        "null_capacity": int(policy.null_capacity),
        "polish_iterations": int(policy.polish_iterations),
        "axis": np.asarray(axis).tolist(),
        "saddle": np.asarray(saddle).tolist(),
        "target_sha256": __import__("hashlib").sha256(targets.tobytes()).hexdigest(),
    }
    assert info["cells"] > 0 and np.isfinite(targets).all()
    assert np.all(np.min(np.abs(corners[:, 1, None] - heights), axis=1) <= 1e-7)
    return field, geometry, jnp.asarray(targets), info


def derivative(order):
    def value(point, coupling, moments):
        return coupling.value_gradient(point, moments)[0]

    def gradient(point, coupling, moments):
        return coupling.value_gradient(point, moments)[1]

    def hessian(point, coupling, moments):
        return coupling.evaluate(point, moments).hessian

    if order < 3:
        return (value, gradient, hessian)[order]
    function = hessian
    for _ in range(order - 2):
        function = jax.jacfwd(function, argnums=0)
    return function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-instruction-limit", type=int, default=140000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--mode", choices=("derivative", "curvature", "batch"), default="derivative"
    )
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd().resolve()} command={sys.argv}",
        flush=True,
    )
    print(f"module={topology.__file__} cwd={Path.cwd().resolve()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "cpu"
    field, geometry, targets, info = fixture()
    calls = install(args.variant)
    function = derivative(args.order)
    operands = (targets[0], field.coupling, field.moments)
    if args.mode == "curvature":

        def function(point, coupling, moments):
            local = topology.TotalField(moments, coupling, field.exterior)
            return topology._curvature_derivatives(
                local, point, jnp.median(geometry.pitch)
            )
    elif args.mode == "batch":
        function = jax.vmap(function, in_axes=(0, None, None))
        operands = (targets[: args.batch_size], field.coupling, field.moments)
    row = {
        "variant": args.variant,
        "order": args.order,
        "mode": args.mode,
        "revision": revision,
        "module": topology.__file__,
        "cwd": str(Path.cwd().resolve()),
        "fixture": info,
        "backend": str(jax.devices()[0]),
    }
    start = time.monotonic()
    lowered = jax.jit(function).lower(*operands)
    row["lower_seconds"] = time.monotonic() - start
    text = lowered.as_text(debug_info=True)
    row["stablehlo_bytes"] = len(text.encode())
    hlo = lowered.compiler_ir("hlo").as_hlo_module().to_string()
    instructions = [
        line for line in hlo.splitlines() if re.match(r"\s*(?:ROOT )?%\S+ =", line)
    ]
    attributed = [line for line in instructions if "packed_instance_" in line]
    markers = sorted(set(re.findall(r"packed_instance_\d+", "\n".join(attributed))))
    row.update(
        hlo_instructions=len(instructions),
        packed_kernel_ops=len(attributed),
        packed_kernel_instances=len(markers),
        traced_kernel_calls=len(calls),
        packed_markers=markers,
        temp_bytes=None,
        compile_seconds=None,
    )
    assert attributed and markers, (
        "source attribution did not see the known-present packed kernel"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".hlo").write_text(hlo)
    row["phase"] = "lowered"
    write(args.output, row)
    if len(instructions) > args.compile_instruction_limit:
        row["phase"] = "compile-skipped-size"
        row["compile_skip_reason"] = (
            f"HLO instructions exceed {args.compile_instruction_limit}"
        )
        write(args.output, row)
        return
    start = time.monotonic()
    compiled = lowered.compile()
    row["compile_seconds"] = time.monotonic() - start
    analysis = compiled.memory_analysis()
    row["temp_bytes"] = analysis.temp_size_in_bytes
    row["argument_bytes"] = analysis.argument_size_in_bytes
    row["output_bytes"] = analysis.output_size_in_bytes
    row["phase"] = "compiled"
    write(args.output, row)
    if args.mode == "derivative":
        # Reuse the one-point executable serially; a vmap would change this measurement.
        values = np.stack(
            [
                np.asarray(compiled(point, field.coupling, field.moments))
                for point in targets
            ]
        )
        row["finite_values"] = int(np.isfinite(values).sum())
        row["total_values"] = int(values.size)
        np.savez(
            args.output.with_suffix(".npz"), targets=np.asarray(targets), values=values
        )
    row["phase"] = "complete"
    row["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(args.output, row)


if __name__ == "__main__":
    main()
