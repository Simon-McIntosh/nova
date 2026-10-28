"""Split a production forward solve's wall into stages, cold and warm.

Each program a forward solve compiles is built from the published diverted
certificate carrier and timed in its own right: the trace, lowering and XLA
compile walls of a cold compile, the first execute, and the steady-state wall
over repeated calls. The topology read, the booking and the map are the three
programs the Newton loop calls per step, and the linear solve is timed on the
map's own Jacobian, so the per-step split is measured on the same carrier the
solve runs at.

Nothing here changes product code. The measured objects are the production call
sites (`nova.equilibrium.topology.read`, `nova.equilibrium.current.book`, the
Newton map) on a carrier the certificate publishes.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import glob
import gzip
import json
import logging
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_map_production import program as mapped_program
from scripts.prototypes.fast_map_production.solver import (
    CachedExterior,
    certified_machine,
    generated_bytes,
    operand_block,
    operand_tangent,
)
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.support import certificate_fixtures
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def wall(program, arguments, repetitions):
    jax.block_until_ready(program(*arguments))
    elapsed = []
    for _ in range(repetitions):
        started = perf_counter()
        jax.block_until_ready(program(*arguments))
        elapsed.append(perf_counter() - started)
    return float(np.median(elapsed)), float(np.percentile(elapsed, 95))


class CompileCounter(logging.Handler):
    """Count the lines JAX emits for each program it compiles."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("Compiling "):
            self.lines.append(message)


def describe(argument):
    """Return a JSON-safe shape and dtype record for one program argument."""
    record = []

    def leaf(value):
        if hasattr(value, "shape"):
            record.append(
                {"shape": list(np.shape(value)), "dtype": str(np.asarray(value).dtype)}
            )
        else:
            record.append({"type": type(value).__name__})
        return value

    jax.tree_util.tree_map(leaf, argument)
    return record


def cold_stages(name, function, arguments, revision):
    """Time trace, lowering, XLA compile, first execute and warm wall."""
    jax.clear_caches()
    started = perf_counter()
    traced = jax.make_jaxpr(function)(*arguments)
    trace_seconds = perf_counter() - started

    started = perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lower_seconds = perf_counter() - started

    counter = CompileCounter()
    logger = logging.getLogger("jax")
    logger.addHandler(counter)
    started = perf_counter()
    compiled = lowered.compile()
    compile_seconds = perf_counter() - started
    logger.removeHandler(counter)

    started = perf_counter()
    result = compiled(*arguments)
    jax.block_until_ready(result)
    first_execute_seconds = perf_counter() - started

    warm_seconds, warm_p95 = wall(compiled, arguments, 50)
    return {
        "program": name,
        "revision": revision,
        "trace_seconds": trace_seconds,
        "lower_seconds": lower_seconds,
        "xla_compile_seconds": compile_seconds,
        "first_execute_seconds": first_execute_seconds,
        "warm_median_seconds": warm_seconds,
        "warm_p95_seconds": warm_p95,
        "compilation_count": len(counter.lines),
        "compilation_lines": counter.lines,
        "executable_bytes": generated_bytes(compiled),
        "jaxpr_equations": len(traced.jaxpr.eqns),
        "argument_leafs": [describe(a) for a in arguments],
    }



def profile_program(label, program, arguments, directory, repetitions=20):
    """Trace repeated warm calls and table their device kernels by device time."""
    directory.mkdir(parents=True, exist_ok=True)
    prepared = program(*arguments)
    jax.block_until_ready(prepared)
    started = perf_counter()
    with jax.profiler.trace(str(directory), create_perfetto_link=False):
        for _ in range(repetitions):
            jax.block_until_ready(program(*arguments))
    total = perf_counter() - started
    record = {
        "label": label,
        "repetitions": repetitions,
        "wall_ms": 1000 * total,
        "per_call_ms": 1000 * total / repetitions,
        "directory": str(directory),
        "kernels": [],
    }
    files = sorted(Path(directory).glob("plugins/profile/*/*.trace.json.gz"))
    if not files:
        record["trace_absent"] = True
        return record
    payload = json.loads(gzip.decompress(files[0].read_bytes()))
    events = payload.get("traceEvents", [])
    kernel_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") in ("kernel", "Kernel", "gpu_user_annotation")
        and event.get("dur")
    ]
    totals = {}
    for event in kernel_events:
        totals[event.get("args", {}).get("name", event.get("name", "?"))] = (
            totals.get(event.get("args", {}).get("name", event.get("name", "?")), 0.0)
            + event["dur"]
    )
    device_us = sum(totals.values()) / repetitions
    record["device_us_per_call"] = device_us
    record["host_gap_ms_per_call"] = 1000 * total / repetitions - device_us / 1000
    record["trace_file"] = str(files[0])
    record["kernels"] = [
        {"name": name, "ms_per_call": value / repetitions / 1000}
        for name, value in sorted(totals.items(), key=lambda item: -item[1])[:12]
    ]
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd().resolve()} "
        f"nova={nova.__file__} module={sys.argv[0]} "
        f"command={json.dumps(sys.argv)} backend={jax.default_backend()}",
        flush=True,
    )

    kind, cells = "diverted", args.cells
    name = "diverted-single-null"
    started = perf_counter()
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    requested = -500 if cells == 550 else -cells
    machine = certificate._case_machine(
        name, carrier, exact, requested, clip_mode="exact"
    )
    carrier_seconds = perf_counter() - started
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    profile = fixture.analytic_profile(source)

    def read_function(field, geometry):
        return topology.read(field, geometry, convention, policy)

    reading = jax.jit(read_function)(field, geometry)
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )

    def book_function(psi, reading, geometry):
        return current.book(
            psi, reading, geometry, profile,
            current.Quadrature(sigma=convention.sigma),
        )

    booked = jax.jit(book_function)(samples, reading, geometry)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, values_receipt = build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jets_receipt = build_jets(coupling, geometry, field, reference)

    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    cached_rows = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(len(jets.nodes), dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )
    generators = np.asarray(jets.mesh.generators)
    margin = float(
        np.max(
            np.min(np.linalg.norm(generators[:, None] - roots[None], axis=-1), axis=1)
        )
        + 2 * np.max(geometry.pitch)
    )
    fixed_points.maintain_rows(
        geometry,
        coupling,
        CachedExterior(jets.nodes, jets.exterior),
        jnp.asarray(roots),
        margin,
        cache=cached_rows,
        capacity=len(jets.nodes),
    )
    window = cached_rows
    axis_sign = float(jnp.sign(field.evaluate(jnp.asarray(roots[0])).hessian[0, 0]))
    seeds = jnp.asarray(roots)
    operands = (
        values, jets, window, rule, fit, design, geometry, seeds, convention,
        booked.raw_current,
    )

    def mapped(moments, *carried):
        (
            values_, jets_, window_, rule_, fit_, design_, geometry_, seeds_,
            convention_, target,
        ) = carried
        return mapped_program.map_moments(
            moments, values_, jets_, window_, rule_, fit_, design_, geometry_,
            profile, seeds_, convention_, policy, target, axis_sign,
        )

    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )

    def equation(scaled, *carried):
        state = mapped(scaled.reshape(reference.shape) * scale, *carried)
        return (state.moment_residual / scale).reshape(-1)

    results = {
        "revision": revision,
        "nova_module": nova.__file__,
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jnp.zeros(1).devices()],
        "cells": int(len(machine.node)),
        "carrier_load_seconds": carrier_seconds,
        "carrier_cache": machine.cache,
        "values": values_receipt,
        "jets": jets_receipt,
        "window": {
            "rows": int(jnp.sum(window.lookup >= 0)),
            "capacity": len(window.rows),
        },
        "programs": [],
    }

    started = perf_counter()
    try:
        reloaded = certificate_fixtures.cached_carrier(
            carrier, requested, certificate._diverted_wall(exact)
        )
        results["operand_load_seconds"] = perf_counter() - started
        results["operand_load_cache"] = reloaded.cache
    except FileNotFoundError:
        results["operand_load_seconds"] = None
        results["operand_load_cache"] = {"absent": "certificate content store has no carrier at this revision"}

    def persist():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")

    results["programs"].append(
        cold_stages("topology_read", read_function, (field, geometry), revision)
    )
    results["programs"].append(
        cold_stages("booking", book_function, (samples, reading, geometry), revision)
    )
    persist()
    results["programs"].append(
        cold_stages("map", mapped, (reference, *operands), revision)
    )
    persist()

    map_program = jax.jit(mapped).lower(reference, *operands).compile()
    read_program = jax.jit(read_function).lower(field, geometry).compile()
    book_program = jax.jit(book_function).lower(samples, reading, geometry).compile()

    steps = {}
    read_ms, read_p95 = wall(lambda _: read_program(field, geometry), (jnp.zeros(1),), 50)
    book_ms, book_p95 = wall(
        lambda psi: book_program(psi, reading, geometry), (samples,), 50
    )
    number = len(roots)
    solved = fixed_points.FixedPointResult(
        position=jnp.asarray(roots),
        jet=jax.vmap(field.evaluate)(jnp.asarray(roots)),
        valid=jnp.ones((number,), dtype=bool),
        status=jnp.zeros((number,), dtype=jnp.int32),
        iterations=jnp.zeros((number,), dtype=jnp.int32),
        retried=jnp.zeros((number,), dtype=bool),
    )

    def read_without_tree(field, geometry, solved):
        return topology.read_at_stationary_points(
            field, geometry, convention, policy, solved
        )

    no_tree_program = (
        jax.jit(read_without_tree).lower(field, geometry, solved).compile()
    )
    no_tree_ms, no_tree_p95 = wall(
        no_tree_program, (field, geometry, solved), 50
    )
    map_ms, map_p95 = wall(map_program, (reference, *operands), 50)

    flat = reference.reshape(-1)
    residual = equation(flat, *operands)
    tangent = operand_tangent(equation, flat)
    identity = jnp.eye(residual.size, dtype=residual.dtype)
    in_axes = (0,) + (None,) * len(operands)

    def columns_image(columns, *carried):
        return jax.vmap(
            lambda c: operand_block(tangent, 1)(c, *carried), in_axes=in_axes
        )(columns, *carried)

    jacobian_program = jax.jit(columns_image).lower(identity, *operands).compile()
    matrix = jacobian_program(identity, *operands)
    jax.block_until_ready(matrix)
    solve_ms, solve_p95 = wall(
        lambda: jnp.linalg.solve(identity - matrix.T, residual), (), 20
    )
    jacobian_ms, jacobian_p95 = wall(
        lambda: jacobian_program(identity, *operands), (), 20
    )

    steps["1"] = {
        "batch": 1,
        "topology_read_ms": 1000 * read_ms,
        "topology_read_p95_ms": 1000 * read_p95,
        "booking_ms": 1000 * book_ms,
        "booking_p95_ms": 1000 * book_p95,
        "map_ms": 1000 * map_ms,
        "map_p95_ms": 1000 * map_p95,
        "linear_solve_ms": 1000 * solve_ms,
        "linear_solve_p95_ms": 1000 * solve_p95,
        "read_without_tree_ms": 1000 * no_tree_ms,
        "read_without_tree_p95_ms": 1000 * no_tree_p95,
        "jacobian_block_ms": 1000 * jacobian_ms,
        "jacobian_block_p95_ms": 1000 * jacobian_p95,
        "residual_size": int(residual.size),
        "cells": int(len(machine.node)),
    }

    batch = args.batch
    states = reference[None] + jnp.asarray(
        1e-7 * np.arange(batch), dtype=reference.dtype
    ).reshape((-1,) + (1,) * reference.ndim)

    def batched(moments, *carried):
        return jax.vmap(lambda m: mapped(m, *carried))(moments)

    started = perf_counter()
    batch_program = jax.jit(batched).lower(states, *operands).compile()
    batch_compile_seconds = perf_counter() - started
    started = perf_counter()
    jax.block_until_ready(batch_program(states, *operands))
    batch_first_seconds = perf_counter() - started
    batch_ms, batch_p95 = wall(batch_program, (states, *operands), 50)
    steps[str(batch)] = {
        "batch": batch,
        "compile_seconds": batch_compile_seconds,
        "first_execute_seconds": batch_first_seconds,
        "map_batched_ms": 1000 * batch_ms,
        "map_batched_p95_ms": 1000 * batch_p95,
        "map_per_solve_ms": 1000 * batch_ms / batch,
    }
    results["steps"] = steps
    profiles = []
    for label, program, arguments_ in (
        ("map", map_program, (reference, *operands)),
        ("topology_read", read_program, (field, geometry)),
    ):
        profiles.append(
            profile_program(
                label, program, arguments_, args.output.parent / f"profile-{label}"
            )
        )
    results["profiles"] = profiles

    results["host_peak_rss_gb"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(
        "SUMMARY "
        + json.dumps(
            {
                "programs": [
                    (p["program"], round(p["xla_compile_seconds"], 3),
                     p["compilation_count"], p["executable_bytes"])
                    for p in results["programs"]
                ],
                "step_batch1": steps["1"],
                "step_batchN": steps[str(batch)],
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()