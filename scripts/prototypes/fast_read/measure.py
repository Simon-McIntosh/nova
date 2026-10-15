"""Measure the fixed-shape read against the analytic certificate carriers."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.fast_read.kernel import read
from scripts.prototypes.fused_newton.measure import count_trace
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


class AnalyticRows(NamedTuple):
    source: object
    coefficient: jax.Array

    def evaluate(self, point):
        return self.source.evaluate(point)

    def sampled_coefficient(self, geometry):
        return self.coefficient


def wall(program, repetitions=7):
    durations = []
    for _ in range(repetitions):
        start = perf_counter()
        jax.block_until_ready(program())
        durations.append(1000 * (perf_counter() - start))
    return float(np.median(durations))


def run(kind, cache, output, traces):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, _, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic, _, _, _ = _analytic_inputs(kind)
    source = certificate_field(analytic, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reference = jax.jit(lambda: topology.read(source, geometry, convention, policy))()
    jax.block_until_ready(reference)
    positions = jnp.asarray(
        np.concatenate(
            (
                np.asarray(reference.axis)[None],
                np.asarray(reference.x_points)[np.asarray(reference.x_point_valid)],
            ),
            axis=0,
        )
    )
    jets = jax.vmap(source.evaluate)(positions)
    roots = fixed_points.FixedPointResult(
        positions,
        jets,
        jnp.ones(len(positions), bool),
        jnp.zeros(len(positions), jnp.int32),
        jnp.zeros(len(positions), jnp.int32),
        jnp.zeros(len(positions), bool),
    )
    points = geometry.sample_points.reshape(-1, 2)
    samples = jax.vmap(source.evaluate)(points).value.reshape(
        geometry.sample_points.shape[:-1]
    )
    coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, samples)
    field = AnalyticRows(source, coefficient)
    baseline = jax.jit(
        lambda: topology.read_at_stationary_points(
            field, geometry, convention, policy, roots
        )
    )
    started = perf_counter()
    baseline_program = baseline.lower().compile()
    baseline_compile = perf_counter() - started
    baseline_state = baseline_program()
    jax.block_until_ready(baseline_state)
    baseline_wall = wall(baseline_program)

    def candidate_read():
        return read(field, roots, geometry, convention, policy)

    candidate = jax.jit(candidate_read)
    started = perf_counter()
    candidate_program = candidate.lower().compile()
    compile_seconds = perf_counter() - started
    state = candidate_program()
    jax.block_until_ready(state)
    wall_ms = wall(candidate_program)
    trace_dir = traces / kind
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir)):
        jax.block_until_ready(candidate_program())
    result = {
        "kind": kind,
        "cells": len(geometry.centre),
        "reference_valid": bool(reference.valid and reference.qualified),
        "baseline_valid": bool(baseline_state.valid and baseline_state.qualified),
        "baseline_wall_ms": baseline_wall,
        "baseline_compile_seconds": baseline_compile,
        "valid": bool(state.valid and state.qualified),
        "wall_ms": wall_ms,
        "compile_seconds": compile_seconds,
        "axis_distance_pitches": float(
            jnp.linalg.norm(state.axis - reference.axis) / jnp.median(geometry.pitch)
        ),
        "reference_boundary_flux": float(reference.boundary_flux),
        "boundary_flux": float(state.boundary_flux),
        "boundary_class": int(state.boundary_class),
        "wall_candidate": float(state.wall_candidate),
        "saddle_candidate": float(state.saddle_candidate),
        "normal_cells": int(jnp.sum(state.normal_form_cells)),
        "boundary_delta": float(state.boundary_flux - reference.boundary_flux),
        "membership_max_delta": float(
            jnp.max(jnp.abs(state.membership - baseline_state.membership))
        ),
        "selected_cells": int(jnp.sum(state.selected_cells)),
        "reference_cells": int(jnp.sum(reference.membership > 0)),
        "trace": count_trace(trace_dir, "jit_candidate_read"),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output, args.trace_dir)


if __name__ == "__main__":
    main()
