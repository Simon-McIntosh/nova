"""Time full-capacity device adjacency and dense fragment reachability."""

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
from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def adjacency(area, intervals, geometry, tolerance):
    cells, capacity = area.shape
    edge_interval = jnp.transpose(intervals, (0, 2, 1, 3, 4))
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_interval = edge_interval[peer, geometry.neighbour_edge]
    peer_interval = 1.0 - peer_interval[..., ::-1]
    overlap = jnp.minimum(
        edge_interval[:, :, :, None, :, None, 1],
        peer_interval[:, :, None, :, None, :, 1],
    ) - jnp.maximum(
        edge_interval[:, :, :, None, :, None, 0],
        peer_interval[:, :, None, :, None, :, 0],
    )
    links = (
        jnp.any(overlap > tolerance, axis=(-2, -1))
        & (geometry.neighbour >= 0)[:, :, None, None]
        & (area > 0)[:, None, :, None]
        & (area[peer] > 0)[:, :, None, :]
    )
    source = (
        jnp.arange(cells)[:, None, None, None] * capacity
        + jnp.arange(capacity)[None, None, :, None]
    )
    target = (
        peer[:, :, None, None] * capacity + jnp.arange(capacity)[None, None, None, :]
    )
    source = jnp.broadcast_to(source, links.shape)
    target = jnp.broadcast_to(target, links.shape)
    count = cells * capacity
    matrix = (
        jnp.eye(count, dtype=jnp.int32)
        .at[source.reshape(-1), target.reshape(-1)]
        .add(links.reshape(-1).astype(jnp.int32))
    )
    return matrix > 0


def reach(area, intervals, geometry, tolerance, seed):
    matrix = adjacency(area, intervals, geometry, tolerance).astype(jnp.float32)
    for _ in range(10):
        matrix = (matrix @ matrix > 0).astype(jnp.float32)
    return matrix[seed] > 0


def time_call(function, *args):
    started = perf_counter()
    program = jax.jit(function).lower(*args).compile()
    compile_seconds = perf_counter() - started
    jax.block_until_ready(program(*args))
    walls = []
    for _ in range(7):
        started = perf_counter()
        jax.block_until_ready(program(*args))
        walls.append(1000 * (perf_counter() - started))
    return program(*args), float(np.median(walls)), compile_seconds


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(f"nova={nova.__file__} module={topology.__file__}", flush=True)
    print(f"cwd={Path.cwd().resolve()}", flush=True)
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, _, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    assert bool(reading.valid) and bool(reading.qualified)
    area = reading.fragment_area
    intervals = reading.edge_interval
    exact = np.asarray(reading.fragment_selected).reshape(-1)
    active = np.asarray(area).reshape(-1) > 0
    seed = int(np.flatnonzero(exact)[0])
    def function(one, two):
        return reach(one, two, geometry, policy.edge_tolerance, seed)
    result, wall_ms, compile_seconds = time_call(function, area, intervals)
    agreement = np.asarray(result) == exact
    assert np.all(agreement[active])
    paired = jax.jit(jax.vmap(function))(
        jnp.stack((area, area)), jnp.stack((intervals, intervals))
    )
    assert np.array_equal(np.asarray(paired[0]), np.asarray(paired[1]))
    assert np.array_equal(np.asarray(paired[0]), np.asarray(result))
    receipt = {
        "kind": kind,
        "cells": len(machine.node),
        "fragment_slots": int(area.size),
        "active_fragments": int(active.sum()),
        "selected_fragments": int(exact.sum()),
        "wall_ms": wall_ms,
        "compile_s": compile_seconds,
        "agreement": float(np.mean(agreement[active])),
        "vmap_width": 2,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("RESULT " + json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
