"""Measure the contour-tree stages and count dependent sweep visits."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import FunctionType

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import nova  # noqa: E402
from nova.equilibrium import topology  # noqa: E402
from nova.equilibrium import contour_tree as installed_contour_tree  # noqa: E402
from tests.equilibrium import test_topology_read as oracle_fixture  # noqa: E402

assert jax.config.jax_enable_x64 is True

FIELDS = (
    "node_vertex",
    "node_psi",
    "node_valid",
    "critical_type",
    "edges",
    "edge_valid",
    "overflow",
)


def load_builder(path: Path):
    if path.resolve() == Path(installed_contour_tree.__file__).resolve():
        return installed_contour_tree
    spec = importlib.util.spec_from_file_location("contour_tree_profile_target", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(kind: str, cells: int):
    _, field, wall, _, _ = oracle_fixture._analytic_inputs(kind)
    geometry = oracle_fixture._realised_hex_geometry(wall, cells)
    _, centre_flux = jax.jit(topology._sampled_flux)(field, geometry)
    mesh = geometry.contour_mesh
    vertex_flux = jnp.sum(
        centre_flux[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    sigma = topology.TopologyConvention.from_cocos(17, 1.0).sigma
    return (
        vertex_flux,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        sigma,
    ), int(geometry.centre.shape[0])


def compiled_wall(function, *args, static_argnums=()):
    start = perf_counter()
    executable = jax.jit(function, static_argnums=static_argnums).lower(*args).compile()
    compile_seconds = perf_counter() - start
    dynamic_args = tuple(
        value for index, value in enumerate(args) if index not in static_argnums
    )
    result = executable(*dynamic_args)
    jax.block_until_ready(result)
    start = perf_counter()
    result = executable(*dynamic_args)
    jax.block_until_ready(result)
    return result, perf_counter() - start, compile_seconds


def perturbed_sweep(module):
    """Compile a one-position visit-order mutation for the identity control."""

    source = inspect.getsource(module._sweep_tree)
    original = "order = jnp.lexsort((index, -values if descending else values))"
    assert source.count(original) == 1
    source = source.replace(
        original,
        "order = jnp.roll(jnp.lexsort((index, -values if descending else values)), 1)",
    )
    namespace = dict(module.__dict__)
    exec(compile(source, str(module.__file__), "exec"), namespace)
    return namespace["_sweep_tree"]


def roots_with_depth(parents, neighbours):
    """Follow the same parent links while counting pointer-doubling rounds."""

    def unresolved(state):
        roots, _ = state
        return jnp.any(roots != parents[roots])

    def advance(state):
        roots, depth = state
        return parents[roots], depth + jnp.int32(1)

    return jax.lax.while_loop(unresolved, advance, (neighbours, jnp.int32(0)))


def instrumented_sweep(module, arguments, mutation: bool = False):
    """Count parent-link rounds while preserving the sweep's output arrays."""

    counts: list[int] = []
    selected = perturbed_sweep(module) if mutation else module._sweep_tree

    def counted_roots(parents, neighbours):
        roots, depth = roots_with_depth(parents, neighbours)
        jax.debug.callback(lambda value: counts.append(int(value)), depth)
        return roots

    sweep = FunctionType(
        selected.__code__,
        {**selected.__globals__, "_neighbour_roots": counted_roots},
        selected.__name__,
        selected.__defaults__,
        selected.__closure__,
    )
    executable = jax.jit(sweep, static_argnums=(5, 6, 7)).lower(*arguments).compile()
    result = executable(*arguments[:5])
    jax.block_until_ready(result)
    return result, {
        "visits": len(counts),
        "roots_total": sum(counts),
        "roots_max": max(counts),
    }


def replay_build(module, arguments, adjacency, passes):
    """Feed counted sweep arrays through the builder's unchanged merge logic."""

    raw_builder = module.build_contour_tree.__wrapped__

    def recorded_sweep(*args):
        return passes["join" if args[5] else "split"]

    builder = FunctionType(
        raw_builder.__code__,
        {
            **raw_builder.__globals__,
            "_carrier_adjacency": lambda *args: adjacency,
            "_sweep_tree": recorded_sweep,
        },
        raw_builder.__name__,
        raw_builder.__defaults__,
        raw_builder.__closure__,
    )
    return builder(*arguments)


def assert_sweep_identity(reference, candidate):
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise AssertionError(f"sweep output {index} differs")


def assert_identity(reference, candidate):
    for name in FIELDS:
        left = np.asarray(getattr(reference, name))
        right = np.asarray(getattr(candidate, name))
        if not np.array_equal(left, right):
            difference_count = np.count_nonzero(left != right)
            raise AssertionError(f"{name} differs: {difference_count} entries")


def without_append_wall(module, sweep_args):
    original = module._append_edges

    def skip(edges, valid, overflow, sources, source_valid, target):
        return edges, valid, overflow

    try:
        module._append_edges = skip

        def no_append(*dynamic):
            return module._sweep_tree(*dynamic, *sweep_args[5:])

        _, wall, compile_seconds = compiled_wall(no_append, *sweep_args[:5])
    finally:
        module._append_edges = original
    return wall, compile_seconds


def measure(module, kind: str, cells: int):
    arguments, realised_cells = inputs(kind, cells)
    vertex_flux, vertex_valid, vertex_is_wall, edges, edge_valid, sigma = arguments
    signed = jnp.asarray(sigma, dtype=vertex_flux.dtype) * vertex_flux
    full, full_wall, full_compile = compiled_wall(module.build_contour_tree, *arguments)
    adjacency, adjacency_wall, adjacency_compile = compiled_wall(
        module._carrier_adjacency,
        edges,
        edge_valid,
        vertex_flux.size,
        static_argnums=(2,),
    )
    neighbours, neighbour_valid, _ = adjacency
    sweep_args = (
        signed,
        vertex_valid,
        vertex_is_wall,
        neighbours,
        neighbour_valid,
    )
    stages = {"full build": full_wall, "adjacency": adjacency_wall}
    compiles = {"full build": full_compile, "adjacency": adjacency_compile}
    passes = {}
    sweeps = {}
    for name, descending in (("join", True), ("split", False)):
        args = (*sweep_args, descending, vertex_flux.size, edges.shape[0])
        sweep, wall, compile_seconds = compiled_wall(
            module._sweep_tree, *args, static_argnums=(5, 6, 7)
        )
        stages[f"{name} sweep"] = wall
        compiles[f"{name} sweep"] = compile_seconds
        no_append, no_append_compile = without_append_wall(module, args)
        stages[f"{name} append estimate"] = max(0.0, wall - no_append)
        compiles[f"{name} no-append control"] = no_append_compile
        passes[name] = {"no_append_wall": no_append}
        sweeps[name] = sweep
    join, split = sweeps["join"], sweeps["split"]
    split_only = split[0] & ~join[0]
    _, insert_wall, insert_compile = compiled_wall(
        module._insert_split_nodes,
        join[0] | split[0],
        join[2],
        join[3],
        join[4] | split[4],
        split_only,
        signed,
    )
    stages["split insertion"] = insert_wall
    compiles["split insertion"] = insert_compile

    def remaining_build(tree_args, carrier_adjacency, join_pass, split_pass):
        return replay_build(
            module,
            tree_args,
            carrier_adjacency,
            {"join": join_pass, "split": split_pass},
        )

    _, rest_wall, rest_compile = compiled_wall(
        remaining_build, arguments, adjacency, join, split
    )
    stages["other build"] = rest_wall
    compiles["other build"] = rest_compile
    return {
        "case": kind,
        "requested_cells": cells,
        "realised_cells": realised_cells,
        "vertices": int(vertex_flux.size),
        "edges": int(edges.shape[0]),
        "device": str(jax.devices()[0]),
        "module": str(Path(module.__file__).resolve()),
        "nova_module": str(Path(nova.__file__).resolve()),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "receipt_overflow": bool(full.overflow),
        "receipt_nodes": int(full.node_valid.sum()),
        "receipt_edges": int(full.edge_valid.sum()),
        "stage_seconds": stages,
        "compile_seconds": compiles,
        "passes": passes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load_builder(args.module)
    row = measure(module, args.kind, args.cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print("PROFILE_ROW " + json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
