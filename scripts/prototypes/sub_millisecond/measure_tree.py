"""Time the existing fixed-capacity contour tree on certificate carriers."""

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
from nova.equilibrium import contour_tree, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(f"nova={nova.__file__} module={contour_tree.__file__}", flush=True)
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
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    _, centre_flux = topology._sampled_flux(field, geometry)
    mesh = geometry.contour_mesh
    vertex_flux = jnp.sum(
        centre_flux[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    arguments = (
        vertex_flux,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        convention.sigma,
    )
    started = perf_counter()
    program = jax.jit(contour_tree.build_contour_tree).lower(*arguments).compile()
    compile_seconds = perf_counter() - started
    reference = program(*arguments)
    jax.block_until_ready(reference)
    assert not bool(reference.overflow)
    walls = []
    for _ in range(7):
        started = perf_counter()
        jax.block_until_ready(program(*arguments))
        walls.append(1000 * (perf_counter() - started))
    counts = np.bincount(
        np.asarray(reference.critical_type)[np.asarray(reference.node_valid)],
        minlength=4,
    )
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "carrier_vertices": int(jnp.sum(mesh.vertex_valid)),
        "carrier_edges": int(jnp.sum(mesh.edge_valid)),
        "wall_ms": float(np.median(walls)),
        "compile_s": compile_seconds,
        "node_counts": counts.tolist(),
        "overflow": bool(reference.overflow),
        "selected_support_cells": int(jnp.sum(reading.membership > 0)),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
