"""Measure complete and moving Bell-jet access on a larger carrier."""

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
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.read_robustness.window import full_window, mobile_window
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def warm_wall(program, *args):
    program(*args)
    jax.block_until_ready(program(*args))
    walls = []
    for _ in range(7):
        start = perf_counter()
        jax.block_until_ready(program(*args))
        walls.append(1000 * (perf_counter() - start))
    return float(np.median(walls))


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 500)
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
    sample_flux = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    booked = jax.jit(
        lambda samples: current.book(
            samples,
            reading,
            geometry,
            fixture.analytic_profile(source),
            current.Quadrature(sigma=convention.sigma),
        )
    )(sample_flux)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jet_receipt = build_jets(coupling, geometry, field, reference)
    roots = jnp.vstack((reading.axis[None], reading.x_points[reading.x_point_valid]))
    fixed, fixed_receipt = gather_window(
        jets, geometry, reading.membership, np.asarray(roots)
    )
    full = full_window(jets)
    assert bool(jnp.all(full.lookup >= 0))
    missing = np.asarray(fixed.lookup) < 0
    sample = np.asarray(jets.sample_index)
    crossing = np.flatnonzero(np.any(missing[sample], axis=1))
    assert len(crossing) > 0
    cell = int(crossing[0])
    newly_cut = reading.membership.at[cell].set(0.5)
    capacity = fixed_receipt["capacity"]
    gather = jax.jit(
        lambda membership: mobile_window(jets, geometry, membership, roots, capacity)
    )
    old = gather(reading.membership)
    moved = gather(newly_cut)
    jax.block_until_ready((old, moved))
    node = int(sample[cell][np.flatnonzero(missing[sample[cell]])[0]])
    assert int(fixed.lookup[node]) < 0
    assert int(moved.cache.lookup[node]) >= 0
    assert not bool(moved.overflow)
    contract = jax.jit(lambda moments: jets.gather(moments))
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "jet_rows": jet_receipt["nodes"],
        "jet_bytes": jet_receipt["matrix_bytes"],
        "fixed_required": fixed_receipt["required_rows"],
        "capacity": capacity,
        "moved_required": int(moved.required),
        "new_cut_cell": cell,
        "newly_required_node": node,
        "fixed_lookup": int(fixed.lookup[node]),
        "mobile_lookup": int(moved.cache.lookup[node]),
        "full_contraction_ms": warm_wall(contract, reference),
        "mobile_gather_ms": warm_wall(gather, newly_cut),
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
