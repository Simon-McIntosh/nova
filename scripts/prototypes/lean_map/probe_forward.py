"""Exercise the live fixed-shape map at a certificate's analytic state."""

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
from scripts.prototypes.lean_map.forward import apply
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _write(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("CHECKPOINT " + json.dumps(result, default=str), flush=True)


def run(kind, cells, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} probe={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, builder = _machine(cache, carrier, cells)
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "builder": builder,
        "stage": "loaded",
    }
    _write(output, result)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    psi = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    result.update(
        topology_valid=bool(reading.valid),
        topology_qualified=bool(reading.qualified),
        topology_reason=int(reading.reason),
    )
    _write(output, result)
    if not (result["topology_valid"] and result["topology_qualified"]):
        return
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda sample: current.book(
            sample,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(psi)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit_inverse, sample_design = quadratic_fit(rule, geometry)
    values, values_receipt = build_values(machine, rule, field, reference)
    result.update(stage="values-built", values=values_receipt)
    _write(output, result)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jets_receipt = build_jets(coupling, geometry, field, reference)
    result.update(stage="jets-built", jets=jets_receipt)
    _write(output, result)
    admitted = np.asarray(reading.x_point_valid)
    roots = np.vstack(
        (np.asarray(reading.axis)[None], np.asarray(reading.x_points)[admitted])
    )
    window, window_receipt = gather_window(jets, geometry, reading.membership, roots)
    result.update(stage="window-built", window=window_receipt)
    _write(output, result)
    cut_count = int(
        jnp.sum((reading.membership > 0) & (reading.membership < 1 - 1e-14))
    )
    capacity = int(np.ceil(1.2 * cut_count)) + 16
    axis_jet = field.evaluate(jnp.asarray(roots[0]))
    axis_sign = float(jnp.sign(axis_jet.hessian[0, 0]))
    seeds = jnp.asarray(roots)

    def mapped(
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
    ):
        return apply(
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
            seeds,
            convention,
            policy,
            target,
            capacity,
            axis_sign,
        )

    started = perf_counter()
    compiled = (
        jax.jit(mapped)
        .lower(
            reference,
            values,
            jets,
            window,
            rule,
            fit_inverse,
            sample_design,
            geometry,
            seeds,
            convention,
            booked.raw_current,
        )
        .compile()
    )
    compile_seconds = perf_counter() - started
    arguments = (
        reference,
        values,
        jets,
        window,
        rule,
        fit_inverse,
        sample_design,
        geometry,
        seeds,
        convention,
        booked.raw_current,
    )
    started = perf_counter()
    state = compiled(*arguments)
    jax.block_until_ready(state)
    first_wall = perf_counter() - started
    walls = []
    for _ in range(5):
        started = perf_counter()
        state = compiled(*arguments)
        jax.block_until_ready(state)
        walls.append(perf_counter() - started)
    root_shift = np.linalg.norm(np.asarray(state.roots) - roots, axis=1)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = float(jnp.max(jnp.abs(reference[:, 0])))
    result.update(
        stage="map-evaluated",
        compile_seconds=compile_seconds,
        first_wall_ms=1000 * first_wall,
        warm_wall_ms=1000 * float(np.median(walls)),
        map_residual_relative_sup=float(jnp.max(jnp.abs(state.moments - reference)))
        / scale,
        root_shift_pitches=(root_shift / pitch).tolist(),
        root_status=np.asarray(state.root_status).tolist(),
        booking_status=int(state.booking_status),
        band_status=int(state.band_status),
        cut_count=int(state.cut_count),
        topology_valid=bool(state.topology_valid),
        topology_qualified=bool(state.topology_qualified),
    )
    _write(output, result)
    print("RESULT " + json.dumps(result, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cells, args.cache, args.output)


if __name__ == "__main__":
    main()
