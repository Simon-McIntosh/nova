"""Measure the dense interior map and its analytic-state booking error."""

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
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.lean_map.quadrature import (
    BookingStatus,
    cut_moments,
    interior_moments,
    triangle_rule,
)
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _warm_wall(function, operand, repeats=5):
    compiled = jax.jit(function).lower(operand).compile()
    jax.block_until_ready(compiled(operand))
    walls = []
    for _ in range(repeats):
        started = perf_counter()
        jax.block_until_ready(compiled(operand))
        walls.append(perf_counter() - started)
    return float(np.median(walls))


def _read_bandwidth():
    sample = jnp.arange(32 * 1024 * 1024, dtype=jnp.float64)
    wall = _warm_wall(lambda values: jnp.sum(values), sample)
    return sample.nbytes / wall


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
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    psi = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    if not (bool(reading.valid) and bool(reading.qualified)):
        result = {
            "kind": kind,
            "cells": int(len(machine.node)),
            "topology_valid": bool(reading.valid),
            "topology_qualified": bool(reading.qualified),
            "reason": int(reading.reason),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT " + json.dumps(result), flush=True)
        return
    profile = fixture.analytic_profile(source)
    exact_booking = jax.jit(
        lambda sample: current.book(
            sample,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(psi)
    assert bool(exact_booking.valid)
    exact_moments = jnp.stack(
        (
            exact_booking.cell_current,
            exact_booking.radial_moment,
            exact_booking.vertical_moment,
        ),
        axis=-1,
    )
    rule = triangle_rule(geometry)
    values, matrix_receipt = build_values(machine, rule, field, exact_moments)
    gauss_flux = values.apply(exact_moments).reshape(rule.points.shape[:-1])
    full = reading.membership >= 1 - 1e-14
    cut = (reading.membership > 0) & ~full
    count = int(jnp.sum(cut))
    capacity = int(np.ceil(1.2 * count)) + 16
    interior = interior_moments(
        rule, gauss_flux, profile, reading.axis_flux, reading.boundary_flux, full
    )
    cut_booking = jax.jit(
        lambda sample: cut_moments(
            sample, reading, geometry, profile, convention.sigma, capacity
        )
    )(psi)
    assert int(cut_booking.status) == int(BookingStatus.VALID)
    booked = interior + cut_booking.moments
    booked *= exact_booking.raw_current / jnp.sum(booked[:, 0])
    operator = fixture.forward_operator(source, machine)
    exterior, exterior_key = _exterior(cache, key)
    if exterior is None:
        raise FileNotFoundError("the exact-support exterior is missing")
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0 and float(jnp.ptp(analytic)) > 0
    error = (
        float(jnp.max(jnp.abs(exterior + _image(operator, booked) - analytic))) / span
    )
    exact_error = (
        float(jnp.max(jnp.abs(exterior + _image(operator, exact_moments) - analytic)))
        / span
    )
    apply_wall = _warm_wall(lambda moments: values.apply(moments), booked)
    read_bandwidth = _read_bandwidth()
    result = {
        "kind": kind,
        "requested_cells": cells,
        "realised_cells": int(len(machine.node)),
        "builder": builder,
        "exterior_key": exterior_key,
        "cut_count": count,
        "cut_capacity": capacity,
        "map_error": error,
        "exact_error": exact_error,
        "map_wall_ms": 1000 * apply_wall,
        "read_bandwidth_gbps": read_bandwidth / 1e9,
        "memory_floor_ms": 1000 * matrix_receipt["matrix_bytes"] / read_bandwidth,
        **matrix_receipt,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


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
