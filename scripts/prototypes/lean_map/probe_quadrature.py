"""Qualify fixed interior quadrature against exact connected-current booking."""

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
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


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
    assert bool(reading.valid) and bool(reading.qualified)
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
    gauss_flux = jax.vmap(field.value)(rule.points.reshape(-1, 2)).reshape(
        rule.points.shape[:-1]
    )
    full = reading.membership >= 1 - 1e-14
    cut = (reading.membership > 0) & ~full
    cut_count = int(jnp.sum(cut))
    capacity = int(np.ceil(1.2 * cut_count)) + 16
    interior = jax.jit(
        lambda flux: interior_moments(
            rule, flux, profile, reading.axis_flux, reading.boundary_flux, full
        )
    )(gauss_flux)
    compact = jax.jit(
        lambda sample: cut_moments(
            sample, reading, geometry, profile, convention.sigma, capacity
        )
    )
    started = perf_counter()
    cut_booking = compact(psi)
    jax.block_until_ready(cut_booking)
    cold_cut_seconds = perf_counter() - started
    assert int(cut_booking.status) == int(BookingStatus.VALID)
    started = perf_counter()
    cut_booking = compact(psi)
    jax.block_until_ready(cut_booking)
    warm_cut_seconds = perf_counter() - started
    refusal = jax.jit(
        lambda sample: cut_moments(
            sample, reading, geometry, profile, convention.sigma, 1
        )
    )(psi)
    assert int(refusal.status) == int(BookingStatus.NEEDS_CUT_ROWS)
    raw = interior + cut_booking.moments
    scale = exact_booking.raw_current / jnp.sum(raw[:, 0])
    with_moments = raw * scale
    one_current = with_moments.at[:, 1:].set(cut_booking.moments[:, 1:] * scale)
    operator = fixture.forward_operator(source, machine)
    exterior, exterior_key = _exterior(cache, key)
    if exterior is None:
        raise FileNotFoundError("exact-support exterior missing in read-only cache")
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0 and float(jnp.ptp(analytic)) > 0

    def error(moments):
        return (
            float(jnp.max(jnp.abs(exterior + _image(operator, moments) - analytic)))
            / span
        )

    result = {
        "kind": kind,
        "cells": int(len(machine.node)),
        "builder": builder,
        "exterior_key": exterior_key,
        "full_cells": int(jnp.sum(full)),
        "cut_count": cut_count,
        "cut_capacity": capacity,
        "cut_margin": capacity - cut_count,
        "over_capacity_status": int(refusal.status),
        "cut_current_relative_difference": float(
            jnp.max(
                jnp.abs(
                    jnp.where(cut, cut_booking.moments[:, 0] - exact_moments[:, 0], 0)
                )
            )
            / jnp.max(jnp.abs(exact_moments[:, 0]))
        ),
        "exact_error": error(exact_moments),
        "gauss_first_moments_error": error(with_moments),
        "gauss_one_current_error": error(one_current),
        "cold_cut_seconds": cold_cut_seconds,
        "warm_cut_seconds": warm_cut_seconds,
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
