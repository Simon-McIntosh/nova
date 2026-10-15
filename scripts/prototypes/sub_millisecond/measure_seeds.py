"""Probe whether distributed moments give the connected read a usable cold state."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

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


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
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
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda flux: current.book(
            flux, reading, geometry, profile, current.Quadrature(sigma=convention.sigma)
        )
    )(samples)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, _ = build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, _ = build_jets(coupling, geometry, field, reference)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    window, _ = gather_window(jets, geometry, reading.membership, roots)
    seeds = jnp.asarray(roots)
    axis_sign = float(jnp.sign(field.evaluate(seeds[0]).hessian[0, 0]))
    cut_count = int(
        jnp.sum(
            (reading.membership > 0)
            & ~((reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6))
        )
    )
    capacity = int(np.ceil(1.2 * cut_count)) + 16

    def mapped(moments):
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
            booked.raw_current,
            capacity,
            axis_sign,
        )

    program = jax.jit(mapped).lower(reference).compile()
    areas = np.asarray(geometry.full_area)
    centres = np.asarray(geometry.centre)
    total = float(booked.raw_current)
    assert total != 0 and float(np.max(np.abs(reference))) > 0
    uniform = total * areas / areas.sum()
    centroid = np.average(centres, axis=0, weights=np.asarray(reference[:, 0]))
    radius = np.sqrt(float(np.sum(areas * np.asarray(reading.membership))) / np.pi)
    circle_mask = np.linalg.norm(centres - centroid, axis=1) <= radius
    assert circle_mask.sum() > 1
    circle = total * areas * circle_mask / np.sum(areas * circle_mask)

    def moments(current_values):
        return jnp.stack(
            (jnp.asarray(current_values), jnp.zeros(len(areas)), jnp.zeros(len(areas))),
            axis=-1,
        )

    arms = {
        "analytic_moments": reference,
        "zero": jnp.zeros_like(reference),
        "uniform": moments(uniform),
        "centroid_circle": moments(circle),
    }
    rows = {}
    for label, state in arms.items():
        result = program(state)
        jax.block_until_ready(result)
        rows[label] = {
            "input_current": float(jnp.sum(state[:, 0])),
            "root_status": np.asarray(result.root_status).tolist(),
            "booking_status": int(result.booking_status),
            "band_status": int(result.band_status),
            "topology_valid": bool(result.topology_valid),
            "topology_qualified": bool(result.topology_qualified),
            "output_finite": bool(jnp.all(jnp.isfinite(result.moments))),
        }
        print("ARM " + label + " " + json.dumps(rows[label]), flush=True)
    assert rows["analytic_moments"]["topology_valid"]
    assert rows["analytic_moments"]["booking_status"] == 0
    output.write_text(
        json.dumps(
            {
                "kind": kind,
                "cells": len(areas),
                "circle_radius": radius,
                "circle_cells": int(circle_mask.sum()),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
