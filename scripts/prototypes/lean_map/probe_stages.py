"""Time the fixed-shape map's matvec, read and booking stages separately."""

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
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.lean_map.forward import MixedField, apply
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.lean_map.quadrature import (
    cut_moments,
    interior_moments,
    quadratic_fit,
    triangle_rule,
)
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _wall(function, *arguments):
    program = jax.jit(function).lower(*arguments).compile()
    jax.block_until_ready(program(*arguments))
    samples = []
    for _ in range(7):
        started = perf_counter()
        jax.block_until_ready(program(*arguments))
        samples.append(1000 * (perf_counter() - started))
    return float(np.median(samples))


def run(kind, cells, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} probe={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    analytic_field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(
        lambda: topology.read(analytic_field, geometry, convention, policy)
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    sample_flux = jax.vmap(analytic_field.value)(
        geometry.sample_points.reshape(-1, 2)
    ).reshape(geometry.sample_points.shape[:-1])
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda flux: current.book(
            flux, reading, geometry, profile, current.Quadrature(sigma=convention.sigma)
        )
    )(sample_flux)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, values_receipt = build_values(machine, rule, analytic_field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jets_receipt = build_jets(coupling, geometry, analytic_field, reference)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    window, window_receipt = gather_window(jets, geometry, reading.membership, roots)
    seeds = jnp.asarray(roots)
    axis_sign = float(jnp.sign(analytic_field.evaluate(seeds[0]).hessian[0, 0]))
    cut_count = int(
        jnp.sum(
            (reading.membership > 0)
            & ~((reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6))
        )
    )
    capacity = int(np.ceil(1.2 * cut_count)) + 16
    slots = window.lookup[jets.sample_index]
    safe_slots = jnp.maximum(slots, 0)
    complete = jnp.all(slots >= 0, axis=1)

    def root_stage(moments):
        return fixed_points.solve(
            window, moments, seeds, seeds, jnp.median(geometry.pitch), axis_sign
        )

    def values_stage(moments):
        return values.apply(moments).reshape(rule.points.shape[:-1])

    def band_stage(moments):
        return window.exterior[safe_slots, 0] + jnp.einsum(
            "nscm,cm->ns", window.rows[safe_slots, ..., 0], moments
        )

    def read_stage(moments, gauss_flux, band_values, positions):
        gauss_coefficient = jnp.einsum("nkq,nq->nk", fit, gauss_flux)
        band_coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
        coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
        cached = fixed_points._CachedField(
            window.mesh, window.lookup, window.rows, moments, window.exterior
        )
        field = MixedField(cached, coefficient)
        return topology.read_at_stationary_points(
            field, geometry, convention, policy, positions
        )

    def cut_stage(samples, support):
        return cut_moments(
            samples, support, geometry, profile, convention.sigma, capacity
        )

    def full_stage(moments):
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

    positions = root_stage(reference).position
    gauss_flux = values_stage(reference)
    band_values = band_stage(reference)
    support = read_stage(reference, gauss_flux, band_values, positions)
    assert bool(support.valid) and bool(support.qualified)
    samples = jnp.where(
        complete[:, None],
        band_values,
        jnp.einsum("nsk,nk->ns", design, support.field_coefficients),
    )
    cut_result = cut_stage(samples, support)
    assert int(cut_result.status) == 0
    full = (support.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    walls = {
        "values_matvec": _wall(values_stage, reference),
        "band_matvec": _wall(band_stage, reference),
        "root_solve": _wall(root_stage, reference),
        "support_read": _wall(
            read_stage, reference, gauss_flux, band_values, positions
        ),
        "interior_gauss": _wall(
            lambda flux: interior_moments(
                rule, flux, profile, support.axis_flux, support.boundary_flux, full
            ),
            gauss_flux,
        ),
        "exact_cut": _wall(cut_stage, samples, support),
        "full_map": _wall(full_stage, reference),
    }
    peak_bandwidth = 4.8e12
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "walls_ms": walls,
        "values_bytes": values_receipt["matrix_bytes"],
        "band_bytes": window_receipt["gather_bytes"],
        "values_floor_ms": 1000 * values_receipt["matrix_bytes"] / peak_bandwidth,
        "band_floor_ms": 1000 * window_receipt["gather_bytes"] / peak_bandwidth,
        "peak_bandwidth_bytes_per_second": peak_bandwidth,
        "source_nodes": jets_receipt["nodes"],
        "band_required_rows": window_receipt["required_rows"],
        "band_capacity": window_receipt["capacity"],
        "cut_count": cut_count,
        "cut_capacity": capacity,
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
