"""Measure batched field contractions, roots and full-cell quadrature."""

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
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.lean_map import forward as lean_forward
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.lean_map.quadrature import (
    interior_moments,
    quadratic_fit,
    triangle_rule,
)
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_vmap import measure
from scripts.prototypes.sub_millisecond.measure_slice import arc_cut_moments
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(cache: Path, output: Path, full_only: bool = False):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    carrier, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs("diverted")
    field = certificate_field(analytic_field, "diverted")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
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
    slots = jnp.maximum(window.lookup[jets.sample_index], 0)
    seeds = jnp.asarray(roots)
    axis_sign = float(jnp.sign(field.evaluate(seeds[0]).hessian[0, 0]))
    full = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    perturbation = reference * 1e-6

    def moments(shift):
        return reference + shift * perturbation

    def values_stage(shift):
        return values.apply(moments(shift))

    def band_stage(shift):
        return window.exterior[slots, 0] + jnp.einsum(
            "nscm,cm->ns", window.rows[slots, ..., 0], moments(shift)
        )

    def root_stage(shift):
        return fixed_points.solve(
            window,
            moments(shift),
            seeds,
            seeds,
            jnp.median(geometry.pitch),
            axis_sign,
        ).position

    gauss_flux = values.apply(reference).reshape(rule.points.shape[:-1])

    def interior_stage(shift):
        return interior_moments(
            rule,
            gauss_flux
            + shift * 1e-6 * jnp.abs(reading.boundary_flux - reading.axis_flux),
            profile,
            reading.axis_flux,
            reading.boundary_flux,
            full,
        )

    cut_count = int(
        jnp.sum(
            (reading.membership > 0)
            & ~((reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6))
        )
    )
    capacity = int(np.ceil(1.2 * cut_count)) + 16
    lean_forward.cut_moments = arc_cut_moments

    def live_map(shift):
        state = lean_forward.apply(
            moments(shift),
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
            TopologyPolicy(),
            booked.raw_current,
            capacity,
            axis_sign,
        )
        return state.moments

    rows = []
    stages = (
        (("live_bell_map", live_map),)
        if full_only
        else (
            ("values_map", values_stage),
            ("bell_jet_band", band_stage),
            ("fixed_points", root_stage),
            ("interior_gauss", interior_stage),
        )
    )
    for name, function in stages:
        rows = measure(name, function, output, rows)
    print("RESULT " + json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()
    run(args.cache, args.output, args.full_only)


if __name__ == "__main__":
    main()
