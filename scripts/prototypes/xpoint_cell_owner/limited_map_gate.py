"""Measure the limited carrier through both owner selections on one map."""

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
from scripts.prototypes.fine_grid_convergence import context
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments
from scripts.prototypes.xpoint_cell_owner.runtime_solve import without_profile


def _limited_context(cells, cache):
    """Build the limited analogue of the existing certificate map context."""
    carrier, source, exact = certificate._case(
        "weak-rotation-reactor-static", clip_mode="exact"
    )
    machine, key, _ = context._machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = context._analytic_inputs("limited")
    field = context.certificate_field(analytic_field, "limited")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda psi: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(samples)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = context.triangle_rule(geometry)
    fit, design = context.quadratic_fit(rule, geometry)
    values, _ = context.build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, _ = context.build_jets(coupling, geometry, field, reference)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    window = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(len(jets.nodes), dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )
    seeds = jnp.asarray(roots)
    axis_sign = float(jnp.sign(field.evaluate(seeds[0]).hessian[0, 0]))
    operands = (
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        profile,
        seeds,
        convention,
        policy,
        booked.raw_current,
        axis_sign,
    )
    return context.Context(
        geometry,
        reference,
        jnp.ones(3),
        operands,
        window,
        values,
        jets,
        fit,
        rule,
        profile,
        seeds,
        convention,
        policy,
        axis_sign,
        machine,
        source,
        exact,
        key,
        cache,
    )


def run_row(cells, cache):
    built = _limited_context(cells, cache)
    print(f"BUILT requested={cells} realised={len(built.reference)}", flush=True)
    reading = built.reading(built.reference)
    owner_cells = np.flatnonzero(
        np.asarray(reading.booking_cells & (reading.membership > 0))
    ).tolist()
    operands = without_profile(built.operands)

    def mapped(component):
        return jax.jit(
            lambda moments: map_moments(
                moments,
                *operands[:7],
                built.profile,
                *operands[7:],
                component=component,
            )
        )(built.reference)

    lobe = mapped("lobe")
    halfplane = mapped("halfplane")
    assert bool(lobe.valid) and bool(halfplane.valid)
    exterior, _ = _exterior(cache, built.carrier_key)
    if exterior is None:
        raise FileNotFoundError("the limited carrier has no read-only exterior")
    operator = fixture.forward_operator(built.source, built.machine)
    coordinates = np.vstack(
        (built.machine.node, built.machine.wall_node, built.machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state(
            "weak-rotation-reactor-static", built.exact, coordinates
        )
    )
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0 and float(jnp.ptp(analytic)) > 0

    def error(moments):
        return (
            float(jnp.max(jnp.abs(exterior + _image(operator, moments) - analytic)))
            / span
        )

    result = {
        "requested": cells,
        "realised": len(built.reference),
        "owner_cells": owner_cells,
        "lobe_error": error(lobe.moments),
        "halfplane_error": error(halfplane.moments),
        "reference_error": error(built.reference),
        "max_lobe_halfplane_moment_delta": float(
            jnp.max(jnp.abs(lobe.moments - halfplane.moments))
        ),
    }
    print("SUMMARY " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    receipt = {"rows": [], "errors": []}

    def persist():
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    persist()
    for cells in args.sizes:
        try:
            receipt["rows"].append(run_row(cells, args.cache))
        except Exception as error:  # noqa: BLE001
            receipt["errors"].append({"requested": cells, "error": repr(error)})
            print(f"FAILED requested={cells} error={error!r}", flush=True)
        persist()
    if receipt["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
