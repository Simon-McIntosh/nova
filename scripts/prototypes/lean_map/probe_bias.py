"""Compare centroid membership with the connected certificate support."""

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
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine, _point_booking
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _image(operator, values):
    moments = CellCurrentMoments(values[:, 0], values[:, 1], values[:, 2])
    return operator.current_moment_image(operator.coupling_current_moments(moments))


def run(kind, cells, cache, output):
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} probe={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
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
    target = booked.raw_current
    exact_moments = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    old = _point_booking(
        psi,
        geometry,
        profile,
        reading.boundary_flux,
        reading.axis_flux,
        convention.sigma,
        target,
        False,
    )
    full = reading.membership >= 1 - 1e-14
    cut = (reading.membership > 0) & ~full
    centre_flux = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)[:, 0]
    old_inside = convention.sigma * (centre_flux - reading.boundary_flux) > 0
    density = profile.current_density(
        geometry.centre[:, 0],
        (centre_flux - reading.axis_flux) / (reading.boundary_flux - reading.axis_flux),
    )
    whole = jnp.where(full, geometry.full_area * density, 0.0)
    corrected_raw = jnp.stack(
        (whole, jnp.zeros_like(whole), jnp.zeros_like(whole)), axis=-1
    ) + jnp.where(cut[:, None], exact_moments, 0.0)
    corrected = corrected_raw * target / jnp.sum(corrected_raw[:, 0])
    operator = fixture.forward_operator(source, machine)
    exterior, exterior_key = _exterior(cache, key)
    if exterior is None:
        raise FileNotFoundError("the read-only cache needs its exact-support exterior")
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0 and float(jnp.ptp(analytic)) > 0
    baseline = exterior + _image(operator, exact_moments)
    old_mapped = exterior + _image(operator, old)
    corrected_mapped = exterior + _image(operator, corrected)
    result = {
        "kind": kind,
        "cells": int(len(machine.node)),
        "builder": builder,
        "machine_key": key,
        "exterior_key": exterior_key,
        "exact_error": float(jnp.max(jnp.abs(baseline - analytic))) / span,
        "old_error": float(jnp.max(jnp.abs(old_mapped - analytic))) / span,
        "corrected_error": float(jnp.max(jnp.abs(corrected_mapped - analytic))) / span,
        "old_selected": int(jnp.sum(old_inside)),
        "outside_selected": int(jnp.sum(old_inside & (reading.membership == 0))),
        "cut_selected": int(jnp.sum(old_inside & cut)),
        "full_cells": int(jnp.sum(full)),
        "cut_cells": int(jnp.sum(cut)),
        "excluded_current_fraction": float(
            jnp.sum(
                jnp.where(
                    old_inside & (reading.membership == 0),
                    geometry.full_area * density,
                    0.0,
                )
            )
            / jnp.sum(jnp.where(old_inside, geometry.full_area * density, 0.0))
        ),
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
