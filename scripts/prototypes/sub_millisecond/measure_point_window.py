"""Discriminate disconnected support from out-of-range profile evaluation."""

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
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def mapped(operator, moments):
    triple = CellCurrentMoments(moments[:, 0], moments[:, 1], moments[:, 2])
    return operator.current_moment_image(operator.coupling_current_moments(triple))


def run(cache: Path, output: Path):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = "diverted-single-null"
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, _ = _machine(cache, carrier, 2000)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs("diverted")
    field = certificate_field(analytic_field, "diverted")
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda flux: current.book(
            flux, reading, geometry, profile, current.Quadrature(sigma=convention.sigma)
        )
    )(samples)
    assert bool(booked.valid)
    exact_moments = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    centre_flux = jnp.einsum("nij,nj->ni", geometry.fit_inverse, samples)[:, 0]
    psi_n = (centre_flux - reading.boundary_flux) / (
        reading.axis_flux - reading.boundary_flux
    )
    # psi_n is zero at the boundary and one at the axis.
    level_inside = convention.sigma * (centre_flux - reading.boundary_flux) > 0
    window = level_inside & (psi_n <= 1)
    connected = reading.membership > 0
    density = profile.current_density(
        geometry.centre[:, 0],
        (centre_flux - reading.axis_flux) / (reading.boundary_flux - reading.axis_flux),
    )
    raw = geometry.full_area * density
    operator = fixture.forward_operator(source, machine)
    exterior, exterior_key = _exterior(cache, key)
    assert exterior is not None
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0
    variants = {
        "unbounded": level_inside,
        "windowed": window,
        "connected_unbounded": level_inside & connected,
        "connected_windowed": window & connected,
    }
    errors = {}
    current_fractions = {}
    for label, mask in variants.items():
        values = jnp.where(mask, raw, 0)
        triple = jnp.stack(
            (values, jnp.zeros_like(values), jnp.zeros_like(values)), axis=-1
        )
        triple *= booked.raw_current / jnp.sum(values)
        errors[label] = (
            float(jnp.max(jnp.abs(exterior + mapped(operator, triple) - analytic)))
            / span
        )
        current_fractions[label] = float(jnp.sum(values)) / float(
            jnp.sum(jnp.where(level_inside, raw, 0))
        )
    private = level_inside & ~connected
    private_window = private & window
    private_outside = private & ~window
    private_np = np.asarray(private)
    p = np.asarray(geometry.centre)[private_np]
    pn = np.asarray(psi_n)[private_np]
    pd = np.asarray(density)[private_np]
    result = {
        "cells": int(len(machine.node)),
        "exterior_key": exterior_key,
        "span": span,
        "exact_error": float(
            jnp.max(jnp.abs(exterior + mapped(operator, exact_moments) - analytic))
        )
        / span,
        "error": errors,
        "raw_current_relative_to_unbounded": current_fractions,
        "selected": {key: int(jnp.sum(mask)) for key, mask in variants.items()},
        "private_count": int(jnp.sum(private)),
        "private_windowed_count": int(jnp.sum(private_window)),
        "private_outside_window_count": int(jnp.sum(private_outside)),
        "private_psi_n_range": [float(pn.min()), float(pn.max())],
        "private_density_range": [float(pd.min()), float(pd.max())],
        "private_positions": p.tolist(),
        "private_psi_n": pn.tolist(),
        "private_density": pd.tolist(),
        "private_current_fraction": float(jnp.sum(jnp.where(private, raw, 0)))
        / float(jnp.sum(jnp.where(level_inside, raw, 0))),
        "private_windowed_current_fraction": float(
            jnp.sum(jnp.where(private_window, raw, 0))
        )
        / float(jnp.sum(jnp.where(level_inside, raw, 0))),
    }
    assert result["private_count"] > 0 and result["exact_error"] < 0.01
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
