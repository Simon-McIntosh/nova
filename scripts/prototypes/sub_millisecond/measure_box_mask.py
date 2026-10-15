"""Measure a binary box-capped flux mask against connected current support."""

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
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_cuts import fan_rule, image, time_call
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = "diverted-single-null"
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, _ = _machine(cache, carrier, 250)
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
    exact_moments = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    vertices = np.asarray(geometry.vertices)
    count = np.asarray(geometry.vertex_count)
    selected = np.asarray(reading.membership) > 0
    rows = np.arange(vertices.shape[1])[None] < count[:, None]
    edge = vertices[rows & selected[:, None]]
    lower = edge.min(axis=0)
    upper = edge.max(axis=0)
    centre = geometry.centre
    scalar = jax.vmap(field.value)(centre)
    psi_n = (scalar - reading.axis_flux) / (reading.boundary_flux - reading.axis_flux)
    box = jnp.all((centre >= lower) & (centre <= upper), axis=1)
    mask = box & (psi_n >= 0) & (psi_n <= 1)
    point, weight = fan_rule(geometry, 8)

    def moments():
        local = (point - centre[:, None, None, None]) / geometry.pitch[
            :, None, None, None, None
        ]
        x, y = local[..., 0], local[..., 1]
        coefficients = reading.field_coefficients
        flux = (
            coefficients[:, 0, None, None, None]
            + coefficients[:, 1, None, None, None] * x
            + coefficients[:, 2, None, None, None] * y
            + coefficients[:, 3, None, None, None] * x * x
            + coefficients[:, 4, None, None, None] * x * y
            + coefficients[:, 5, None, None, None] * y * y
        )
        normalized = (flux - reading.axis_flux) / (
            reading.boundary_flux - reading.axis_flux
        )
        density = profile.current_density(point[..., 0], normalized)
        weighted = jnp.where(mask[:, None, None, None], density * weight, 0)
        displacement = point - centre[:, None, None, None]
        raw = jnp.stack(
            (
                jnp.sum(weighted, axis=(1, 2, 3)),
                jnp.sum(weighted * displacement[..., 0], axis=(1, 2, 3)),
                jnp.sum(weighted * displacement[..., 1], axis=(1, 2, 3)),
            ),
            axis=1,
        )
        return raw * booked.raw_current / jnp.sum(raw[:, 0])

    mask_moments, wall_ms, compile_s = time_call(moments)
    operator = fixture.forward_operator(source, machine)
    exterior, _ = _exterior(cache, key)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    exact_error = (
        float(jnp.max(jnp.abs(exterior + image(operator, exact_moments) - analytic)))
        / span
    )
    error = (
        float(jnp.max(jnp.abs(exterior + image(operator, mask_moments) - analytic)))
        / span
    )
    reference_cells = selected
    mismatch = int(np.sum(np.asarray(mask) != reference_cells))
    result = {
        "cells": len(machine.node),
        "box": [lower.tolist(), upper.tolist()],
        "selected_cells": int(jnp.sum(mask)),
        "exact_selected_cells": int(selected.sum()),
        "cell_mismatch": mismatch,
        "wall_ms": wall_ms,
        "compile_s": compile_s,
        "map_error": error,
        "exact_map_error": exact_error,
    }
    assert mismatch > 0
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
