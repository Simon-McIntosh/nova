"""Compare fixed subcell cut moments with exact connected booking."""

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
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def image(operator, moments):
    values = CellCurrentMoments(moments[:, 0], moments[:, 1], moments[:, 2])
    return operator.current_moment_image(operator.coupling_current_moments(values))


def fan_rule(geometry, width):
    centre = np.asarray(geometry.centre)
    vertices = np.asarray(geometry.vertices)
    count = np.asarray(geometry.vertex_count)
    slots = np.arange(vertices.shape[1])
    next_slot = np.where(slots[None] + 1 < count[:, None], slots[None] + 1, 0)
    following = np.take_along_axis(vertices, next_slot[..., None], axis=1)
    first = vertices - centre[:, None]
    second = following - centre[:, None]
    oriented = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    triangle_area = np.where(slots[None] < count[:, None], 0.5 * oriented, 0.0)
    assert np.min(triangle_area) >= -1e-12
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(width) + 0.5) / width
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points = (
        centre[:, None, None, None]
        + uu[None, None, ..., None] * first[:, :, None, None]
        + (uu * vv)[None, None, ..., None] * (second - first)[:, :, None, None]
    )
    weights = triangle_area[:, :, None, None] * 2 * uu[None, None] / width**2
    weights = np.broadcast_to(weights, points.shape[:-1])
    return jnp.asarray(points), jnp.asarray(weights)


def subcell_moments(points, weights, geometry, reading, profile, sigma):
    local = (points - geometry.centre[:, None, None, None]) / geometry.pitch[
        :, None, None, None, None
    ]
    x, y = local[..., 0], local[..., 1]
    coefficient = reading.field_coefficients
    value = (
        coefficient[:, 0, None, None, None]
        + coefficient[:, 1, None, None, None] * x
        + coefficient[:, 2, None, None, None] * y
        + coefficient[:, 3, None, None, None] * x * x
        + coefficient[:, 4, None, None, None] * x * y
        + coefficient[:, 5, None, None, None] * y * y
    )
    span = reading.boundary_flux - reading.axis_flux
    normalized = (value - reading.axis_flux) / span
    density = profile.current_density(points[..., 0], normalized)
    inside = sigma * (value - reading.boundary_flux) > 0
    selected = (reading.membership > 0)[:, None, None, None]
    current_weight = jnp.where(inside & selected, weights * density, 0.0)
    displacement = points - geometry.centre[:, None, None, None]
    return jnp.stack(
        (
            jnp.sum(current_weight, axis=(1, 2, 3)),
            jnp.sum(current_weight * displacement[..., 0], axis=(1, 2, 3)),
            jnp.sum(current_weight * displacement[..., 1], axis=(1, 2, 3)),
        ),
        axis=-1,
    )


def time_call(function, *args):
    started = perf_counter()
    program = jax.jit(function).lower(*args).compile()
    compile_seconds = perf_counter() - started
    jax.block_until_ready(program(*args))
    walls = []
    for _ in range(7):
        started = perf_counter()
        jax.block_until_ready(program(*args))
        walls.append(1000 * (perf_counter() - started))
    return program(*args), float(np.median(walls)), compile_seconds


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
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
    full = reading.membership >= 1 - 1e-14
    cut = (reading.membership > 0) & ~full
    operator = fixture.forward_operator(source, machine)
    exterior, _ = _exterior(cache, key)
    assert exterior is not None
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0 and float(jnp.ptp(analytic)) > 0
    baseline = exterior + image(operator, exact_moments)
    exact_error = float(jnp.max(jnp.abs(baseline - analytic))) / span
    rows = {}
    for width in (4, 8):
        points, weights = fan_rule(geometry, width)

        def approximate(orientation):
            raw = subcell_moments(
                points, weights, geometry, reading, profile, orientation
            )
            mixed = jnp.where(cut[:, None], raw, exact_moments)
            return mixed * booked.raw_current / jnp.sum(mixed[:, 0])

        moments, wall_ms, compile_seconds = time_call(approximate, convention.sigma)
        wrong, _, _ = time_call(approximate, -convention.sigma)
        candidate = exterior + image(operator, moments)
        wrong_candidate = exterior + image(operator, wrong)
        cut_difference = jnp.max(jnp.abs((moments - exact_moments)[cut]))
        error = float(jnp.max(jnp.abs(candidate - analytic))) / span
        wrong_error = float(jnp.max(jnp.abs(wrong_candidate - analytic))) / span
        rows[str(width)] = {
            "wall_ms": wall_ms,
            "compile_s": compile_seconds,
            "map_error": error,
            "wrong_orientation_map_error": wrong_error,
            "cut_moment_max_difference": float(cut_difference),
        }
        print("ARM " + str(width) + " " + json.dumps(rows[str(width)]), flush=True)
    assert (
        max(row["wrong_orientation_map_error"] for row in rows.values()) > exact_error
    )
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "cut_cells": int(jnp.sum(cut)),
        "exact_map_error": exact_error,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
