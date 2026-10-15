"""Probe wall-domain level crossings, connected cells and fractional area."""

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
from scripts.prototypes.sub_millisecond.measure_bell_cut import (
    bell_moments,
    bell_polygon,
    image,
    time_call,
)
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def edge_connections(geometry, reading, sigma):
    vertices = geometry.vertices
    count = geometry.vertex_count
    cells, edges, _ = vertices.shape
    slot = jnp.arange(edges)
    following = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=1)
    centre = geometry.centre
    pitch = geometry.pitch
    coefficient = reading.field_coefficients

    def signed(points):
        local = (points - centre[:, None]) / pitch[:, None, None]
        x, y = local[..., 0], local[..., 1]
        return sigma * (
            coefficient[:, 0, None]
            + coefficient[:, 1, None] * x
            + coefficient[:, 2, None] * y
            + coefficient[:, 3, None] * x * x
            + coefficient[:, 4, None] * x * y
            + coefficient[:, 5, None] * y * y
            - reading.boundary_flux
        )

    present = (
        (signed(vertices) > 0) | (signed((vertices + end) / 2) > 0) | (signed(end) > 0)
    ) & (slot[None] < count[:, None])
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_edge = jnp.maximum(geometry.neighbour_edge, 0)
    mutual = present & present[peer, peer_edge] & (geometry.neighbour >= 0)
    row = jnp.broadcast_to(jnp.arange(cells)[:, None], (cells, edges))
    matrix = jnp.eye(cells, dtype=jnp.int32).at[row, peer].add(mutual.astype(jnp.int32))
    return matrix > 0


def connected_cells(geometry, reading, sigma, seed):
    matrix = edge_connections(geometry, reading, sigma).astype(jnp.float32)
    for _ in range(9):
        matrix = (matrix @ matrix > 0).astype(jnp.float32)
    return matrix[seed] > 0


def inside_own_cell(points, geometry):
    vertices = geometry.vertices
    count = geometry.vertex_count
    edges = vertices.shape[1]
    slot = jnp.arange(edges)
    following = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=1)
    edge = end - vertices
    delta = points[:, None] - vertices[:, :, None]
    cross = edge[:, :, None, 0] * delta[..., 1] - edge[:, :, None, 1] * delta[..., 0]
    return jnp.all(
        jnp.where(slot[None, :, None] < count[:, None, None], cross >= -1e-10, True),
        axis=1,
    )


def polygon_area(vertices, count):
    slot = jnp.arange(vertices.shape[1])
    following = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=1)
    cross = vertices[..., 0] * end[..., 1] - vertices[..., 1] * end[..., 0]
    return 0.5 * jnp.sum(jnp.where(slot[None] < count[:, None], cross, 0), axis=1)


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
    seed = int(jnp.argmin(jnp.sum((geometry.centre - reading.axis) ** 2, axis=1)))
    selected, graph_ms, graph_compile = time_call(
        lambda: connected_cells(geometry, reading, convention.sigma, seed)
    )
    exact_cells = reading.membership > 0
    disagreement = int(jnp.sum(selected != exact_cells))
    polygon, count, valid = bell_polygon(geometry, reading, convention.sigma, 4)
    inside = inside_own_cell(polygon, geometry)
    live = jnp.arange(polygon.shape[1])[None] < count[:, None]
    wall_escape = int(jnp.sum(valid & jnp.any(live & ~inside, axis=1)))
    fraction = polygon_area(polygon, count) / geometry.full_area
    ordinary = exact_cells & ~reading.normal_form_cells & valid
    area_max = float(jnp.max(jnp.abs((fraction - reading.membership)[ordinary])))

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

    def candidate():
        raw, candidate_valid = bell_moments(
            geometry, reading, profile, convention.sigma, 4
        )
        use = selected & candidate_valid & ~reading.normal_form_cells
        fallback = selected & ~use
        mixed = jnp.where(
            use[:, None], raw, jnp.where(fallback[:, None], exact_moments, 0.0)
        )
        return mixed * booked.raw_current / jnp.sum(mixed[:, 0])

    moments, booking_ms, booking_compile = time_call(candidate)
    operator = fixture.forward_operator(source, machine)
    exterior, _ = _exterior(cache, key)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    map_error = (
        float(jnp.max(jnp.abs(exterior + image(operator, moments) - analytic))) / span
    )
    exact_error = (
        float(jnp.max(jnp.abs(exterior + image(operator, exact_moments) - analytic)))
        / span
    )

    neighbour = np.asarray(geometry.neighbour)
    wall_cell = np.flatnonzero(np.any(neighbour < 0, axis=1))[0]
    wall_edge = np.flatnonzero(neighbour[wall_cell] < 0)[0]
    cell_vertices = np.asarray(
        geometry.vertices[wall_cell, : int(geometry.vertex_count[wall_cell])]
    )
    start = cell_vertices[wall_edge]
    end = cell_vertices[(wall_edge + 1) % len(cell_vertices)]
    midpoint = (start + end) / 2
    vector = end - start
    outward = np.array((vector[1], -vector[0])) / np.linalg.norm(vector)
    injected = np.asarray(geometry.centre).copy()
    injected[wall_cell] = midpoint + 0.135 * outward
    rejected = not bool(
        np.asarray(inside_own_cell(jnp.asarray(injected[:, None]), geometry))[
            wall_cell, 0
        ]
    )
    assert rejected, "the wall-domain instrument must reject an exterior crossing"
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "graph_wall_ms": graph_ms,
        "graph_compile_s": graph_compile,
        "cell_disagreement": disagreement,
        "selected_cells": int(jnp.sum(selected)),
        "exact_selected_cells": int(jnp.sum(exact_cells)),
        "ordinary_area_fraction_max_error": area_max,
        "wall_escape_cells": wall_escape,
        "synthetic_135mm_wall_pierce_rejected": rejected,
        "booking_wall_ms": booking_ms,
        "booking_compile_s": booking_compile,
        "map_error": map_error,
        "exact_map_error": exact_error,
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
