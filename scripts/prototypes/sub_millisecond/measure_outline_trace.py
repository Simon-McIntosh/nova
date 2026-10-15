"""Trace a fixed-capacity Bell contour from the admitted saddle cell."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_cuts import time_call
from scripts.prototypes.sub_millisecond.measure_wall_march import inside_own_cell
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def crossing_segments(geometry, reading, sigma):
    vertices = geometry.vertices
    count = geometry.vertex_count
    cells, edges, _ = vertices.shape
    slot = jnp.arange(edges)
    following = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=1)
    centre = geometry.centre
    pitch = geometry.pitch
    coefficient = reading.field_coefficients

    def signed(point):
        local = (point - centre[:, None]) / pitch[:, None, None]
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

    start_value = signed(vertices)
    end_value = signed(end)
    middle = signed((vertices + end) / 2)
    quadratic = 2 * (end_value + start_value - 2 * middle)
    linear = end_value - start_value - quadratic
    discriminant = jnp.maximum(linear * linear - 4 * quadratic * start_value, 0)
    safe_quadratic = jnp.where(jnp.abs(quadratic) > 1e-12, quadratic, 1)
    roots = jnp.stack(
        (
            (-linear - jnp.sqrt(discriminant)) / (2 * safe_quadratic),
            (-linear + jnp.sqrt(discriminant)) / (2 * safe_quadratic),
        ),
        axis=-1,
    )
    root = jnp.where(
        (roots[..., 0] >= 0) & (roots[..., 0] <= 1), roots[..., 0], roots[..., 1]
    )
    root = jnp.where(
        jnp.abs(quadratic) > 1e-12,
        root,
        -start_value / jnp.where(jnp.abs(linear) > 1e-12, linear, 1),
    )
    point = vertices + jnp.clip(root, 0, 1)[..., None] * (end - vertices)
    crossing = (start_value * end_value < 0) & (slot[None] < count[:, None])
    return point, crossing


def trace(points, crossing, neighbour, neighbour_edge, start, axis):
    cells, edges = crossing.shape
    slot = jnp.arange(edges)

    def step(carry, index):
        cell, incoming, done, closed, wall_hit, repeated, visited = carry
        possible = crossing[cell] & (slot != incoming)
        outgoing = jnp.argmax(possible)
        available = jnp.any(possible)
        point = points[cell, outgoing]
        peer = neighbour[cell, outgoing]
        next_cell = jnp.maximum(peer, 0)
        next_incoming = neighbour_edge[cell, outgoing]
        closing = (peer == start) & (index > 0)
        repeat = visited[next_cell] & ~closing
        wall = peer < 0
        stop = done | ~available | closing | repeat | wall
        visited = visited.at[cell].set(True)
        next_cell = jnp.where(stop, cell, next_cell)
        next_incoming = jnp.where(stop, incoming, next_incoming)
        return (
            next_cell,
            next_incoming,
            stop,
            closed | (closing & ~done),
            wall_hit | (wall & ~done),
            repeated | (repeat & ~done),
            visited,
        ), (point, ~done & available)

    initial = (
        jnp.asarray(start, jnp.int32),
        jnp.asarray(-1, jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.zeros(cells, bool),
    )
    final, (path, live) = jax.lax.scan(step, initial, jnp.arange(2 * cells))
    shifted = jnp.roll(path, -1, axis=0)
    vector = path - axis
    next_vector = shifted - axis
    angle = jnp.arctan2(
        vector[:, 0] * next_vector[:, 1] - vector[:, 1] * next_vector[:, 0],
        jnp.sum(vector * next_vector, axis=1),
    )
    count = jnp.sum(live)
    last = vector[jnp.maximum(count - 1, 0)]
    first = vector[0]
    closing_angle = jnp.arctan2(
        last[0] * first[1] - last[1] * first[0],
        jnp.sum(last * first),
    )
    winding = jnp.sum(jnp.where(live & jnp.roll(live, -1), angle, 0.0))
    winding += jnp.where(final[3] & (count > 1), closing_angle, 0.0)
    return final[3], final[4], final[5], count, winding, path, live


def run(cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    carrier, _, _ = certificate._case("diverted-single-null", clip_mode="exact")
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
    points, crossing = crossing_segments(geometry, reading, convention.sigma)
    saddle = reading.x_points[jnp.argmax(reading.x_point_valid)]
    candidates = jnp.sum(crossing, axis=1) >= 2
    distance = jnp.sum((geometry.centre - saddle) ** 2, axis=1)
    start = int(jnp.argmin(jnp.where(candidates, distance, jnp.inf)))

    def function():
        return trace(
            points,
            crossing,
            geometry.neighbour,
            geometry.neighbour_edge,
            start,
            reading.axis,
        )

    (closed, wall, repeated, count, winding, path, live), wall_ms, compile_s = (
        time_call(function)
    )
    edge_inside = inside_own_cell(points, geometry)
    escaped = int(jnp.sum(crossing & ~edge_inside))
    assert escaped == 0
    result = {
        "cells": len(machine.node),
        "start_cell": start,
        "crossing_cells": int(jnp.sum(candidates)),
        "wall_ms": wall_ms,
        "compile_s": compile_s,
        "closed": bool(closed),
        "hit_wall": bool(wall),
        "repeated_cell": bool(repeated),
        "trace_points": int(count),
        "axis_winding_radians": float(winding),
        "axis_inside": bool(closed & (jnp.abs(winding) > jnp.pi)),
        "wall_escape_crossings": escaped,
    }
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
