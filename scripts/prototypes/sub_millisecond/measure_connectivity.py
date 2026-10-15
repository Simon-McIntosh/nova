"""Compare connected-region reads on one analytic certificate carrier."""

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
from matplotlib.path import Path as PolygonPath

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import flux_surface_connectivity, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def time_call(function, *args, repeats=7):
    started = perf_counter()
    program = jax.jit(function).lower(*args).compile()
    compile_seconds = perf_counter() - started
    jax.block_until_ready(program(*args))
    walls = []
    for _ in range(repeats):
        started = perf_counter()
        jax.block_until_ready(program(*args))
        walls.append(1000 * (perf_counter() - started))
    return program(*args), float(np.median(walls)), compile_seconds


def fragment_graph(reading, geometry, tolerance):
    area = np.asarray(reading.fragment_area)
    intervals = np.asarray(reading.edge_interval)
    neighbour = np.asarray(geometry.neighbour)
    neighbour_edge = np.asarray(geometry.neighbour_edge)
    selected = np.asarray(reading.fragment_selected)
    cells, capacity = area.shape
    active = np.flatnonzero(area.reshape(-1) > 0)
    address = np.full(cells * capacity, -1, dtype=np.int32)
    address[active] = np.arange(len(active))
    links = np.eye(len(active), dtype=bool)
    for cell in range(cells):
        for edge in range(neighbour.shape[1]):
            peer = neighbour[cell, edge]
            if peer < 0:
                continue
            peer_edge = neighbour_edge[cell, edge]
            for left in range(capacity):
                i = address[cell * capacity + left]
                if i < 0:
                    continue
                one = intervals[cell, left, edge]
                for right in range(capacity):
                    j = address[peer * capacity + right]
                    if j < 0:
                        continue
                    other = 1 - intervals[peer, right, peer_edge, :, ::-1]
                    overlap = np.minimum(
                        one[:, None, 1], other[None, :, 1]
                    ) - np.maximum(one[:, None, 0], other[None, :, 0])
                    if np.any(overlap > tolerance):
                        links[i, j] = True
                        links[j, i] = True
    exact = selected.reshape(-1)[active]
    seed = int(np.flatnonzero(exact)[0])
    return links, exact, seed, active, address


def dense_reach(links, seed):
    power = links.astype(jnp.float32)
    for _ in range(11):
        power = (power @ power) > 0
        power = power.astype(jnp.float32)
    return power[seed] > 0


def merge_levels(values, neighbour, wall, seed):
    """Sorted cell-graph union with a fixed device scan and both first events."""
    count = values.size
    order = jnp.argsort(values).astype(jnp.int32)
    index = jnp.arange(count, dtype=jnp.int32)

    def root(parent, start):
        return jax.lax.while_loop(
            lambda position: parent[position] != position,
            lambda position: parent[position],
            jnp.asarray(start, dtype=jnp.int32),
        )

    def visit(carry, current):
        parent, active, axis, size, limited, diverted = carry
        active = active.at[current].set(True)
        parent = parent.at[current].set(current)
        axis = axis.at[current].set(current == seed)
        size = size.at[current].set(1)

        def join(edge, state):
            parent, axis, size, diverted = state
            other = neighbour[current, edge]
            valid = (other >= 0) & active[jnp.maximum(other, 0)]
            one_root = root(parent, current)
            other_root = root(parent, jnp.maximum(other, 0))
            distinct = valid & (one_root != other_root)
            merge_axis = axis[one_root] | axis[other_root]
            saddle = (
                distinct & axis[one_root] & ~axis[other_root] & (size[other_root] > 1)
            )
            saddle = saddle | (
                distinct & axis[other_root] & ~axis[one_root] & (size[one_root] > 1)
            )
            diverted = jnp.where(
                jnp.isnan(diverted) & saddle, values[current], diverted
            )
            parent = parent.at[other_root].set(
                jnp.where(distinct, one_root, parent[other_root])
            )
            axis = axis.at[one_root].set(
                jnp.where(distinct, merge_axis, axis[one_root])
            )
            size = size.at[one_root].set(
                jnp.where(distinct, size[one_root] + size[other_root], size[one_root])
            )
            return parent, axis, size, diverted

        parent, axis, size, diverted = jax.lax.fori_loop(
            0, neighbour.shape[1], join, (parent, axis, size, diverted)
        )
        axis_root = root(parent, seed)
        limited = jnp.where(
            jnp.isnan(limited)
            & active[seed]
            & (root(parent, current) == axis_root)
            & wall[current],
            values[current],
            limited,
        )
        return (parent, active, axis, size, limited, diverted), None

    initial = (
        index,
        jnp.zeros(count, bool),
        jnp.zeros(count, bool),
        jnp.zeros(count, jnp.int32),
        jnp.asarray(jnp.nan, values.dtype),
        jnp.asarray(jnp.nan, values.dtype),
    )
    final, _ = jax.lax.scan(visit, initial, order)
    return final[-2], final[-1]


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={topology.__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, _, _ = certificate._case(name, clip_mode="exact")
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
    links, exact, seed, active, address = fragment_graph(
        reading, geometry, policy.edge_tolerance
    )
    graph_component, graph_ms, graph_compile = time_call(
        dense_reach, jnp.asarray(links), seed
    )
    graph_component = np.asarray(graph_component)
    assert np.array_equal(graph_component, exact), (
        "graph extraction must reproduce the exact fragment labels"
    )
    _, label_ms, label_compile = time_call(
        lambda area, intervals: topology._connected_fragment_labels(
            area, intervals, geometry, policy.edge_tolerance
        ),
        reading.fragment_area,
        reading.edge_interval,
    )
    centre = np.asarray(geometry.centre)
    axis = np.asarray(reading.axis)
    wall = np.asarray(machine.wall_node)
    path = PolygonPath(wall)
    width = 32
    radial = np.linspace(centre[:, 0].min(), centre[:, 0].max(), width)
    vertical = np.linspace(centre[:, 1].min(), centre[:, 1].max(), width)
    rr, zz = np.meshgrid(radial, vertical)
    points = np.stack((rr, zz), axis=-1)
    values = jax.vmap(field.value)(jnp.asarray(points.reshape(-1, 2))).reshape(
        width, width
    )
    inside = jnp.asarray(
        path.contains_points(points.reshape(-1, 2)).reshape(width, width)
    )
    confined = inside & (convention.sigma * (values - reading.boundary_flux) > 0)
    ai = int(np.argmin(abs(radial - axis[0])))
    aj = int(np.argmin(abs(vertical - axis[1])))
    seed_grid = jnp.zeros((width, width), bool).at[aj, ai].set(True)
    (raster, steps), raster_ms, raster_compile = time_call(
        lambda mask: flux_surface_connectivity.flood_fill_core_with_steps(
            mask, seed_grid, 64
        ),
        confined,
    )
    ri = np.clip(np.searchsorted(radial, centre[:, 0]), 0, width - 1)
    zi = np.clip(np.searchsorted(vertical, centre[:, 1]), 0, width - 1)
    exact_cells = np.any(np.asarray(reading.fragment_selected), axis=1)
    raster_cells = np.asarray(raster)[zi, ri] > 0
    scalar = np.asarray(jax.vmap(field.value)(geometry.centre))
    local = np.asarray(convention.sigma * (scalar - reading.boundary_flux) > 0)
    cell_neighbour = np.asarray(geometry.neighbour)
    wall_cell = np.any(cell_neighbour < 0, axis=1)
    axis_cell = int(np.argmin(np.sum((centre - axis) ** 2, axis=1)))
    outward = jnp.asarray(convention.sigma * (scalar - float(reading.axis_flux)))
    (limited, diverted), merge_ms, merge_compile = time_call(
        lambda row: merge_levels(
            row, jnp.asarray(cell_neighbour), jnp.asarray(wall_cell), axis_cell
        ),
        outward,
    )
    result = {
        "kind": kind,
        "cells": len(centre),
        "fragments": len(active),
        "exact_selected_cells": int(exact_cells.sum()),
        "exact_selected_fragments": int(exact.sum()),
        "candidate": {
            "fragment_labels": {
                "wall_ms": label_ms,
                "compile_s": label_compile,
                "agreement": 1.0,
            },
            "dense_reachability": {
                "wall_ms": graph_ms,
                "compile_s": graph_compile,
                "agreement": float(np.mean(graph_component == exact)),
            },
            "raster_doubling": {
                "wall_ms": raster_ms,
                "compile_s": raster_compile,
                "steps": int(steps),
                "cell_agreement": float(np.mean(raster_cells == exact_cells)),
            },
            "critical_point_local": {
                "cell_agreement": float(np.mean(local == exact_cells))
            },
            "sorted_merge_tree": {
                "wall_ms": merge_ms,
                "compile_s": merge_compile,
                "limited_level": float(limited),
                "diverted_level": float(diverted),
            },
        },
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
