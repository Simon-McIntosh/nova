"""Measure tree overflow against the exact carrier supplied to the read."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import nova  # noqa: E402
from nova.equilibrium import contour_tree, topology  # noqa: E402
from scripts.prototypes.contour_tree_sweep_profile.profile import inputs  # noqa: E402
from scripts.prototypes.efit_field_swap.crosscheck import (  # noqa: E402
    domain_rings,
    lattice,
    read_trace_record,
)
from scripts.prototypes.efit_field_swap.orientation import read_convention  # noqa: E402
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField  # noqa: E402
from scripts.prototypes.efit_maximin_certificate.run import ROWS  # noqa: E402

assert jax.config.jax_enable_x64 is True


def mastu_inputs(resolution: int):
    record = read_trace_record(ROWS[0]["source"])
    field = TraceSplineField.from_record(record)
    boundary, obstacles = domain_rings(record)
    cells, samples = lattice(boundary, resolution, obstacles)
    geometry = topology.TopologyGeometry.from_cells(
        cells, samples, (boundary, *obstacles)
    )
    _, centres = jax.jit(topology._sampled_flux)(field, geometry)
    values = jnp.sum(
        centres[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    mesh = geometry.contour_mesh
    sigma = read_convention(polarity=record.polarity).sigma
    return (
        (
            values,
            mesh.vertex_valid,
            mesh.vertex_is_wall,
            mesh.edges,
            mesh.edge_valid,
            sigma,
        ),
        len(cells),
        mesh,
    )


def diagnose(arguments, case: str, cells: int, mesh=None):
    values, valid, wall, edges, edge_valid, sigma = arguments
    signed = sigma * values
    neighbours, neighbour_valid, adjacency_overflow = contour_tree._carrier_adjacency(
        edges, edge_valid, values.size
    )
    join = contour_tree._sweep_tree(
        signed,
        valid,
        wall,
        neighbours,
        neighbour_valid,
        True,
        values.size,
        edges.shape[0],
    )
    split = contour_tree._sweep_tree(
        signed,
        valid,
        wall,
        neighbours,
        neighbour_valid,
        False,
        values.size,
        edges.shape[0],
    )
    split_only = split[0] & ~join[0]
    merged = contour_tree._insert_split_nodes(
        join[0] | split[0], join[2], join[3], join[4] | split[4], split_only, signed
    )
    receipt = contour_tree.build_contour_tree(*arguments)
    jax.block_until_ready(receipt)
    live_edges = np.asarray(edges)[np.asarray(edge_valid)]
    canonical_edges = np.sort(live_edges, axis=1)
    degree = np.bincount(live_edges.ravel(), minlength=values.size)
    adjacency_capacity = 16 if edges.shape[0] > 2 * values.size else edges.shape[0]
    overfull_vertices = np.flatnonzero(degree > adjacency_capacity)
    live_values = np.asarray(values)[np.asarray(valid)]
    endpoint_values = np.asarray(values)[live_edges]
    result = {
        "case": case,
        "cells": cells,
        "carrier_vertices": int(np.asarray(valid).sum()),
        "carrier_vertex_slots": int(values.size),
        "carrier_edges": len(live_edges),
        "carrier_edge_slots": int(edges.shape[0]),
        "carrier_wall_vertices": int(
            np.count_nonzero(np.asarray(valid) & np.asarray(wall))
        ),
        "carrier_duplicate_edges": len(live_edges)
        - len(np.unique(canonical_edges, axis=0)),
        "carrier_self_edges": int(
            np.count_nonzero(live_edges[:, 0] == live_edges[:, 1])
        ),
        "carrier_max_degree": int(degree.max()),
        "adjacency_capacity": adjacency_capacity,
        "adjacency_overfull_vertices": int(len(overfull_vertices)),
        "adjacency_overfull_degrees": degree[overfull_vertices].tolist(),
        "adjacency_overfull_wall_vertices": int(
            np.count_nonzero(np.asarray(wall)[overfull_vertices])
        ),
        "carrier_equal_flux_edges": int(
            np.count_nonzero(endpoint_values[:, 0] == endpoint_values[:, 1])
        ),
        "carrier_unique_flux_values": int(len(np.unique(live_values))),
        "join_nodes": int(np.asarray(join[0]).sum()),
        "join_edges": int(np.asarray(join[3]).sum()),
        "split_nodes": int(np.asarray(split[0]).sum()),
        "split_edges": int(np.asarray(split[3]).sum()),
        "split_only": int(np.asarray(split_only).sum()),
        "merged_nodes": int(np.asarray(merged[0]).sum()),
        "merged_edges": int(np.asarray(merged[2]).sum()),
        "node_capacity": contour_tree.ContourTreeResult.node_capacity,
        "edge_capacity": contour_tree.ContourTreeResult.edge_capacity,
        "adjacency_overflow": bool(adjacency_overflow),
        "join_append_overflow": bool(join[4]),
        "split_append_overflow": bool(split[4]),
        "split_insertion_overflow": bool(merged[3]),
        "node_capacity_exceeded": int(np.asarray(merged[0]).sum())
        > contour_tree.ContourTreeResult.node_capacity,
        "edge_capacity_exceeded": int(np.asarray(merged[2]).sum())
        > contour_tree.ContourTreeResult.edge_capacity,
        "native_overflow": bool(receipt.overflow),
        "native_nodes": int(np.asarray(receipt.node_valid).sum()),
        "native_edges": int(np.asarray(receipt.edge_valid).sum()),
    }
    if mesh is not None:
        coords = np.asarray(mesh.vertex_rz)[np.asarray(valid)]
        result["carrier_duplicate_coordinates"] = len(coords) - len(
            np.unique(coords, axis=0)
        )
        result["adjacency_overfull_positions"] = np.asarray(mesh.vertex_rz)[
            overfull_vertices
        ].tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("mastu", "limited"), required=True)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    command = shlex.join([sys.executable, *sys.argv])
    print(
        f"revision={revision} tree={os.getcwd()} command={command}",
        flush=True,
    )
    print(f"nova.__file__={nova.__file__}", flush=True)
    print(f"contour_tree.__file__={contour_tree.__file__}", flush=True)
    print(f"cwd={os.getcwd()}", flush=True)
    if args.case == "mastu":
        if args.resolution is None:
            parser.error("mastu requires --resolution")
        arguments, cells, mesh = mastu_inputs(args.resolution)
    else:
        arguments, cells = inputs("limited", 550)
        mesh = None
    result = diagnose(arguments, args.case, cells, mesh)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DIAGNOSIS=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
