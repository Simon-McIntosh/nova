"""Locate overflow and capacity use in one contour-tree carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prototypes.contour_tree_sweep_profile.profile import inputs, load_builder

import jax
import jax.numpy as jnp
import numpy as np


def diagnose(module, kind: str, cells: int) -> dict:
    arguments, realised = inputs(kind, cells)
    flux, valid, wall, edges, edge_valid, sigma = arguments
    signed = jnp.asarray(sigma, dtype=flux.dtype) * flux
    adjacency = module._carrier_adjacency(edges, edge_valid, flux.size)
    neighbours, neighbour_valid, adjacency_overflow = adjacency
    passes = {}
    for name, descending in (("join", True), ("split", False)):
        passes[name] = module._sweep_tree(
            signed,
            valid,
            wall,
            neighbours,
            neighbour_valid,
            descending,
            flux.size,
            edges.shape[0],
        )
    join, split = passes["join"], passes["split"]
    inserted = module._insert_split_nodes(
        join[0] | split[0],
        join[2],
        join[3],
        join[4] | split[4],
        split[0] & ~join[0],
        signed,
    )
    receipt = module.build_contour_tree(*arguments)
    jax.block_until_ready(receipt)

    node_valid = np.asarray(inserted[0]).copy()
    merged_edges = np.asarray(inserted[1]).copy()
    merged_valid = np.asarray(inserted[2])
    join_order = np.asarray(join[-1])
    usable_ranks = np.flatnonzero(np.asarray(valid)[join_order])
    terminal = int(join_order[usable_ranks[-1]])
    invalid_slots = np.flatnonzero(~np.asarray(valid))
    virtual = int(invalid_slots[0]) if invalid_slots.size else 0
    replace_terminal = bool(invalid_slots.size and np.asarray(join[0])[terminal])
    if replace_terminal:
        node_valid[terminal] = False
        node_valid[virtual] = True
        merged_edges[merged_edges == terminal] = virtual
    active_edges = merged_edges[merged_valid]
    invalid_endpoints = int((~node_valid[active_edges]).any(axis=1).sum())
    source_flags = {
        "adjacency": bool(adjacency_overflow),
        "join_append": bool(join[4]),
        "split_append": bool(split[4]),
        "split_insertion": bool(inserted[3]),
        "node_capacity": int(node_valid.sum()) > module.ContourTreeResult.node_capacity,
        "edge_capacity": int(merged_valid.sum())
        > module.ContourTreeResult.edge_capacity,
        "invalid_edge_endpoint": invalid_endpoints > 0,
    }
    return {
        "module": str(Path(module.__file__).resolve()),
        "device": str(jax.devices()[0]),
        "case": kind,
        "requested_cells": cells,
        "realised_cells": realised,
        "carrier_vertices": int(flux.size),
        "carrier_edges": int(edge_valid.sum()),
        "join_nodes": int(join[0].sum()),
        "join_edges": int(join[3].sum()),
        "split_nodes": int(split[0].sum()),
        "split_edges": int(split[3].sum()),
        "split_only": int((split[0] & ~join[0]).sum()),
        "merged_nodes": int(node_valid.sum()),
        "merged_edges": int(merged_valid.sum()),
        "invalid_edge_endpoints": invalid_endpoints,
        "node_capacity": module.ContourTreeResult.node_capacity,
        "edge_capacity": module.ContourTreeResult.edge_capacity,
        "source_flags": source_flags,
        "source_or": any(source_flags.values()),
        "native_overflow": bool(receipt.overflow),
        "native_nodes": int(receipt.node_valid.sum()),
        "native_edges": int(receipt.edge_valid.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(load_builder(args.module), args.kind, args.cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("OVERFLOW_DIAGNOSIS " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
