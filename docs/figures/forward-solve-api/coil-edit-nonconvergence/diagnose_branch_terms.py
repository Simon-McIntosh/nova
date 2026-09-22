"""Report, per persisted coil-edit state, which invalidation term fires.

The panel states are the twenty terminal rasters of one coil-edit sweep.  For
each edit, this driver assembles the separatrix branches at that state's own
boundary level -- the flux read at the state's own admitted saddle, which is
the level that pairs that saddle cell's crossings -- and reports which term
fires, then decomposes the traced level set into connected components and
states where an open arc ends.  A state
whose axis-enclosing component is not a cycle is reported with the coordinates
of the two degree-one ends, which is what tells a box-boundary exit from a
saddle join that merged the lobe into a leg.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.flux_surface_connectivity import traced_spline_contour
from nova.equilibrium.separatrix_branches import (
    assemble_separatrix_branches,
    boundary_flux_at_admitted_saddle,
)

ROOT = Path(__file__).resolve().parents[4]
CASE = ROOT / "docs/figures/forward-solve-api/coil-edit-nonconvergence"
STATES = CASE / "panel-states.npz"
OUTPUT = CASE / "branch-terms.json"


def components(contour):
    endpoints = np.asarray(contour["segment_endpoints_rz"]).reshape(-1, 2, 2)
    valid = np.asarray(contour["segment_valid"]).reshape(-1).astype(bool)
    flat = np.round(endpoints[valid].reshape(-1, 2), 9)
    key: dict[tuple, int] = {}
    ids = np.zeros(flat.shape[0], dtype=int)
    for position, point in enumerate(flat):
        token = (float(point[0]), float(point[1]))
        if token not in key:
            key[token] = len(key)
        ids[position] = key[token]
    ids = ids.reshape(-1, 2)
    degree = np.zeros(len(key), dtype=int)
    for left, right in ids:
        degree[left] += 1
        degree[right] += 1
    parent = list(range(len(key)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in ids:
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b
    groups = defaultdict(list)
    for node in range(len(key)):
        groups[find(node)].append(node)
    inverse = {value: name for name, value in key.items()}
    found = []
    for _, members in sorted(groups.items(), key=lambda item: -len(item[1])):
        ends = [member for member in members if degree[member] == 1]
        found.append(
            {
                "node_count": len(members),
                "open": bool(ends),
                "ends": [[float(inverse[e][0]), float(inverse[e][1])] for e in ends],
            }
        )
    return found


def failing_terms(branches) -> list[str]:
    """Return the names of the well-formedness terms a state fails."""
    terms = []
    if not bool(branches["contour_well_formed"]):
        terms.append("contour")
    if not bool(branches["graph_well_formed"]):
        terms.append("graph_junction")
    if int(branches["closed_candidate_count"]) != 1:
        terms.append("closed_candidate_count!=1")
    if bool(branches["branch_overflow"]):
        terms.append("branch_overflow")
    if bool(branches["open_slot_overflow"]):
        terms.append("open_slot_overflow")
    return terms


def main(states_path: Path = STATES, output_path: Path = OUTPUT) -> None:
    jax.config.update("jax_enable_x64", True)
    states = np.load(states_path, allow_pickle=False)
    radius = jnp.asarray(states["radius"])
    height = jnp.asarray(states["height"])
    rows = []
    # The archive states its own extent: a sweep that stopped early is reported
    # over the states it persisted rather than refused for the ones it did not.
    for index in range(len(np.asarray(states["edit_index"]))):
        psi = jnp.asarray(states[f"psi_{index}"])
        axis = jnp.asarray(states[f"axis_{index}"])
        xpoint = jnp.asarray(states[f"xpoints_{index}"]).reshape(-1, 2)
        level = boundary_flux_at_admitted_saddle(psi, radius, height, axis, xpoint[0])
        branches = assemble_separatrix_branches(psi, radius, height, level, axis)
        parts = components(traced_spline_contour(psi, radius, height, level))
        rows.append(
            {
                "edit_index": index,
                "level": float(level),
                "closed_candidate_count": int(branches["closed_candidate_count"]),
                "well_formed": bool(branches["well_formed"]),
                "cycle_component_count": int(branches["cycle_component_count"]),
                "axis_enclosing_component_count": int(
                    branches["axis_enclosing_component_count"]
                ),
                "open_branch_count": int(branches["open_branch_count"]),
                "closed_segment_count": int(jnp.sum(branches["closed_valid"])),
                "component_count": len(parts),
                "components": parts,
                "terms_that_fire": failing_terms(branches),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    failing = [row for row in rows if not row["well_formed"]]
    print(f"failing {len(failing)} of {len(rows)}")
    for row in rows:
        ends = sum(len(part["ends"]) for part in row["components"])
        print(
            f"{row['edit_index']:2d} well={int(row['well_formed'])} "
            f"cycle={row['cycle_component_count']} "
            f"axis={row['axis_enclosing_component_count']} "
            f"open={row['open_branch_count']} closed={row['closed_segment_count']} "
            f"comps={row['component_count']} ends={ends} "
            f"terms={','.join(row['terms_that_fire']) or 'none'} "
            f"level={row['level']:.9f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=STATES,
        help="panel archive to read; defaults to the case's own fixture",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="record to write; defaults to the case's own branch-terms record",
    )
    arguments = parser.parse_args()
    main(arguments.states, arguments.output)
