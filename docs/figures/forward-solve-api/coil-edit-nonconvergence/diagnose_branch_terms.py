"""Report, per persisted coil-edit state, which invalidation term fires.

The panel states are the twenty terminal rasters of one coil-edit sweep.  For
each edit, this driver assembles the separatrix branches at that state's own
boundary level and reports the term counts, then decomposes the traced level
set into connected components and states where an open arc ends.  A state
whose axis-enclosing component is not a cycle is reported with the coordinates
of the two degree-one ends, which is what tells a box-boundary exit from a
saddle join that merged the lobe into a leg.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    traced_spline_contour,
)
from nova.equilibrium.separatrix_branches import assemble_separatrix_branches

ROOT = Path(__file__).resolve().parents[4]
STATES = ROOT / "docs/figures/forward-solve-api/coil-edit-nonconvergence/panel-states.npz"
OUTPUT = ROOT / "docs/figures/forward-solve-api/coil-edit-nonconvergence/branch-terms.json"


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


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    states = np.load(STATES, allow_pickle=False)
    radius = jnp.asarray(states["radius"])
    height = jnp.asarray(states["height"])
    rows = []
    for index in range(20):
        psi = jnp.asarray(states[f"psi_{index}"])
        axis = jnp.asarray(states[f"axis_{index}"])
        xpoint = jnp.asarray(states[f"xpoints_{index}"]).reshape(-1, 2)
        level = fit_tensor_spline(radius, height, psi)(xpoint[0, 0], xpoint[0, 1])
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
                "term_that_fires": (
                    "none"
                    if bool(branches["well_formed"])
                    else "closed_candidate_count!=1"
                ),
            }
        )
    OUTPUT.write_text(json.dumps(rows, indent=2))
    failing = [row for row in rows if not row["well_formed"]]
    print(f"failing {len(failing)} of {len(rows)}")
    for row in rows:
        ends = sum(len(part["ends"]) for part in row["components"])
        print(
            f"{row['edit_index']:2d} well={int(row['well_formed'])} "
            f"cycle={row['cycle_component_count']} axis={row['axis_enclosing_component_count']} "
            f"open={row['open_branch_count']} comps={row['component_count']} ends={ends}"
        )


if __name__ == "__main__":
    main()
