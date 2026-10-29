"""Report the diverted 300-cell carrier and its target current, without a solve.

The certificate row's convergence depends on which cells the carryer gives the
solve and on the declared target current, both of which are fixed before the
Newton loop starts. This program reports them for one tree, so two revisions can
be compared in seconds rather than in a compile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PATH_ROOT = Path(__file__).resolve().parents[3]
if str(PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATH_ROOT))

from benchmarks import solovev_certificate as certificate  # noqa: E402

try:  # the lattice identity is absent before the change that introduced it
    from nova.equilibrium.stencil_mesh import lattice_vertex_groups
except ImportError:  # pragma: no cover - older tree
    lattice_vertex_groups = None

try:  # the canonicaliser is absent before the identity that needs it
    from nova.equilibrium.stencil_mesh import canonical_polygons as _canonical
except ImportError:  # pragma: no cover - older tree
    _canonical = None
from nova.jax.config import configure_dtypes  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as oracle_fixture  # noqa: E402


def _duplicate_vertex_cells(polygons) -> int:
    return int(
        sum(
            len(np.unique(np.round(polygon, 12), axis=0)) != len(polygon)
            for polygon in polygons
        )
    )


def report(case_name: str, requested_cells: int) -> dict[str, object]:
    configure_dtypes()
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, fixture_cache = (
        oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = oracle_fixture.forward_operator(source_case, machine, fixture_exterior)
    target_current, centroid, _ = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    cells = [np.asarray(cell, dtype=np.float64) for cell in machine.cell_polygons]
    groups = (
        lattice_vertex_groups(cells)[0] if lattice_vertex_groups is not None else None
    )
    canonical = _canonical(cells) if _canonical is not None else cells
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "hex_cells": int(len(machine.node)),
        "solve_coordinates": int(machine.node.shape[0]),
        "sample_coordinates": int(machine.sample_coordinates.shape[0]),
        "lattice_group_count": (
            int(len(np.unique(groups))) if groups is not None else None
        ),
        "cells_carrying_a_repeated_vertex": _duplicate_vertex_cells(cells),
        "canonical_cells_carrying_a_repeated_vertex": _duplicate_vertex_cells(
            canonical
        ),
        "maximum_cell_vertex_count": int(max(len(cell) for cell in canonical)),
        "target_current_a": float(target_current),
        "current_centroid_m": np.asarray(centroid).tolist(),
        "fixture_cache_hit": bool(fixture_cache.get("hit")),
        "fixture_cache_store": str(fixture_cache.get("store")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="diverted-single-null")
    parser.add_argument("--requested-cells", type=int, default=-300)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    import nova

    print(f"module={nova.__file__}", flush=True)
    row = report(arguments.case, arguments.requested_cells)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
