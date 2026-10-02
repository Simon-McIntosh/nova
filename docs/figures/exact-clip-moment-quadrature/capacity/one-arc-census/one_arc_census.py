"""Census diverted certificate and production rows against the one-arc bound.

The committed ``realised_layout_census.py`` builds every analytic case at one
requested cell count (110) under the one-run bound, so it cannot report the
diverted row at 300 or 1000 cells, and its ``vertex_count.max()`` cannot see a
refused cell: the packer zeroes a refused cell's geometry before the maximum is
taken, so a row that refuses a cell still reads a maximum below the bound. This
driver measures the diverted analytic row at the three requested cell counts the
section names, records the refused cells by building each row a second time
under a raised capacity and differencing the two supports, and reports whether
any saddle wedge the spline wedge expansion builds would need a second level
run.

One process measures one analytic cell count or one production-bank member.
Each process writes its own partial receipt so a lost process loses one row
rather than the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks import exact_clip_moment_floor as floor
from benchmarks import jitted_eager_parity_gate as mast_forward
from benchmarks import strict_exit_incidence as production
from nova.equilibrium import separatrix_clip as sc
from nova.equilibrium import forward_operator as fo
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.separatrix_clip import (
    _SPLINE_BOUNDARY_SEGMENTS,
    traced_polygon_vertex_capacity,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from nova.linalg.split_spline import fit_split_spline

OUT = Path(__file__).resolve().parent
PRODUCTION_OUT = OUT / "production"
CELLS = (110, 300, 1000)
# The analytic certificate bank carries one diverted row, a single null. No
# double-null diverted row exists in the bank, so the census reports the row it
# can build and states the absence for the other shape.
DIVERTED_ROWS = ("diverted-single-null",)
RAISED_CAPACITY = 4096
PRODUCTION_MEMBER_COUNTS = {"mast": 6, "diiid": 5}


def _provenance_header() -> None:
    root = Path(__file__).resolve().parents[5]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    print(
        "PROVENANCE "
        + json.dumps(
            {
                "command": [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                ],
                "cwd": str(Path.cwd().resolve()),
                "module": str(Path(__file__).resolve()),
                "revision": revision,
                "tree": str(root),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _build(case_name: str, requested_cells: int):
    set_support_clip_mode("exact")
    operator, support, _field, _bank_capacity, _flux_span = floor._build(
        case_name, -requested_cells
    )
    return operator, support, int(operator.moment_geometry.atomic_mesh.support_capacity)


def _refused_cells(case_name: str, requested_cells: int, base) -> list[int]:
    """Return the cell indices refused at the one-arc bound.

    The packer zeroes a refused cell, so the geometry cannot name it. The row is
    rebuilt under a capacity raised past every realised vertex count and the two
    supports differenced: a cell the base run dropped whose raised count is
    positive is a refused cell.
    """
    base_count = np.asarray(base.vertex_count, dtype=np.intp)
    base_included = np.asarray(base.included, dtype=bool)
    original = sc.traced_polygon_vertex_capacity
    sc.traced_polygon_vertex_capacity = lambda straight_capacity, level_run_count=1: (
        RAISED_CAPACITY
    )
    try:
        _operator, raised, _straight = _build(case_name, requested_cells)
    finally:
        sc.traced_polygon_vertex_capacity = original
    raised_count = np.asarray(raised.vertex_count, dtype=np.intp)
    index = np.arange(len(base_count), dtype=np.intp)
    return [
        int(cell)
        for cell in index[
            (raised_count != base_count) & ~base_included & (raised_count > 0)
        ]
    ]


def one_case(case_name: str, requested_cells: int) -> dict:
    operator, support, straight = _build(case_name, requested_cells)
    count = np.asarray(support.vertex_count, dtype=np.intp)
    capacity = traced_polygon_vertex_capacity(straight)
    live = int(count.max()) if count.size else 0
    refused = _refused_cells(case_name, requested_cells, support)
    return {
        "schema": "nova.exact-clip-one-arc-census.v1",
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": int(len(operator.moment_geometry.atomic_mesh.centroids)),
        "included_cells": int(np.asarray(support.included, dtype=bool).sum()),
        "straight_vertex_capacity": straight,
        "one_arc_bound": capacity,
        "realised_maximum_vertex_count": live,
        "refused_cell_count": int(support.refused_cells()),
        "refused_cells": refused,
        "arc_sample_count": _SPLINE_BOUNDARY_SEGMENTS,
    }


def aggregate() -> dict:
    """Combine the three per-cell-count partials into the one census receipt."""
    rows = []
    for cells in CELLS:
        rows.extend(json.loads((OUT / f"cells-{cells}.json").read_text()))
    rows.sort(key=lambda row: (row["case"], row["requested_cells"]))
    refuses = [row for row in rows if row["refused_cell_count"]]
    receipt = {
        "schema": "nova.exact-clip-one-arc-census-receipt.v1",
        "bound": "one-arc (traced_polygon_vertex_capacity at the one-run default)",
        "cell_counts": list(CELLS),
        "diverted_rows_in_bank": list(DIVERTED_ROWS),
        "double_null_diverted_row_in_bank": False,
        "rows": rows,
        "any_row_refuses_a_cell": bool(refuses),
        "refusing_rows": refuses,
        "verdict": (
            "a diverted row refuses a cell: the held two-run wiring is the repair"
            if refuses
            else "no diverted row refuses a cell at any sampled cell count: the "
            "held two-run wiring is retired and the mechanism stays at its "
            "one-run default"
        ),
    }
    (OUT / "one-arc-census-receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return receipt


def _production_member(machine: str, member_number: int):
    if machine == "mast":
        shot, slice_index, _bank_row = mast_forward._case_rows(mast_forward.SHOT_STORE)[
            member_number - 1
        ]
        group = zarr.open_group(
            str(mast_forward.SHOT_STORE / f"{shot}.zarr"), mode="r"
        )["efm"]
        profile, _reference_seed, _reference, _provenance = mast_forward.build_profile(
            group, shot, slice_index, "fcoil_c"
        )
        profile = mast_forward._with_moment_geometry(profile)
        boundary = mast_forward._stored_lcfs(group, slice_index)
        target_current = abs(float(group["plasma_current_c"][slice_index]))
        seed = profile.moment_seed(boundary, target_current)
        member = SimpleNamespace(
            identity=f"{shot}/{slice_index}", profile=profile, state=seed.flux
        )
        selection = {
            "bank_member": member_number,
            "bank_width": PRODUCTION_MEMBER_COUNTS[machine],
            "identity": member.identity,
            "shot": shot,
            "slice_index": slice_index,
            "selection": (
                "every unique slice in the committed paired-arm MAST bank; "
                "the production moment seed is slice-owned rather than arm-owned"
            ),
        }
        evidence = {
            "bank": {
                "path": str(production.MAST_BANK.relative_to(production.ROOT)),
                "sha256": production._sha256(production.MAST_BANK),
            }
        }
        return member, selection, evidence
    member, evidence = production.build_diiid_member(
        production.DEFAULT_DIIID_MACHINE_CACHE,
        member_number,
        member_count=PRODUCTION_MEMBER_COUNTS[machine],
    )
    bank_row = production._diiid_bank_rows()[member_number - 1]
    selection = {
        "bank_member": member_number,
        "bank_width": PRODUCTION_MEMBER_COUNTS[machine],
        "shot": str(bank_row["shot"]),
        "frame": int(bank_row["frame"]),
        "selection": "every frame in the committed DIII-D forward gate bank",
    }
    return member, selection, evidence


def _profile_support(operator, state):
    return jax.block_until_ready(
        operator._support_partition(state, TopologyClass.DIVERTED)
    )


def _saddle_wedges(operator, state):
    physical = jnp.asarray(state)
    masks, topology, _connected, _admitted = jax.block_until_ready(
        operator._fixed_design_read(physical, TopologyClass.DIVERTED)
    )
    sample_flux = operator.sample_node_flux(physical)
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    flux_coefficient = operator.support_flux_coefficients(
        masks.psi_norm, sample_psi_norm
    )
    inside_coefficient = (-flux_coefficient).at[:, 0].add(1.0)
    coordinate = jnp.asarray(operator.grid.coordinate, dtype=masks.psi_norm.dtype)
    surface_value = masks.psi_norm[None, :]
    surface = fit_split_spline(
        coordinate[None, :, 0],
        coordinate[None, :, 1],
        surface_value,
        surface_value - 1.0,
        order=6,
        regularization=1.0e-14,
    )
    curved_level = fo._ExactClipLevel(
        surface,
        inside_coefficient,
        operator._support_curve_centre,
        operator._support_curve_scale,
    )
    shared_flux = operator.shared_node_flux(physical)
    inside_boundary = operator.polarity * (shared_flux - topology.boundary_flux)
    return jax.block_until_ready(
        operator.moment_geometry.atomic_mesh.traced_saddle_wedges(
            inside_boundary,
            saddle_vertex=topology.x_point,
            core_reference=topology.axis,
            curve_evaluator=curved_level,
            arc_tracer=fo._implicit_traced_level_arc,
        )
    )


def _production_refused_cells(operator, state, base) -> list[int]:
    if int(base.refused_cells()) == 0:
        return []
    base_count = np.asarray(base.vertex_count, dtype=np.intp)
    base_included = np.asarray(base.included, dtype=bool)
    original = sc.traced_polygon_vertex_capacity
    sc.traced_polygon_vertex_capacity = lambda straight_capacity, level_run_count=1: (
        RAISED_CAPACITY
    )
    try:
        _masks, _topology, _sample, raised = _profile_support(operator, state)
    finally:
        sc.traced_polygon_vertex_capacity = original
    raised_count = np.asarray(raised.vertex_count, dtype=np.intp)
    index = np.arange(len(base_count), dtype=np.intp)
    return [
        int(cell)
        for cell in index[
            (raised_count != base_count) & ~base_included & (raised_count > 0)
        ]
    ]


def production_case(machine: str, member_number: int) -> dict:
    set_support_clip_mode("exact")
    member, selection, evidence = _production_member(machine, member_number)
    operator = member.profile.operator
    _masks, topology, _sample, support = _profile_support(operator, member.state)
    wedges = _saddle_wedges(operator, member.state)
    count = np.asarray(support.vertex_count, dtype=np.intp)
    wedge_count = np.asarray(wedges.vertex_count, dtype=np.intp)
    straight = int(operator.moment_geometry.atomic_mesh.support_capacity)
    capacity = traced_polygon_vertex_capacity(straight)
    refused = _production_refused_cells(operator, member.state, support)
    return {
        "schema": "nova.exact-clip-production-one-arc-census.v1",
        "machine": machine.upper() if machine == "mast" else "DIII-D",
        "member_identity": member.identity,
        "selection": selection,
        "production_cell_count": int(
            len(operator.moment_geometry.atomic_mesh.centroids)
        ),
        "straight_vertex_capacity": straight,
        "one_arc_capacity": capacity,
        "realised_maximum_live_vertices": int(count.max()) if count.size else 0,
        "saddle_wedge_maximum_live_vertices": (
            int(wedge_count.max()) if wedge_count.size else 0
        ),
        "saddle_wedge_margin": capacity - int(wedge_count.max()),
        "saddle_cell_count": int(np.asarray(wedges.saddle, dtype=bool).sum()),
        "refused_cell_count": int(support.refused_cells()),
        "refused_cell_indices": refused,
        "topology": {
            "axis": np.asarray(topology.axis, dtype=float).tolist(),
            "x_point": np.asarray(topology.x_point, dtype=float).tolist(),
        },
        "bank": evidence["bank"],
    }


def aggregate_production() -> dict:
    rows = []
    for machine, member_count in PRODUCTION_MEMBER_COUNTS.items():
        for member_number in range(1, member_count + 1):
            path = PRODUCTION_OUT / f"{machine}-{member_number:02d}.json"
            rows.append(json.loads(path.read_text()))
    refusing = [row for row in rows if row["refused_cell_count"]]
    receipt = {
        "schema": "nova.exact-clip-production-one-arc-census-receipt.v1",
        "bound": "one traced level run plus the production mesh straight chain",
        "row_selection": {
            "mast": (
                "all six unique committed MAST slices, each rebuilt through the "
                "production moment-seed route shared by its paired bank arms"
            ),
            "diiid": "all five committed DIII-D forward gate frames",
        },
        "rows": rows,
        "row_count": len(rows),
        "any_row_refuses_a_cell": bool(refusing),
        "refusing_rows": refusing,
        "smallest_saddle_wedge_margin": min(row["saddle_wedge_margin"] for row in rows),
        "verdict": (
            "at least one production diverted row refuses the one-arc bound"
            if refusing
            else "no production diverted row refuses the one-arc bound"
        ),
    }
    path = PRODUCTION_OUT / "production-one-arc-census-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    _provenance_header()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, choices=list(CELLS))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--production-machine", choices=sorted(PRODUCTION_MEMBER_COUNTS)
    )
    parser.add_argument("--member", type=int)
    parser.add_argument("--aggregate-production", action="store_true")
    arguments = parser.parse_args()
    if arguments.aggregate_production:
        receipt = aggregate_production()
        print(
            "PRODUCTION_CENSUS_RECEIPT "
            + json.dumps(
                {
                    "any_row_refuses_a_cell": receipt["any_row_refuses_a_cell"],
                    "rows": receipt["row_count"],
                    "smallest_saddle_wedge_margin": receipt[
                        "smallest_saddle_wedge_margin"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if arguments.production_machine:
        if arguments.member is None:
            parser.error("--member is required with --production-machine")
        member_count = PRODUCTION_MEMBER_COUNTS[arguments.production_machine]
        if not 1 <= arguments.member <= member_count:
            parser.error(f"--member must be in [1, {member_count}]")
        configure_dtypes()
        PRODUCTION_OUT.mkdir(parents=True, exist_ok=True)
        row = production_case(arguments.production_machine, arguments.member)
        path = (
            PRODUCTION_OUT
            / f"{arguments.production_machine}-{arguments.member:02d}.json"
        )
        path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print("PRODUCTION_CENSUS_ROW " + json.dumps(row, sort_keys=True), flush=True)
        return
    if arguments.aggregate:
        receipt = aggregate()
        print(
            "CENSUS_RECEIPT "
            + json.dumps(
                {
                    "any_row_refuses_a_cell": receipt["any_row_refuses_a_cell"],
                    "rows": len(receipt["rows"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if arguments.cells is None:
        parser.error("--cells is required unless --aggregate is given")
    configure_dtypes()
    rows = [one_case(name, arguments.cells) for name in DIVERTED_ROWS]
    (OUT / f"cells-{arguments.cells}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    for row in rows:
        print("CENSUS_ROW " + json.dumps(row, sort_keys=True), flush=True)
    print(f"CENSUS_DONE {arguments.cells} {len(rows)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
