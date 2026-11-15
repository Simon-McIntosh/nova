"""Read one landed row's source-cutoff discriminator from its saved state.

The row's solve writes its terminal state and its convergence receipts; the
discriminator that compares the profile's booked cell current with the
value-only assignment needs the operator, not a solve. This driver rebuilds the
profile exactly as the row did — the edge arm's own tables, the DIII-D current
adapter and the sampling-node conductor completion — and reads the two routes
at the terminal state the row left on disk.

The comparison is on one operator, so it differs only in how the cell current
is taken from the sampled density: the booked route integrates the moment
blocks the consumer supplies, and the value-only route evaluates the point
weight at the cell centre.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BAND = 0.02


def load(name: str, path: Path):
    """Load a cohort module, with its own directory on the import path."""

    sibling = str(path.parent)
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


arms = load(
    "extent_arms_read",
    ROOT / "scripts/prototypes/diiid_xpoint_profile_extent/arms.py",
)
branches = load(
    "convergence_branches_read",
    ROOT / "scripts/prototypes/diiid_forward_convergence/measure_branches.py",
)

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402


def row_receipts(folder: Path) -> dict[str, object]:
    """Return the row's own convergence, offset and current-pin readings."""

    receipt = json.loads((folder / "receipt.json").read_text())
    core = json.loads((folder / "core-analysis.json").read_text())
    shadow = json.loads((folder / "shadow-image.json").read_text())
    return {
        "converged": bool(receipt["converged"]),
        "terminal_residual": float(receipt["residual"]),
        "terminal_reason": int(receipt["terminal_reason"]),
        "attempted": int(receipt["attempted"]),
        "accepted": int(receipt["accepted"]),
        "lower_x_distance_m": float(shadow["lower_x_distance_m"]),
        "lower_x_rz_m": [float(value) for value in shadow["lower_x_rz_m"]],
        "efit_x_rz_m": [float(value) for value in receipt["efit_x_point_rz_m"]],
        "current_pin_amplitude": float(core["amplitude"]),
        "target_current_a": float(core["target_current_a"]),
        "terminal_topology": receipt.get("terminal_topology"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=str, default="edge")
    parser.add_argument(
        "--probe-cell",
        type=float,
        nargs=2,
        metavar=("R", "Z"),
        help="report both routes at the grid cell nearest this position [m]",
    )
    args = parser.parse_args()

    configure_dtypes()
    arms.install(bench, args.arm)
    complete = bench.complete_profile_current_adapter

    def adapter(profile, **kwargs):
        return bench.extend_sample_current_response(complete(profile, **kwargs))

    bench.complete_profile_current_adapter = adapter

    prepared = branches._prepare_frame(args.frame, None)
    operator = prepared["profile"].operator
    state = np.load(args.output / "terminal.npz")
    psi = np.asarray(state["state"], dtype=np.float64)

    cells = int(operator.grid.node_number)
    topology = operator._fixed_design_read(psi)[1]
    axis = float(np.asarray(topology.axis_flux))
    boundary = float(np.asarray(topology.boundary_flux))
    receipt: dict[str, object] = {
        "frame": args.frame,
        "branch": args.branch,
        "arm": args.arm,
        "moment_route": bench.FINITE_EDGE_MOMENT_ROUTE,
        # The route the row's own operator carries, read rather than declared:
        # the booked route needs the linear moments and the sampling target.
        "operator_linear_moments": bool(operator.use_linear_moments),
        "operator_has_sample_target": operator.sample is not None,
        "state_size": int(psi.size),
        "grid_cells": cells,
        "axis_flux_wb": axis,
        "boundary_flux_wb": boundary,
    }
    # The span's sign follows the solve's own polarity and the normalised level
    # reads one at the boundary either way, so the band test takes the span as
    # it comes and rejects only an unresolved or degenerate level.
    if not (np.isfinite(axis) and np.isfinite(boundary)) or boundary == axis:
        receipt["status"] = "unresolved boundary level"
        receipt.update(row_receipts(args.output))
        emit(args.output, receipt)
        return

    normalised = (psi[:cells] - axis) / (boundary - axis)
    band = np.abs(normalised - 1.0) <= BAND
    receipt["status"] = "measured" if band.any() else "empty band"
    if band.any():
        booked = np.asarray(operator.cell_current(psi))[:cells]
        value_only = np.asarray(
            replace(operator, use_linear_moments=False, sample=None).cell_current(psi)
        )[:cells]
        difference = np.abs(booked - value_only)[band]
        band_peak = float(np.abs(booked[band]).max())
        receipt.update(
            band_cells=int(band.sum()),
            booked_band_current_sum_a=float(booked[band].sum()),
            value_only_band_current_sum_a=float(value_only[band].sum()),
            max_cell_difference_a=float(difference.max()),
            band_peak_current_a=band_peak,
            difference_relative_to_band_peak=(
                float(difference.max() / band_peak) if band_peak else None
            ),
            raw_current_a=float(booked.sum()),
        )
        if args.probe_cell is not None:
            receipt["probe_cell"] = probe(
                state, psi, booked, value_only, args.probe_cell
            )
    receipt.update(row_receipts(args.output))
    emit(args.output, receipt)


def probe(state, psi, booked, value_only, cell: list[float]) -> dict[str, object]:
    """Return both routes' current at the grid cell nearest a named position.

    The grid coordinates are the row's own product: the terminal state's radius
    and height meshes ravel in the order the row stores its grid nodes, and the
    saved grid flux matches the solved state under that same ravel. That match
    is asserted rather than assumed, because a mesh raveled in another order
    would put the probe on a cell the diagnosis never named.
    """

    radius = np.asarray(state["radius"], dtype=np.float64)
    height = np.asarray(state["height"], dtype=np.float64)
    mesh_flux = np.asarray(state["flux"], dtype=np.float64)
    if not np.allclose(
        mesh_flux.ravel(), np.asarray(psi)[: mesh_flux.size], rtol=0, atol=1e-9
    ):
        raise SystemExit(
            "the saved grid flux does not match the solved state under the "
            "radius-height ravel, so a cell cannot be named by position"
        )
    radius_mesh, height_mesh = np.meshgrid(radius, height, indexing="ij")
    distance = np.hypot(radius_mesh - cell[0], height_mesh - cell[1])
    i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)
    index = int(i * height.size + j)
    return {
        "requested_rz_m": [float(cell[0]), float(cell[1])],
        "grid_cell_index": index,
        "grid_cell_ij": [int(i), int(j)],
        "grid_cell_rz_m": [float(radius[i]), float(height[j])],
        "distance_m": float(distance[i, j]),
        "booked_current_a": float(booked[index]),
        "value_only_current_a": float(value_only[index]),
        "difference_a": float(booked[index] - value_only[index]),
    }


def emit(folder: Path, receipt: dict[str, object]) -> None:
    (folder / "finite-edge.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("FINITE_EDGE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
