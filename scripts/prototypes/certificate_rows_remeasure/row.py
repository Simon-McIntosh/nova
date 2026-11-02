"""Measure one Solovev certificate row in a private root and emit its panel input."""

# ruff: noqa: E402

import argparse
from functools import partial
import json
from pathlib import Path
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import nova
from nova.biot import polygonanalytic

assert jax.config.jax_enable_x64 is True

LOGICAL = {
    ("diverted-single-null", -300): (
        "docs/figures/solovev/diverted-single-null-production-route-cells-300.png"
    ),
    ("weak-rotation-reactor-static", -500): (
        "docs/figures/solovev/weak-rotation-reactor-static-production-route-cells-500.png"
    ),
}


def _roots(topology):
    axis = topology["axis_rz_m"]
    return [axis, *topology["x_point_rz_m"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--requested-cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks import solovev_certificate as certificate

    logical = LOGICAL[(args.case, args.requested_cells)]
    root = args.output.parent / "private"
    certificate.FIGURE_ROOT = root / "figures"
    certificate.PART_ROOT = root / "parts"
    certificate.DIAGNOSTIC_ROOT = root / "diagnostics"
    certificate._figure_receipts = partial(
        certificate._figure_receipts, logical=logical
    )
    print(
        f"nova={nova.__file__} measurement_module={polygonanalytic.__file__} "
        f"cwd={Path.cwd().resolve()}",
        flush=True,
    )
    sys.argv = [
        "benchmarks/solovev_certificate.py",
        "--case",
        args.case,
        "--requested-cells",
        str(args.requested_cells),
    ]
    certificate.main()
    part = json.loads(
        certificate._part_path(args.case, args.requested_cells).read_text()
    )
    solver = part["solver"]
    receipt = part.get("receipt", {})
    row = {
        "case": args.case,
        "requested_cells": args.requested_cells,
        "realised_cells": part["realised_cells"],
        "terminal_residual": solver["terminal_fixed_point_residual"],
        "converged": solver["converged"],
        "qualification": solver["qualification"],
        "qualification_reason": solver["qualification_reason"],
        "termination": solver["termination"],
        "trips": solver["production_telemetry"]["trip_count"],
        "lane": part["lane"],
        "solve_wall_seconds": solver.get("solve_wall_seconds"),
        "row_wall_seconds": part["lane"].get("elapsed_seconds"),
        "stage_wall_seconds": part.get("stage_wall_seconds"),
        "receipt_status": receipt.get("status", "solved"),
        "receipt_served_from": receipt.get("served_from"),
        "solve_status": solver.get("solve_status"),
    }
    data = part["render_data"]
    panel = {
        "coordinates": data["coordinates_rz_m"],
        "wall": data["wall_units_rz_m"][0],
        "solved_flux": data["terminal_flux_wb"],
        "analytic_flux": data["analytic_flux_wb"],
        "solved_roots": _roots(data["terminal_topology"]),
        "analytic_roots": _roots(data["analytic_topology"]),
        "terminal_residual": row["terminal_residual"],
        "converged": row["converged"],
    }
    args.output.write_text(json.dumps(row, indent=2, default=str) + "\n")
    (args.output.parent / "panel.json").write_text(json.dumps(panel) + "\n")
    print("CERTIFICATE_ROW " + json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()