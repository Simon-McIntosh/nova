"""Measure one Solovev certificate row in a private root and draw its panel."""

# ruff: noqa: E402

import argparse
from functools import partial
import json
from pathlib import Path
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--requested-cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
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
    source = part["figure"]["render_source"]
    if source != "fresh_production_solve":
        raise RuntimeError(f"banked part receipt served: render_source={source}")
    banked = part.get("receipt")
    if banked is not None:
        raise RuntimeError(f"banked part receipt served: receipt={banked}")
    solver = part["solver"]
    data = part["render_data"]
    args.panel.parent.mkdir(parents=True, exist_ok=True)
    figure = certificate._plot(
        np.asarray(data["coordinates_rz_m"]),
        np.asarray(data["terminal_flux_wb"]),
        np.asarray(data["analytic_flux_wb"]),
        np.asarray(data["derivative_coordinates_rz_m"]),
        {
            name: np.asarray(data["error_fields"][name])
            for name in certificate.NORM_FIELDS
        },
        np.asarray(data["boundary_rz_m"]),
        np.asarray(data["wall_units_rz_m"][0]),
        data["terminal_topology"],
        data["analytic_topology"],
        args.panel,
        (
            f"{args.case} · {certificate._slug(args.requested_cells)} · "
            f"{solver['qualification']}"
        ),
    )
    plt.close(figure)
    args.panel.with_suffix(".png").unlink(missing_ok=True)
    row = {
        "case": part["case"],
        "requested_cells": part["requested_cells"],
        "realised_cells": part["realised_cells"],
        "terminal_residual": solver["terminal_fixed_point_residual"],
        "converged": solver["converged"],
        "qualification": solver["qualification"],
        "qualification_reason": solver["qualification_reason"],
        "termination": solver["termination"],
        "trips": solver["production_telemetry"]["trip_count"],
        "accepted_newton_promotions": solver["production_telemetry"].get(
            "accepted_newton_promotions"
        ),
        "target_current_a": solver["target_current_a"],
        "lane": part["lane"],
        "solve_wall_seconds": solver["solve_wall_seconds"],
        "row_wall_seconds": part["lane"]["elapsed_seconds"],
        "stage_wall_seconds": part["stage_wall_seconds"],
        "render_source": source,
        "banked_receipt": banked,
        "panel": str(args.panel),
    }
    args.output.write_text(json.dumps(row, indent=2, default=str) + "\n")
    print("CERTIFICATE_ROW " + json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()