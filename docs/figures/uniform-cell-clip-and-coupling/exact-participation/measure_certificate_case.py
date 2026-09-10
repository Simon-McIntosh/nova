"""Measure one production Solovev certificate row in an isolated process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import benchmarks.solovev_certificate as certificate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("requested_cells", type=int)
    parser.add_argument("baseline_residual", type=float)
    parser.add_argument(
        "--criterion",
        choices=("converged", "no-worse", "record"),
        default="record",
    )
    arguments = parser.parse_args()

    output_root = Path(__file__).parent
    certificate.PART_ROOT = output_root / "certificate-parts"
    certificate.FIGURE_ROOT = output_root / "certificate-figures"

    print("SPLINE_CHAIN_STATIC_SAMPLES=128", flush=True)
    row = certificate._measure(arguments.case, arguments.requested_cells)
    residual = row["solver"]["terminal_fixed_point_residual"]
    ratio = residual / arguments.baseline_residual
    if arguments.criterion == "converged":
        passed = residual < 1.0e-12
    elif arguments.criterion == "no-worse":
        passed = residual <= arguments.baseline_residual
    else:
        passed = True

    print(
        json.dumps(
            {
                "baseline_residual": arguments.baseline_residual,
                "case": row["case"],
                "criterion": arguments.criterion,
                "passed": passed,
                "qualification": row["solver"]["qualification"],
                "requested_cells": row["requested_cells"],
                "residual_ratio_to_baseline": ratio,
                "terminal_residual": residual,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
