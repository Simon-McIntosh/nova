"""Reconcile the base and head receipts into the node's table.

Each row reports convergence, terminal residual, terminal reason and the lower
X-point's distance to EFIT's own null, at both arms. Nothing is summed across
rows: a row that stops short of converged is reported as the state it reached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = (
    (137, "diverted", "137D"),
    (146, "diverted", "146D"),
    (146, "limited", "146L"),
    (179, "limited", "179L"),
)
ARMS = ("base", "head")


def termination_name(code):
    """The solver's own name for a termination code."""

    from nova.equilibrium.fixed_point import FixedPointTerminationReason

    try:
        return FixedPointTerminationReason(int(code)).name
    except (TypeError, ValueError):
        return f"unrecognised termination code {code}"


def row_record(folder):
    receipt = json.loads((folder / "receipt.json").read_text())
    built = json.loads((folder / "built-profile.json").read_text())
    solve = json.loads((folder / "solve-receipt.json").read_text())
    topology = receipt.get("terminal_topology") or {}
    distance = topology.get("lower_saddle_distance_m")
    return {
        "residual": receipt["residual"],
        "converged": bool(receipt["converged"]),
        "termination_reason": termination_name(solve.get("terminal_reason")),
        "termination_code": solve.get("terminal_reason"),
        "lower_saddle_distance_m": distance,
        "lower_x_rz_m": topology.get("lower_saddle_rz_m"),
        "efit_x_rz_m": receipt.get("efit_x_point_rz_m"),
        "support_cells": topology.get("support"),
        "boundary_class": topology.get("boundary_class"),
        "p_prime_at_boundary": built["p_prime_at_boundary"],
        "ff_prime_at_boundary": built["ff_prime_at_boundary"],
        "has_finite_edge": built["has_finite_edge"],
        "coordinate_last": built["coordinate_last"],
        "coordinate_count": built["coordinate_count"],
        "revision": built["revision"],
    }


EXTRACTION_LABEL = {"137D": "R137D", "146D": "R146D", "146L": "R146L", "179L": "R179L"}
RELATIVE_TOLERANCE = 1e-10


def boundary_agreement(extraction_path, table):
    """Compare the builder's psi_norm = 1 value with the committed extraction."""

    extraction = json.loads(extraction_path.read_text())["rows"]
    agreement = {}
    for label, arms in table.items():
        record = extraction[EXTRACTION_LABEL[label]]
        head = arms["head"]
        agreement[label] = {
            "extracted_p_prime": record["p_prime"],
            "extracted_ff_prime": record["ff_prime"],
            "builder_p_prime": head["p_prime_at_boundary"],
            "builder_ff_prime": head["ff_prime_at_boundary"],
        }
    return agreement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    args = parser.parse_args()
    table = {}
    for frame, branch, label in ROWS:
        table[label] = {
            arm: row_record(args.input / arm / f"frame-{frame}-{branch}")
            for arm in ARMS
        }
    agreement = boundary_agreement(args.extraction, table)
    args.output.write_text(
        json.dumps(
            {
                "rows": table,
                "boundary_agreement": agreement,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for label, arms in table.items():
        for arm, record in arms.items():
            print(
                "TABLE %s %s converged=%s residual=%.3e reason=%s lower_x=%s"
                % (
                    label,
                    arm,
                    record["converged"],
                    record["residual"],
                    record["termination_reason"],
                    ("%.6f m" % record["lower_saddle_distance_m"])
                    if record["lower_saddle_distance_m"] is not None
                    else "not read",
                ),
                flush=True,
            )
    for label, record in agreement.items():
        for quantity in ("p_prime", "ff_prime"):
            extracted = record[f"extracted_{quantity}"]
            builder = record[f"builder_{quantity}"]
            relative = abs(builder - extracted) / max(abs(extracted), 1.0)
            print(
                "BOUNDARY %s %s extracted=%.17g builder=%.17g relative=%.3e"
                % (label, quantity, extracted, builder, relative),
                flush=True,
            )
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
