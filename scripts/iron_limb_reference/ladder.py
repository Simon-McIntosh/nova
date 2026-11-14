"""Extend the six-limb mesh ladder and judge shrinking changes per measure."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import numpy as np

from scripts.iron_limb_reference.controls import verified_prior
from scripts.iron_limb_reference.measure import (
    metrics_difference,
    solve_resolution,
    write_json,
)


def convergence_verdict(levels, vessel_count):
    """Require mesh ratios, shrinking increments and a small final increment."""
    if len(levels) < 3:
        raise ValueError("convergence needs at least three mesh resolutions")
    sizes = np.asarray([r["size_m"] for r in levels])
    ratios = sizes[:-1] / sizes[1:]
    differences = [
        [
            metrics_difference(a, b, vessel_count)
            for a, b in zip(coarse["rows"], fine["rows"])
        ]
        for coarse, fine in zip(levels[:-1], levels[1:])
    ]
    rows = []
    for index, mu in enumerate((50.0, 200.0)):
        changes = [r[index] for r in differences]
        values = np.asarray(
            [
                [
                    r["column_flux_relative"],
                    r["vessel_iron_relative"],
                    *r["outside_total_relative"],
                ]
                for r in changes
            ]
        )
        names = [
            "flux",
            "vessel",
            *[f"outside_{k}" for k in range(values.shape[1] - 2)],
        ]
        measures = {}
        for column, name in enumerate(names):
            increments = values[:, column]
            shrinking = bool(np.all(np.diff(increments) < 0))
            small_final = bool(increments[-1] < 0.02)
            measures[name] = dict(
                changes_relative=increments.tolist(),
                shrinking=shrinking,
                final_under_two_percent=small_final,
                passed=shrinking and small_final,
            )
        rows.append(
            dict(
                mu=mu,
                measures=measures,
                passed=all(v["passed"] for v in measures.values()),
            )
        )
    spacing_passed = bool(np.all(np.isfinite(ratios)) and np.all(ratios + 1e-12 >= 1.3))
    return dict(
        sizes_m=sizes.tolist(),
        refinement_ratios=ratios.tolist(),
        spacing_passed=spacing_passed,
        rows=rows,
        passed=spacing_passed and all(r["passed"] for r in rows),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    prior = verified_prior(args.case, args.prior)
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest(),
        host=socket.gethostname(),
        prior_receipt=str(args.prior),
        reference=list(prior["reference"]),
        completed=False,
        large_coarse_step_cause=(
            "unexplained: existing receipt does not establish a face-level "
            "or gap-resolution cause"
        ),
    )
    summary = args.output / "summary.json"
    write_json(summary, record)
    for size in (0.1, 0.075):
        record["reference"].append(solve_resolution(case, size, args.output))
        write_json(summary, record)
        gc.collect()
    record["all_steps"] = [
        dict(
            coarse_m=a["size_m"],
            fine_m=b["size_m"],
            changes=[
                metrics_difference(x, y, case["vessel_count"])
                for x, y in zip(a["rows"], b["rows"])
            ],
        )
        for a, b in zip(record["reference"][:-1], record["reference"][1:])
    ]
    window = [r for r in record["reference"] if r["size_m"] in (0.18, 0.13, 0.1, 0.075)]
    if len(window) != 4:
        raise ValueError("the required four-mesh convergence window is incomplete")
    record["convergence"] = convergence_verdict(window, case["vessel_count"])
    record["finest_reference"] = record["reference"][-1]
    record["completed"] = True
    record["passed"] = record["convergence"]["passed"]
    write_json(summary, record)
    print("LADDER_SUMMARY " + json.dumps(record["convergence"]), flush=True)
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
