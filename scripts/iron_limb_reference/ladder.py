"""Evaluate mesh uncertainty and stored controls for the six-limb reference."""

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
    """Require four spaced meshes and bound field uncertainty, stating flux scatter.

    Each uncertainty is the largest of the two relative changes between the
    three finest meshes. Flux is reported without a tolerance; vessel and
    each outside-probe uncertainty must be strictly below two percent.
    """
    if len(levels) < 4:
        raise ValueError("convergence needs at least four mesh resolutions")
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
        values = np.asarray(
            [
                [
                    step[index]["column_flux_relative"],
                    step[index]["vessel_iron_relative"],
                    *step[index]["outside_total_relative"],
                ]
                for step in differences
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
            uncertainty = float(np.max(increments[-2:]))
            valid = bool(np.all(np.isfinite(increments)) and np.all(increments >= 0))
            limit = None if name == "flux" else 0.02
            measures[name] = dict(
                changes_relative=increments.tolist(),
                uncertainty_relative=uncertainty,
                uncertainty_limit_relative=limit,
                passed=valid and (limit is None or uncertainty < limit),
            )
        rows.append(
            dict(
                mu=mu,
                measures=measures,
                passed=all(v["passed"] for v in measures.values()),
            )
        )
    spacing_passed = bool(
        np.all(np.isfinite(sizes))
        and np.all(sizes > 0)
        and np.all(ratios + 1e-12 >= 1.3)
    )
    return dict(
        sizes_m=sizes.tolist(),
        uncertainty_sizes_m=sizes[-3:].tolist(),
        refinement_ratios=ratios.tolist(),
        spacing_passed=spacing_passed,
        rows=rows,
        passed=spacing_passed and all(r["passed"] for r in rows),
    )


def annulus_comparison_verdict(rows):
    """Allow the measured comparator step where it exceeds two percent."""
    measures = []
    for row in rows:
        limit = max(0.02, row["axisymmetric_last_step_relative"])
        difference = row["difference_relative"]
        valid = bool(np.isfinite(limit) and np.isfinite(difference))
        measures.append(
            dict(
                mu=row["mu"],
                measure=row["measure"],
                difference_relative=difference,
                limit_relative=limit,
                axisymmetric_last_step_relative=row["axisymmetric_last_step_relative"],
                comparator=row["comparator"],
                passed=valid and difference < limit,
            )
        )
    return dict(measures=measures, passed=all(v["passed"] for v in measures))


def update_stored_receipt(path, vessel_count, annulus_path=None, symmetry_path=None):
    """Replace the verdict atomically, retaining all measured values verbatim."""
    from scripts.iron_limb_reference.comparison import comparison_rows

    record = json.loads(path.read_text())
    if not record.get("completed"):
        raise ValueError("the stored mesh receipt is incomplete")
    measured = {
        key: record[key] for key in ("reference", "all_steps", "finest_reference")
    }
    fingerprint = hashlib.sha256(
        json.dumps(measured, sort_keys=True).encode()
    ).hexdigest()
    record.setdefault("measurement_verdict_passed", record["passed"])
    record["convergence"] = convergence_verdict(record["reference"][-4:], vessel_count)
    record["controls"] = {}
    if annulus_path is not None:
        annulus = json.loads(annulus_path.read_text())
        record["controls"]["annulus"] = annulus_comparison_verdict(
            comparison_rows(annulus, vessel_count)
        )
        record["controls"]["annulus"]["receipt"] = str(annulus_path)
    if symmetry_path is not None:
        symmetry = json.loads(symmetry_path.read_text())
        if not symmetry.get("completed"):
            raise ValueError("the stored symmetry receipt is incomplete")
        record["controls"]["symmetry"] = dict(
            receipt=str(symmetry_path),
            comparison=symmetry["comparison"],
            passed=symmetry["passed"],
        )
    record["passed"] = record["convergence"]["passed"] and all(
        c["passed"] for c in record["controls"].values()
    )
    record["verdict_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    record["measured_values_sha256"] = fingerprint
    pending = path.with_suffix(".pending")
    if pending.exists():
        raise ValueError("receipt pending path already exists")
    write_json(pending, record)
    pending.replace(path)
    landed = json.loads(path.read_text())
    actual = {key: landed[key] for key in measured}
    if actual != measured or landed["convergence"] != record["convergence"]:
        raise RuntimeError("stored receipt readback differs")
    print(
        "STORED_VERDICT "
        + json.dumps(
            dict(
                passed=landed["passed"],
                convergence=landed["convergence"],
                controls=landed["controls"],
                measured_values_sha256=fingerprint,
            )
        ),
        flush=True,
    )
    return landed


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--vessel-count", type=int, default=16)
    parser.add_argument("--annulus-receipt", type=Path)
    parser.add_argument("--symmetry-receipt", type=Path)
    args = parser.parse_args()
    if args.receipt is not None:
        record = update_stored_receipt(
            args.receipt, args.vessel_count, args.annulus_receipt, args.symmetry_receipt
        )
        raise SystemExit(0 if record["passed"] else 1)
    if any(value is None for value in (args.case, args.prior, args.output)):
        parser.error("provide --receipt, or all of --case, --prior and --output")
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
