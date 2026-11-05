"""Assemble completed paired receipts and the worker's durable delivery record."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from scripts.prototypes.bank_solve_cycle.receipts import assemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    manifest = run / "manifest.md"
    lines = manifest.read_text().splitlines()
    fields = dict(line.split(":", 1) for line in lines if ":" in line)
    base = fields["orientation_base_sha"].strip()
    header = (
        (run / "paired/rings-head-test_bank_solve_cycle.log")
        .read_text()
        .splitlines()[0]
    )
    revision = header.split()[2]
    names = ("test_bank_solve_cycle", "test_bank_solve_driver", "test_support_read")
    before = assemble(
        run, ["cpu-base-" + name for name in names], base, "baseline-suite.log"
    )
    after = assemble(
        run,
        ["paired/rings-head-" + name for name in names]
        + ["paired/rings-head-booking", "paired/scaling-head"],
        revision,
        "after-suite.log",
    )
    negative = run / "paired/negative-control.log"
    negative_text = negative.read_text()
    assert negative_text.rstrip().endswith("EXIT=1")
    assert "FAILED " in negative_text and "test_sixteen_lane_scaling" in negative_text
    added = set(after["failure_ids"]) - set(before["failure_ids"])
    timings = json.loads((run / "paired/head-timings.json").read_text())
    ratio = timings["16:assembly"] / timings["1:assembly"]
    assert ratio <= 4
    precision = json.loads((run / "paired/precision.json").read_text())
    assert max(row["residual"] for row in precision) <= 1e-10
    parity = (run / "paired/rings-head-booking.log").read_text()
    errors = []
    for match in re.finditer(r"MATRIX_RELATIVE_ERRORS (?:1|16) (\[[^\n]+\])", parity):
        errors.extend(json.loads(match[1]))
    assert len(errors) == 17 and max(errors) <= 1e-12
    base_log = (run / "paired/measure-base.log").read_text()
    imported = next(
        json.loads(line.partition(" ")[2])
        for line in base_log.splitlines()
        if line.startswith("HEADER ")
    )
    test_header = (run / "paired/scaling-head.log").read_text().splitlines()[0]
    tree = test_header.split(" tree: ", 1)[1].split(" command: ", 1)[0]
    command = test_header.split(" command: ", 1)[1]
    updates = {
        "status": "in-progress",
        "wait_condition": "",
        "wait_probe": "",
        "wait_terminal": "",
        "resume_brief": "",
        "checkpoint": (
            "Declared paired gates completed; final evidence commit and "
            "manifest audit remain."
        ),
        "commits": json.dumps(
            subprocess.check_output(
                ["git", "rev-list", "--reverse", base + "..HEAD"], text=True
            ).split()
        ),
        "changed_paths": json.dumps(
            subprocess.check_output(
                ["git", "diff", "--name-only", base, "HEAD"], text=True
            ).splitlines()
        ),
        "tests": (
            f"PYTHONPATH={tree} BOOKING_TIMINGS={run}/paired/head-timings.json "
            + command
        ),
        "test_logs": json.dumps(
            [
                before["log_path"],
                after["log_path"],
                str(negative),
                str(run / "paired/measure-base.log"),
                str(run / "paired/measure-head.log"),
                str(run / "paired/report.log"),
            ]
        ),
        "baseline_suite": json.dumps(before),
        "after_suite": json.dumps(after),
        "measurement_module": imported["module"],
        "measurement_cwd": imported["cwd"],
        "negative_control_log": str(negative),
        "negative_control_note": (
            "Scratch mutation restores the lax.map lane loop; its measured "
            "sixteen-lane scaling fails the assertion the unmutated head passes."
        ),
        "landing": (
            "docs/evidence/fragments/sub-millisecond-forward-path/"
            "bank-step-booking-batch.html#bank-step-booking-batch-landing — "
            "vectorized selected buckets and live rings; Jacobian "
            f"{timings['1:assembly']:.3f}/{timings['16:assembly']:.3f} ms at "
            f"1/16 lanes ({ratio:.3f}x); CPU matrix relative error <= "
            f"{max(errors):.6e}; three-node geometry capacity refuses overflow; "
            f"added suite failures: {sorted(added)}."
        ),
        "artifacts": json.dumps(
            {
                "figure": (
                    "docs/figures/sub-millisecond-forward-path/"
                    "bank-step-booking-batch/jacobian-wall.svg"
                ),
                "paired_receipts": str(run / "paired"),
                "scaling": ratio,
                "matrix_relative_error": max(errors),
            }
        ),
        "evidence_inputs": (
            f"Base {base}; measured product {revision}; selected-population loops "
            "remove bucket select lowering and empty ring quadrature; three "
            "nodes/eighteen channels verified across every seed lane. "
            "Final H200 allocation 1306401; final sirius allocation 1306402; "
            "baseline suite allocation 1306389. Earlier exploratory receipts "
            "and scaling refusal remain in the parent run directory."
        ),
        "blockers": "Added CPU suite failures: " + ", ".join(sorted(added))
        if added
        else "none",
        "outside_scope": "none",
    }
    output = [
        key + ": " + updates[key] if (key := line.split(":", 1)[0]) in updates else line
        for line in lines
    ]
    for name in (
        "goal_fidelity",
        "evidence",
        "scope_discipline",
        "durability",
        "fit",
        "reuse",
    ):
        output = [
            line for line in output if not line.startswith("rubric." + name + ":")
        ]
        state = "answered"
        if name == "goal_fidelity" and added:
            state = "CPU restart convergence regression remains: " + ", ".join(added)
        output.append("rubric." + name + ": " + state)
    manifest.write_text("\n".join(output) + "\n")
    print("BASELINE", json.dumps(before), flush=True)
    print("AFTER", json.dumps(after), flush=True)
    print("MATRIX_MAX_RELATIVE_ERROR", max(errors), flush=True)


if __name__ == "__main__":
    main()
