"""Audit paired saved states and distinguish test failures from process aborts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def test_arm(path):
    records = json.loads(path.read_text())
    details = {}
    for name, receipt in records.items():
        text = Path(receipt["log_path"]).read_text()
        assert text.startswith("# revision:")
        assert text.rstrip().endswith(f"EXIT={receipt['exit_status']}")
        summary_ids = re.findall(r"^(?:FAILED|ERROR) (tests/\S+)", text, re.MULTILINE)
        outcomes = re.findall(
            r"^(tests/\S+::\S+) (PASSED|FAILED|ERROR|SKIPPED)", text, re.MULTILINE
        )
        failed = sorted(
            set(summary_ids)
            | {name for name, status in outcomes if status in ("FAILED", "ERROR")}
        )
        aborts = re.findall(
            r"^(tests/\S+::\S+) Fatal Python error:", text, re.MULTILINE
        )
        details[name] = dict(
            **receipt,
            observed_failure_error_ids=failed,
            abort_at_ids=aborts,
            verbose_outcomes=dict(outcomes),
        )
    return details


def events(path):
    trip, stages, ladders, result = 0, [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("STAGE "):
            value = json.loads(line[6:])
            if value["stage"] == "linearization":
                stages.append(value)
        elif line.startswith("LADDER "):
            ladders.append(json.loads(line[7:]))
        elif line.startswith("INNER "):
            refused = [value for value in ladders if not value["accepted"]]
            result.append(
                dict(
                    trip=trip,
                    inner=json.loads(line[6:]),
                    global_linearization_indices=[value["index"] for value in stages],
                    first_refused_ladder=refused[0] if refused else None,
                    last_ladder=ladders[-1] if ladders else None,
                )
            )
            trip += 1
            stages, ladders = [], []
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    stages = json.loads((root / "job-results.json").read_text())
    assert len(stages) == 4 and all(
        stage["exit_status"] == 0 for stage in stages.values()
    )
    rows = json.loads((root / "measurement/comparison.json").read_text())
    assert len(rows) == 10
    paired = {(row["arm"], row["case"]): row for row in rows}
    comparisons = {}
    for case in sorted({row["case"] for row in rows}):
        base, head = (paired[arm, case] for arm in ("base", "head"))
        seed_hash = base["seed_admission"][0]["sha256"]
        assert seed_hash == head["seed_admission"][0]["sha256"]
        assert head["augmented_topology_reader_connected"] is True
        saved = []
        for row in (base, head):
            source = Path(row["source_path"])
            receipt = json.loads((source / "receipt.json").read_text())
            assert receipt["residual"] == row["residual"]
            assert (source / "row.log").read_text().rstrip().endswith("EXIT=0")
            with np.load(source / "terminal.npz") as archive:
                saved.append({key: archive[key] for key in archive.files})
        comparisons[case] = dict(
            seed_sha256=seed_hash,
            head_reader_connected=True,
            max_abs_state_difference_wb=float(
                np.max(np.abs(saved[0]["state"] - saved[1]["state"]))
            ),
            state_sha256=[
                hashlib.sha256(data["state"].tobytes()).hexdigest() for data in saved
            ],
            arms={},
        )
        for row in (base, head):
            source = Path(row["source_path"])
            comparisons[case]["arms"][row["arm"]] = dict(
                source_path=str(source),
                residual=row["residual"],
                converged=row["converged"],
                terminal_reason=row["terminal_reason"],
                terminal_topology=row["terminal_topology"],
                trip_topology=row["trip_topology"],
                warmup=row["warmup"],
                branch_refusals=row["branch_refusals"],
                stops=events(source / "row.log"),
            )
    cpu = {
        arm: test_arm(root / f"regression-{arm}/results.json")
        for arm in ("base", "head")
    }
    assert all(len(arm) == 23 for arm in cpu.values())
    ids = {
        arm: set(
            value
            for receipt in receipts.values()
            for value in receipt["observed_failure_error_ids"]
        )
        for arm, receipts in cpu.items()
    }
    forward = json.loads((root / "measurement/forward-regressions.json").read_text())
    assert all(value["completed"] for value in forward.values())
    gpu_ids = {arm: set(value["failure_ids"]) for arm, value in forward.items()}
    result = dict(
        comparisons=comparisons,
        cpu=cpu,
        cpu_delta=dict(
            common=sorted(ids["base"] & ids["head"]),
            head_only_observed=sorted(ids["head"] - ids["base"]),
            base_only_observed=sorted(ids["base"] - ids["head"]),
            incomplete={
                arm: [
                    name
                    for name, receipt in receipts.items()
                    if not receipt["completed"]
                ]
                for arm, receipts in cpu.items()
            },
            caveat=(
                "An id unobserved after a process abort is not a passing test "
                "or a demonstrated fix."
            ),
        ),
        gpu=forward,
        gpu_delta=dict(
            common=sorted(gpu_ids["base"] & gpu_ids["head"]),
            added=sorted(gpu_ids["head"] - gpu_ids["base"]),
            removed=sorted(gpu_ids["base"] - gpu_ids["head"]),
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                cases=len(comparisons),
                cpu_delta=result["cpu_delta"],
                gpu_delta=result["gpu_delta"],
            )
        )
    )


if __name__ == "__main__":
    main()
