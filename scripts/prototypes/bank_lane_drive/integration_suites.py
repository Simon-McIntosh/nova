"""Measure a pinned integration and challenge its contraction-rebuild call."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from scripts.prototypes.bank_lane_drive.suites import run
from scripts.prototypes.bank_warm_predictor.suites import snapshot

COMMON = (
    "tests/test_bank_drive_continuation.py",
    "tests/test_bank_solve_driver.py",
    "tests/test_tracked_read.py",
)
LANES = "tests/test_bank_lane_drive.py"


def measure(*args):
    receipt = run(*args)
    output = Path(receipt["log_path"]).read_text()
    errors = [
        line.removeprefix("ERROR ").split(" - ")[0]
        for line in output.splitlines()
        if line.startswith("ERROR ")
    ]
    receipt["failure_ids"] = sorted(set(receipt["failure_ids"]) | set(errors))
    if receipt["exit_status"] and not receipt["failure_ids"]:
        receipt["failure_ids"] = [f"runner-exit:{receipt['exit_status']}"]
    receipt["failure_count"] = len(receipt["failure_ids"])
    receipt["completed"] = bool(
        re.search(r"\b\d+ (passed|failed|errors?|skipped)\b", output)
    )
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    candidate = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    for path in (
        "nova/equilibrium/drive_continuation.py",
        "nova/equilibrium/support_read.py",
    ):
        subprocess.run(
            [
                "git",
                "-C",
                str(args.root),
                "diff",
                "--exit-code",
                args.head,
                candidate,
                "--",
                path,
            ],
            check=True,
        )
    scratch = Path(
        tempfile.mkdtemp(prefix="combined-drive-gates-", dir=os.environ["TMPDIR"])
    )
    receipts = {}
    destination = args.run / "integration-suite-receipts.json"
    try:
        for arm, revision, files in (
            ("base", args.base, COMMON),
            ("head", args.head, (*COMMON, LANES)),
            ("candidate", candidate, (LANES,)),
        ):
            tree = scratch / arm
            snapshot(args.root, revision, tree)
            for target in files:
                key = arm + "-" + Path(target).stem
                receipts[key] = measure(
                    tree,
                    revision,
                    target,
                    args.run / ("integration-" + key + ".log"),
                    scratch / key,
                )
                destination.write_text(json.dumps(receipts, indent=2))
        tree = scratch / "two-item-mutant"
        snapshot(args.root, candidate, tree)
        path = tree / "nova/equilibrium/support_read.py"
        source = path.read_text()
        start = source.index(
            "            def rebuild_trial(_):", source.index("def solve_batch(")
        )
        end = source.index("            correction, checked =", start)
        branch = source[start:end]
        marker = "jax.vmap(one_fresh)((state, step_support, lane_indices))"
        assert branch.count(marker) == 1
        branch = branch.replace(marker, "jax.vmap(one_fresh)((state, step_support))")
        path.write_text(source[:start] + branch + source[end:])
        key = "two-item-mutant"
        receipts[key] = measure(
            tree,
            candidate + " two-item contraction rebuild mutant",
            LANES + "::test_incoming_factor_rebuild_uses_lane_profiles",
            args.run / ("integration-" + key + ".log"),
            scratch / "mutant-temp",
        )
        added = {}
        for target in COMMON:
            name = Path(target).stem
            before, after = receipts["base-" + name], receipts["head-" + name]
            assert before["completed"] and after["completed"]
            added[name] = sorted(set(after["failure_ids"]) - set(before["failure_ids"]))
        for key in ("head-test_bank_lane_drive", "candidate-test_bank_lane_drive"):
            assert receipts[key]["completed"]
            added[key] = receipts[key]["failure_ids"]
        receipts["delta"] = added
        destination.write_text(json.dumps(receipts, indent=2))
        assert not any(added.values()), added
        assert receipts["two-item-mutant"]["exit_status"] != 0
        assert receipts["two-item-mutant"]["failure_ids"]
        print(
            "SUMMARY integration: zero added failures; two-item rebuild mutant refused",
            flush=True,
        )
    finally:
        print("Removing job-owned scratch:", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
