"""Pair the driver test files and challenge both lane-specific guards."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from scripts.prototypes.bank_lane_drive.suites import FILES, run
from scripts.prototypes.bank_warm_predictor.suites import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    scratch = Path(
        tempfile.mkdtemp(prefix="lane-support-gates-", dir=os.environ["TMPDIR"])
    )
    receipts = {}
    receipt_path = args.run / "review-suite-receipts.json"
    try:
        for arm, revision in [("base", args.base), ("head", head)]:
            tree = scratch / arm
            snapshot(args.root, revision, tree)
            for target in FILES:
                key = arm + "-" + Path(target).stem
                receipts[key] = run(
                    tree,
                    revision,
                    target,
                    args.run / ("review-" + key + ".log"),
                    scratch / key,
                )
                receipt_path.write_text(json.dumps(receipts, indent=2))
        for mutation in ("shared-step", "cached-profile"):
            tree = scratch / mutation
            snapshot(args.root, head, tree)
            if mutation == "shared-step":
                path = tree / "nova/equilibrium/drive_continuation.py"
                source = path.read_text()
                marker = "        active = ~done & ~stopped\n"
                assert source.count(marker) == 1
                source = source.replace(
                    marker,
                    marker
                    + "        sub_step = jnp.full_like(sub_step, jnp.min(sub_step))\n",
                )
                target = FILES[2] + "::test_stiff_lane_does_not_shrink_easy_lane"
            else:
                path = tree / "nova/equilibrium/support_read.py"
                source = path.read_text()
                start = source.index(
                    "        def one_with_factor(items):",
                    source.index("def solve_batch("),
                )
                end = source.index("        def one_fresh(items):", start)
                branch = source[start:end]
                assert branch.count("lane_profile(lane)") == 1
                branch = branch.replace("lane_profile(lane)", "lane_profile(0)")
                source = source[:start] + branch + source[end:]
                target = FILES[2] + (
                    "::test_solve_batch_maps_lane_profile_into_residual_and_booking[True]"
                )
            path.write_text(source)
            key = mutation + "-mutant"
            receipts[key] = run(
                tree,
                head + " " + mutation + " mutant",
                target,
                args.run / ("review-" + key + ".log"),
                scratch / (key + "-temp"),
            )
            receipt_path.write_text(json.dumps(receipts, indent=2))
            assert receipts[key]["exit_status"] != 0 and receipts[key]["failure_ids"]
        for target in FILES:
            name = Path(target).stem
            before, after = receipts["base-" + name], receipts["head-" + name]
            assert before["completed"] and after["completed"]
            assert set(after["failure_ids"]) <= set(before["failure_ids"])
        print(
            "SUMMARY paired files: zero added failures; "
            "shared-step and cached-profile mutants refused",
            flush=True,
        )
    finally:
        print("Removing job-owned scratch:", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
