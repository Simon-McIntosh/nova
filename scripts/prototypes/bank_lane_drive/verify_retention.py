"""Refresh driver gates after changing rejected-result retention only."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from scripts.prototypes.bank_lane_drive.suites import run
from scripts.prototypes.bank_warm_predictor.suites import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    receipts_path = args.run / "suite-receipts.json"
    receipts = json.loads(receipts_path.read_text())
    scratch = Path(tempfile.mkdtemp(prefix="retained-state-", dir=os.environ["TMPDIR"]))
    try:
        tree = scratch / "head"
        snapshot(args.root, revision, tree)
        for name in ("test_bank_drive_continuation", "test_bank_lane_drive"):
            receipt = run(
                tree,
                revision,
                "tests/" + name + ".py",
                args.run / ("retention-" + name + ".log"),
                scratch / name,
            )
            receipts["head-" + name] = receipt
            receipts_path.write_text(json.dumps(receipts, indent=2))
            assert receipt["completed"] and receipt["exit_status"] == 0
        path = tree / "nova/equilibrium/drive_continuation.py"
        source = path.read_text()
        marker = "        active = ~done & ~stopped\n"
        assert source.count(marker) == 1
        path.write_text(
            source.replace(
                marker,
                marker
                + "        sub_step = jnp.full_like(sub_step, jnp.min(sub_step))\n",
            )
        )
        receipt = run(
            tree,
            revision + " shared-step mutant",
            "tests/test_bank_lane_drive.py::test_stiff_lane_does_not_shrink_easy_lane",
            args.run / "shared-step-mutant-retention.log",
            scratch / "mutant",
        )
        receipts["shared-step-mutant"] = receipt
        receipts_path.write_text(json.dumps(receipts, indent=2))
        assert receipt["exit_status"] != 0 and receipt["failure_ids"]
        print(
            "SUMMARY retained-state driver gates pass; shared-step mutant fails",
            flush=True,
        )
    finally:
        print("Removing job-owned scratch:", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
