"""Run paired files in fresh processes and refuse unguarded factor reuse."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from scripts.prototypes.bank_warm_predictor.suites import snapshot, run, FILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    scratch = Path(tempfile.mkdtemp(prefix="factor-gates-", dir=os.environ["TMPDIR"]))
    receipts = {}
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
                    args.run / (key + ".log"),
                    scratch / (key + "-temp"),
                )
                (args.run / "suite-receipts.json").write_text(
                    json.dumps(receipts, indent=2)
                )
        tree = scratch / "mutant"
        snapshot(args.root, head, tree)
        path = tree / "nova/equilibrium/support_read.py"
        source = path.read_text()
        marker = "& (jnp.max(steps) == 0)"
        assert source.count(marker) == 1
        path.write_text(
            source.replace(
                marker, "& jnp.asarray(False) " + marker
            )
        )
        target = FILES[1] + "::test_carried_factor_first_update_is_guarded"
        receipts["mutant"] = run(
            tree,
            head + " unguarded reuse",
            target,
            args.run / "unguarded-reuse-mutant.log",
            scratch / "mutant-temp",
        )
        (args.run / "suite-receipts.json").write_text(json.dumps(receipts, indent=2))
        assert receipts["mutant"]["exit_status"] != 0
        assert any("4.0-1" in x for x in receipts["mutant"]["failure_ids"])
        for target in FILES:
            name = Path(target).stem
            assert (
                receipts["base-" + name]["completed"]
                and receipts["head-" + name]["completed"]
            )
            assert set(receipts["head-" + name]["failure_ids"]) <= set(
                receipts["base-" + name]["failure_ids"]
            )
        print(
            "SUMMARY paired files have zero added failures; "
            "unguarded reuse mutant refused",
            flush=True,
        )
    finally:
        print("Removing job-owned scratch:", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
