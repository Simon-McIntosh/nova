"""Continue only incomplete file arms, preserving every earlier receipt."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import tempfile

from regressions import ROOT, extract, gate

SCOPED = (
    "tests/test_sampled_flux_batching.py",
    "tests/test_sampled_profile_core_hold.py",
)


def readable(record):
    if not record or record.get("completed") is not True:
        return False
    log = Path(record["log_path"])
    return log.is_file() and log.read_text().rstrip().endswith(
        "EXIT=" + str(record["exit_status"])
    )


def equivalent(revision, retained):
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
            retained,
            revision,
            "--",
            "nova",
            "tests",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("retained product and test trees differ from this arm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("base", "repaired"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.previous.read_text())
    files = json.loads(args.files.read_text())
    retained, pending, interruptions = {}, [], []
    for path in files:
        if path in SCOPED:
            continue
        key = args.arm + ":" + path
        record = previous.get(key)
        if readable(record):
            equivalent(args.revision, record["revision"])
            retained[key] = record
        else:
            pending.append(path)
            if record:
                interruptions.append(dict(path=path, record=record))
    plan = dict(
        arm=args.arm,
        revision=args.revision,
        retained=list(retained),
        pending=pending,
        interruptions=interruptions,
        max_concurrent_pytest=2,
        retained_product_and_tests_identical=True,
    )
    (args.output / "continuation-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    (args.output / "retained-results.json").write_text(
        json.dumps(retained, indent=2) + "\n"
    )
    results = dict(retained)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("CONTINUATION_PLAN " + json.dumps(plan), flush=True)
    with tempfile.TemporaryDirectory(
        prefix="nova-core-continuation-", dir=os.environ["TMPDIR"]
    ) as scratch:
        tree = Path(scratch) / "tree"
        extract(tree, args.revision, ROOT)
        scoped_key = args.arm + ":suite"
        scoped = previous.get(scoped_key)
        if readable(scoped) and scoped["revision"] == args.revision:
            results[scoped_key] = scoped
        else:
            paths = [path for path in SCOPED if (tree / path).is_file()]
            results[scoped_key] = gate(
                tree, args.revision, paths, args.output / "suite.log", Path(scratch)
            )
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for path in pending:
                folder = Path(scratch) / path.replace("/", "-")
                folder.mkdir()
                log = args.output / (path.replace("/", "-") + ".log")
                if log.exists():
                    raise FileExistsError(f"continuation would overwrite {log}")
                future = pool.submit(gate, tree, args.revision, [path], log, folder)
                futures[future] = args.arm + ":" + path
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                print("REGRESSION " + key + " " + json.dumps(results[key]), flush=True)


if __name__ == "__main__":
    main()
