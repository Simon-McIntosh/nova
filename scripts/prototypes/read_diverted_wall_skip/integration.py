"""Verify a pinned combined read against a pinned main baseline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from scripts.prototypes.read_diverted_wall_skip.job import PYTHON, execute, extract


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    scratch = Path(tempfile.mkdtemp(prefix="combined-read-", dir=os.environ["TMPDIR"]))
    receipt = dict(
        base=args.base,
        head=args.head,
        job=os.environ.get("SLURM_JOB_ID"),
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete=False,
        suites={},
        added_failures=[],
        wall_counts=[],
    )
    destination = run / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2))
    print("PROVENANCE", json.dumps(receipt), flush=True)
    print("SCRATCH", scratch, flush=True)
    try:
        os.environ["TMPDIR"] = str(scratch)
        trees = {}
        for arm, revision in (("base", args.base), ("head", args.head)):
            tree = scratch / arm
            tree.mkdir()
            extract(root, revision, tree)
            trees[arm] = tree
        tasks = []
        for filename in (
            "test_tracked_read.py",
            "test_support_read.py",
            "test_read_tie_break.py",
        ):
            for arm, revision in (("base", args.base), ("head", args.head)):
                path = "tests/" + filename
                command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", path]
                tasks.append(
                    (trees[arm], command, run / f"{arm}-{filename}.log", revision)
                )
        path = "tests/test_read_wall_skip.py"
        command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", path]
        tasks.append(
            (trees["head"], command, run / "head-test_read_wall_skip.py.log", args.head)
        )
        command = [
            PYTHON,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "-s",
            path,
            "-k",
            "wall_execution",
        ]
        tasks.append((trees["head"], command, run / "wall-controls.log", args.head))
        cpus = sorted(os.sched_getaffinity(0))
        groups = [",".join(map(str, cpus[index::2])) for index in range(2)]

        def lane(index):
            return [execute(*task, groups[index]) for task in tasks[index::2]]

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lane, range(2)))
        print("OUTCOMES", outcomes, flush=True)
        for tree, command, log_path, revision in tasks:
            text = log_path.read_text()
            status = int(re.search(r"EXIT=(-?\d+)\s*$", text)[1])
            failure_ids = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)
            passed = re.search(r"(\d+) passed", text)
            skipped = re.search(r"(\d+) skipped", text)
            receipt["suites"][log_path.stem] = dict(
                revision=revision,
                command="PYTHONPATH=. JAX_PLATFORMS=cpu " + " ".join(command),
                exit_status=status,
                log_path=str(log_path),
                completed=bool(
                    re.search(r"\d+ (?:passed|failed|errors?|skipped)", text)
                ),
                failure_count=len(failure_ids),
                failure_ids=failure_ids,
                passed=int(passed[1]) if passed else 0,
                skipped=int(skipped[1]) if skipped else 0,
            )
        for filename in (
            "test_tracked_read.py",
            "test_support_read.py",
            "test_read_tie_break.py",
            "test_read_wall_skip.py",
        ):
            before = receipt["suites"].get("base-" + filename, {})
            after = receipt["suites"]["head-" + filename]
            receipt["added_failures"].extend(
                sorted(set(after["failure_ids"]) - set(before.get("failure_ids", [])))
            )
        counter_log = (run / "wall-controls.log").read_text()
        receipt["wall_counts"] = [
            int(value)
            for value in re.findall(
                r"WALL_EXECUTIONS[^\n]+ (\d+)\s*$", counter_log, re.M
            )
        ]
        receipt["complete"] = all(
            suite["completed"] for suite in receipt["suites"].values()
        )
        destination.write_text(json.dumps(receipt, indent=2))
        print("RESULT", json.dumps(receipt), flush=True)
        if (
            not receipt["complete"]
            or receipt["added_failures"]
            or receipt["wall_counts"] != [0, 1, 1, 1]
            or any(suite["exit_status"] for suite in receipt["suites"].values())
        ):
            raise SystemExit(1)
    finally:
        print("REMOVE_SCRATCH", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
