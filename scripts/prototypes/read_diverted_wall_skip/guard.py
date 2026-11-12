"""Reproduce an unsafe skip hint or verify its guarded head in fresh processes."""

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
    parser.add_argument("mode", choices=("reproduce", "head"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        revision=args.revision,
        job=os.environ.get("SLURM_JOB_ID"),
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete=False,
        suites={},
        wall_counts=[],
    )
    destination = run / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2))
    scratch = Path(tempfile.mkdtemp(prefix="guard-read-", dir=os.environ["TMPDIR"]))
    try:
        os.environ["TMPDIR"] = str(scratch)
        tree = scratch / "source"
        tree.mkdir()
        extract(root, args.revision, tree)
        commands = []
        if args.mode == "reproduce":
            source = (tree / "nova/equilibrium/topology.py").read_text()
            assert "if skip_wall:\n        boundary, level, wall_stable = (" in source
            print("UNGUARDED_SKIP_PRESENT=True", flush=True)
            commands.append(
                ("unguarded-skip", ["tests/test_read_wall_skip.py", "-k", "skip_hint"])
            )
        else:
            for filename in (
                "test_read_wall_skip.py",
                "test_tracked_read.py",
                "test_support_read.py",
            ):
                commands.append((filename, ["tests/" + filename]))
            commands.append(
                (
                    "wall-controls",
                    ["-s", "tests/test_read_wall_skip.py", "-k", "wall_execution"],
                )
            )
        tasks = [
            (
                tree,
                [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", *paths],
                run / (name + ".log"),
                args.revision,
            )
            for name, paths in commands
        ]
        cpus = sorted(os.sched_getaffinity(0))
        groups = [",".join(map(str, cpus[index::2])) for index in range(2)]

        def lane(index):
            return [execute(*task, groups[index]) for task in tasks[index::2]]

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lane, range(2)))
        for _, command, log, revision in tasks:
            text = log.read_text()
            ids = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)
            passed = re.search(r"(\d+) passed", text)
            skipped = re.search(r"(\d+) skipped", text)
            receipt["suites"][log.stem] = dict(
                revision=revision,
                command="PYTHONPATH=. JAX_PLATFORMS=cpu " + " ".join(command),
                exit_status=int(re.search(r"EXIT=(-?\d+)\s*$", text)[1]),
                log_path=str(log),
                completed=bool(
                    re.search(r"\d+ (?:passed|failed|errors?|skipped)", text)
                ),
                failure_count=len(ids),
                failure_ids=ids,
                passed=int(passed[1]) if passed else 0,
                skipped=int(skipped[1]) if skipped else 0,
            )
        if args.mode == "head":
            receipt["wall_counts"] = [
                int(v)
                for v in re.findall(
                    r"WALL_EXECUTIONS[^\n]+ (\d+)\s*$",
                    (run / "wall-controls.log").read_text(),
                    re.M,
                )
            ]
        receipt["complete"] = all(s["completed"] for s in receipt["suites"].values())
        destination.write_text(json.dumps(receipt, indent=2))
        print("RESULT", json.dumps(receipt), flush=True)
        if args.mode == "reproduce":
            assert receipt["suites"]["unguarded-skip"]["failure_count"] == 2
            assert outcomes == [[1], []]
        else:
            assert all(code == 0 for lane_codes in outcomes for code in lane_codes)
            assert receipt["wall_counts"] == [0, 1, 1, 1]
        assert receipt["complete"]
    finally:
        print("REMOVE_SCRATCH", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
