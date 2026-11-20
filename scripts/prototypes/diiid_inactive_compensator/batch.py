"""Run every cohort row masked and unmasked inside one GPU allocation.

Each row is an independent solve, so the allocation shards the workload across
its cards rather than submitting one job per row, and each arm writes its
receipt as it lands. The unmasked arm is the matched control: same tree, same
seeds, same source inputs, with only the inactive-image rule restored in the
measurement process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))
ARMS = ("masked", "unmasked")


def case_name(frame, branch):
    return f"frame-{frame}-{branch}"


def tasks():
    """Order the cohort row by row, so a matched pair lands adjacent."""
    return [(arm, frame, branch) for frame, branch in ROWS for arm in ARMS]


def completed(folder):
    """A row has landed only when its solve and its image read both wrote."""
    log = folder / "row.log"
    if not log.is_file():
        return False
    for name in ("receipt.json", "core-analysis.json", "trips.json"):
        if not (folder / name).is_file():
            return False
    return log.read_text().endswith("EXIT=0\n")


def row_environment(tree, arm, slot, slots):
    """Return the child environment, identical across arms but for the rule."""
    values = dict(
        os.environ,
        PYTHONPATH=str(tree),
        NOVA_LANE_RUNG="h200",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        JAX_ENABLE_COMPILATION_CACHE="0",
        JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="none",
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(0.9 / slots),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        CUDA_VISIBLE_DEVICES=str(slot),
    )
    values.pop("NOVA_INACTIVE_MASK_OFF", None)
    if arm == "unmasked":
        values["NOVA_INACTIVE_MASK_OFF"] = "1"
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--cards", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pending = []
    for arm, frame, branch in tasks():
        folder = args.output / arm / case_name(frame, branch)
        if completed(folder):
            print("RETAINED", arm, case_name(frame, branch), flush=True)
            continue
        pending.append((arm, frame, branch, folder))
    plan = dict(
        slots=args.slots,
        cards=args.cards,
        pending=[
            f"{arm}:{case_name(frame, branch)}" for arm, frame, branch, _ in pending
        ],
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print("PLAN " + json.dumps(plan), flush=True)
    if args.plan_only:
        return
    results = {}
    path = args.output / "results.json"
    queue = list(pending)
    running = {}
    while queue or running:
        while queue and len(running) < args.slots:
            arm, frame, branch, folder = queue.pop(0)
            folder.mkdir(parents=True, exist_ok=True)
            slot = len(running) % max(args.cards, 1)
            command = [
                sys.executable,
                str(HERE / "measure.py"),
                "--tree",
                str(args.tree),
                "--revision",
                args.revision,
                "--frame",
                str(frame),
                "--branch",
                branch,
                "--output",
                str(folder),
                "--trace-helper",
                str(args.trace_helper),
                # The interior arm: its table ends inside psi_norm = 1, which
                # the finite-edge flux function requires because the boundary
                # value is EFIT's own read rather than an appended knot.
                "--arm",
                "base",
            ]
            stream = (folder / "row.log").open("w")
            stream.write(
                f"# revision: {args.revision}; tree: {args.tree}; "
                f"module: {args.tree}/nova/equilibrium/constraint.py; "
                f"cwd: {args.tree}; arm: {arm}; command: {shlex.join(command)}\n"
            )
            stream.flush()
            process = subprocess.Popen(
                command,
                cwd=args.tree,
                env=row_environment(args.tree, arm, slot, args.slots),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            running[process] = (arm, frame, branch, folder, stream)
            print(
                "STARTED",
                arm,
                case_name(frame, branch),
                "card",
                slot,
                flush=True,
            )
        time.sleep(5)
        for process in list(running):
            if process.poll() is None:
                continue
            arm, frame, branch, folder, stream = running.pop(process)
            stream.write(f"EXIT={process.returncode}\n")
            stream.close()
            name = case_name(frame, branch)
            results[f"{arm}:{name}"] = dict(
                revision=args.revision,
                exit_status=process.returncode,
                log_path=str(folder / "row.log"),
            )
            receipt = folder / "receipt.json"
            if process.returncode == 0 and receipt.is_file():
                results[f"{arm}:{name}"].update(json.loads(receipt.read_text()))
            path.write_text(json.dumps(results, indent=2, default=str) + "\n")
            print("ROW", arm, name, process.returncode, flush=True)
    unreceipted = [name for name, item in results.items() if "converged" not in item]
    if unreceipted:
        raise SystemExit(f"rows finished without a receipt: {unreceipted}")
    if any(item["exit_status"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
