"""Run every profile-extent row inside one GPU allocation.

Each row is an independent solve, so the allocation shards the cohort across its
cards rather than submitting one job per row. Every completed row is persisted
as it lands, so an expiry loses one row and not the run.
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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402

ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))


def case_name(frame, branch):
    return f"frame-{frame}-{branch}"


def tasks():
    """Order the cohort row by row, so a complete arm pair lands as early as it can."""

    return [(arm, frame, branch) for frame, branch in ROWS for arm in arms.ARMS]


def completed(folder):
    """A row counts as landed only when its measure and its image read both wrote a
    receipt, because the image read is the last step and a row cancelled inside it
    leaves a terminal array that looks finished but has no imaged state."""

    log = folder / "row.log"
    if not log.is_file():
        return False
    for name in ("receipt.json", "core-analysis.json", "shadow-image.json"):
        if not (folder / name).is_file():
            return False
    return log.read_text().endswith("EXIT=0\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
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
            card = len(running) % max(args.cards, 1)
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
                "--arm",
                arm,
            ]
            stream = (folder / "row.log").open("w")
            stream.write(
                f"# revision: {args.revision}; tree: {args.tree}; "
                f"module: {args.tree}/nova/equilibrium/fixed_point.py; "
                f"cwd: {args.tree}; command: {shlex.join(command)}\n"
            )
            stream.flush()
            environment = dict(
                os.environ,
                PYTHONPATH=str(args.tree),
                NOVA_LANE_RUNG="h200",
                XLA_PYTHON_CLIENT_MEM_FRACTION=str(0.9 / args.slots),
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
            )
            environment["CUDA_VISIBLE_DEVICES"] = str(card)
            process = subprocess.Popen(
                command,
                cwd=args.tree,
                env=environment,
            )
            running[process] = (arm, frame, branch, folder, stream)
            print("STARTED", arm, case_name(frame, branch), "card", card, flush=True)
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
                image = folder / "shadow-image.json"
                if image.is_file():
                    results[f"{arm}:{name}"].update(json.loads(image.read_text()))
            path.write_text(json.dumps(results, indent=2, default=str) + "\n")
            print("ROW", arm, name, process.returncode, flush=True)
    unreceipted = [name for name, item in results.items() if "converged" not in item]
    if unreceipted:
        raise SystemExit(f"rows finished without a receipt: {unreceipted}")
    if any(item["exit_status"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
