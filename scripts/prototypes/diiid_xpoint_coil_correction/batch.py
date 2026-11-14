"""Run the measured-current cohort inside one GPU allocation.

The cohort, its case naming and its landing predicate belong to the
profile-extent runner and are imported from it rather than restated. What this
driver adds is the arm sequence that runner does not have: a measured arm per
row, a conductor fit over that arm's saved basis, and then the same row again
with the fitted error removed from its currents. Every completed row is
persisted as it lands, so an expiry loses one row and not the run.
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

from scripts.prototypes.diiid_xpoint_profile_extent.batch import (
    ROWS,
    case_name,
    completed,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def measure_command(args, frame, branch, folder, correction=None):
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
    ]
    if correction is not None:
        command += ["--correction", str(correction)]
    return command


def environment(args, card):
    value = dict(os.environ)
    parts = value.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") or ["0"]
    value["CUDA_VISIBLE_DEVICES"] = parts[card % len(parts)]
    value["PYTHONPATH"] = str(args.tree)
    value["NOVA_LANE_RUNG"] = args.rung
    value["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    value["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    value.pop("SLURM_JOB_ID", None)
    return value


def run_all(args, pending, results):
    running = {}
    while pending or running:
        while pending and len(running) < args.slots:
            name, command, folder = pending.pop(0)
            folder.mkdir(parents=True, exist_ok=True)
            stream = (folder / "row.log").open("w")
            stream.write(
                f"# revision: {args.revision}; tree: {args.tree}; "
                f"command: {shlex.join(command)}\n"
            )
            stream.flush()
            card = len(running) % max(args.slots, 1)
            process = subprocess.Popen(
                command, cwd=args.tree, env=environment(args, card)
            )
            running[process] = (name, folder, stream)
            print("STARTED", name, flush=True)
        time.sleep(5)
        for process in list(running):
            if process.poll() is None:
                continue
            name, folder, stream = running.pop(process)
            stream.write(f"EXIT={process.returncode}\n")
            stream.close()
            entry = results.setdefault(name, {})
            entry.update(
                exit_status=process.returncode, log_path=str(folder / "row.log")
            )
            if process.returncode == 0 and (folder / "receipt.json").is_file():
                entry.update(json.loads((folder / "receipt.json").read_text()))
                entry.update(json.loads((folder / "shadow-image.json").read_text()))
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2, default=str) + "\n"
            )
            print("ROW", name, process.returncode, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--rung", default="h200")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    if (args.output / "results.json").is_file():
        results = json.loads((args.output / "results.json").read_text())
    basis = []
    for frame, branch in ROWS:
        name = case_name(frame, branch)
        folder = args.output / "base" / name
        if not completed(folder):
            basis.append(
                (f"basis:{name}", measure_command(args, frame, branch, folder), folder)
            )
    run_all(args, basis, results)
    fits = []
    for frame, branch in ROWS:
        name = case_name(frame, branch)
        folder = args.output / "fits" / name
        if not (folder / f"correction-{frame}.json").is_file():
            fits.append(
                [
                    f"fit:{name}",
                    sys.executable,
                    str(HERE / "fit.py"),
                    "--frame",
                    str(frame),
                    "--basis",
                    str(args.output / "base" / name),
                    "--output",
                    str(folder),
                ]
            )
    for name, *command in fits:
        folder = Path(command[-1])
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "fit.log").open("w") as stream:
            stream.write(f"# command: {shlex.join(command)}\n")
            stream.flush()
            status = subprocess.run(
                command,
                cwd=args.tree,
                env=environment(args, 0),
                stdout=stream,
                stderr=subprocess.STDOUT,
            ).returncode
            stream.write(f"EXIT={status}\n")
        print("FIT", name, status, flush=True)
    corrected = []
    for frame, branch in ROWS:
        name = case_name(frame, branch)
        folder = args.output / "corrected" / name
        if not completed(folder):
            correction = args.output / "fits" / name / f"correction-{frame}.json"
            corrected.append(
                (
                    f"corrected:{name}",
                    measure_command(args, frame, branch, folder, correction),
                    folder,
                )
            )
    run_all(args, corrected, results)


if __name__ == "__main__":
    main()
