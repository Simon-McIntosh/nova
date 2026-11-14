"""Run the measured-current cohort inside one GPU allocation."""

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
ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))
RECEIPTS = ("receipt.json", "core-analysis.json", "shadow-image.json")


def case_name(frame, branch):
    return f"frame-{frame}-{branch}"


def landed(folder):
    log = folder / "row.log"
    if not log.is_file():
        return False
    if any(not (folder / name).is_file() for name in RECEIPTS):
        return False
    return log.read_text().endswith("EXIT=0\n")


def measure_command(args, frame, branch, folder, correction=None):
    command = [sys.executable, str(HERE / "measure.py"),
               "--tree", str(args.tree), "--revision", args.revision,
               "--frame", str(frame), "--branch", branch,
               "--output", str(folder), "--trace-helper", str(args.trace_helper)]
    if correction is not None:
        command += ["--correction", str(correction)]
    return command


def environment(args, card, slots):
    value = dict(os.environ, PYTHONPATH=str(args.tree), NOVA_LANE_RUNG=args.rung,
                 XLA_PYTHON_CLIENT_PREALLOCATE="false",
                 XLA_PYTHON_CLIENT_MEM_FRACTION=str(round(0.9 / max(slots, 1), 4)),
                 CUDA_VISIBLE_DEVICES=str(card))
    value.pop("SLURM_JOB_ID", None)
    return value


def run_all(args, pending, results):
    running = {}
    while pending or running:
        while pending and len(running) < args.slots:
            name, command, folder = pending.pop(0)
            folder.mkdir(parents=True, exist_ok=True)
            stream = (folder / "row.log").open("w")
            stream.write(f"# revision: {args.revision}; tree: {args.tree}; "
                         f"command: {shlex.join(command)}\n")
            stream.flush()
            card = len(running) % max(args.slots, 1)
            process = subprocess.Popen(command, cwd=args.tree,
                                       env=environment(args, card, args.slots))
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
            entry.update(exit_status=process.returncode, log_path=str(folder / "row.log"))
            if process.returncode == 0 and (folder / "receipt.json").is_file():
                entry.update(json.loads((folder / "receipt.json").read_text()))
                entry.update(json.loads((folder / "shadow-image.json").read_text()))
            (args.output / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
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
        if not landed(folder):
            basis.append((f"basis:{name}", measure_command(args, frame, branch, folder), folder))
    run_all(args, basis, results)
    fits = []
    for frame, branch in ROWS:
        name = case_name(frame, branch)
        folder = args.output / "fits" / name
        if not (folder / f"correction-{frame}.json").is_file():
            fits.append([f"fit:{name}", sys.executable, str(HERE / "fit.py"),
                         "--frame", str(frame), "--basis", str(args.output / "base" / name),
                         "--output", str(folder)])
    for name, *command in fits:
        folder = Path(command[-1])
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "fit.log").open("w") as stream:
            stream.write(f"# command: {shlex.join(command)}\n")
            stream.flush()
            status = subprocess.run(command, cwd=args.tree, env=environment(args, 0, 1),
                                    stdout=stream, stderr=subprocess.STDOUT).returncode
            stream.write(f"EXIT={status}\n")
        print("FIT", name, status, flush=True)
    corrected = []
    for frame, branch in ROWS:
        name = case_name(frame, branch)
        folder = args.output / "corrected" / name
        if not landed(folder):
            correction = args.output / "fits" / name / f"correction-{frame}.json"
            corrected.append((f"corrected:{name}",
                              measure_command(args, frame, branch, folder, correction), folder))
    run_all(args, corrected, results)


if __name__ == "__main__":
    main()