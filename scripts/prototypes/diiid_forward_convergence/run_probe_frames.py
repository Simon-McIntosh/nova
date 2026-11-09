"""Run one fresh process per frame and merge the receipts as they land.

Compiled programs and their host buffers accumulate across frames inside one
process, so a job that solved three frames in sequence exhausted its allocation
on the third.  This driver gives each frame its own process, records what each
one cost, and merges the receipts afterwards.

A frame that fails does not stop the sequence: the remaining frames are still
worth measuring, and the driver reports which ones completed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--part-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pseudo-wall-expansion", type=float, default=None)
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=("limited", "diverted"),
        default=None,
        help="the branches to measure; the rest of the frame is left alone",
    )
    arguments = parser.parse_args()
    arguments.part_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, object]] = []
    for frame in arguments.frames:
        part = arguments.part_dir / f"probe-frame-{frame}.json"
        if part.exists():
            print(json.dumps({"frame": frame, "skipped": str(part)}), flush=True)
            continue
        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            f"{arguments.repository_root}:{environment.get('PYTHONPATH', '')}"
        )
        command = [
            sys.executable,
            str(HERE / "measure_branches.py"),
            "--frames",
            str(frame),
            "--output",
            str(part),
            "--artifact-dir",
            str(arguments.artifact_dir),
        ]
        if arguments.branches:
            command.extend(["--branches", *arguments.branches])
        if arguments.pseudo_wall_expansion is not None:
            command.extend(
                ["--pseudo-wall-expansion", str(arguments.pseudo_wall_expansion)]
            )
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=arguments.repository_root, env=environment
        )
        seconds = time.perf_counter() - started
        peak = None
        if part.exists():
            peak = json.loads(part.read_text()).get("peak_rss_kb")
        outcomes.append(
            {
                "frame": frame,
                "returncode": completed.returncode,
                "seconds": round(seconds, 1),
                "peak_rss_kb": peak,
            }
        )
        print(json.dumps(outcomes[-1]), flush=True)
        if part.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "merge_probe.py"),
                    "--probe",
                    str(arguments.probe),
                    "--parts",
                    str(part),
                ],
                cwd=arguments.repository_root,
                env=dict(os.environ),
            )
    failed = [item["frame"] for item in outcomes if item["returncode"]]
    print(json.dumps({"outcomes": outcomes, "failed": failed}), flush=True)
    if failed:
        raise SystemExit(f"frames that failed: {failed}")


if __name__ == "__main__":
    main()
