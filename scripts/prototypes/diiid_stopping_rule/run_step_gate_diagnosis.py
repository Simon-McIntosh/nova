"""Run the step-gate diagnosis and the missing frame arm in fresh processes.

Each step gets its own interpreter so compiled programs from one do not
accumulate onto the next, and the exit status of every step is printed with
the log so a partial run is readable on its own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    arguments = parser.parse_args()

    target = arguments.repository_root
    run_root = arguments.run_root
    diagnosis = target / "scripts/prototypes/diiid_stopping_rule/diagnose_step_gate.py"
    probe = target / "scripts/prototypes/diiid_forward_convergence/measure_branches.py"
    steps = [
        (
            "179_limited",
            [
                sys.executable,
                str(diagnosis),
                "--frame",
                "179",
                "--branch",
                "limited",
                "--trips",
                "2",
            ],
        ),
        (
            "44_limited",
            [
                sys.executable,
                str(diagnosis),
                "--frame",
                "44",
                "--branch",
                "limited",
                "--trips",
                "2",
            ],
        ),
        (
            "146_diverted",
            [
                sys.executable,
                str(probe),
                "--frames",
                "146",
                "--branches",
                "diverted",
                "--output",
                str(run_root / "probe-parts3/probe-frame-146.json"),
                "--artifact-dir",
                str(run_root / "probe-artifacts3"),
            ],
        ),
    ]
    for name, command in steps:
        print(f"=== {name}", flush=True)
        completed = subprocess.run(command, cwd=target, check=False)
        print(f"EXIT_{name.upper()}={completed.returncode}", flush=True)


if __name__ == "__main__":
    main()
