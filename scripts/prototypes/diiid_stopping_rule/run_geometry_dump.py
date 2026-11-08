"""Record the machine geometry for each rendered terminal panel.

One fresh interpreter per frame keeps the profile builders from accumulating
across frames, and each frame prints its own exit status so a partial run is
readable on its own.
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
    recorder = target / "scripts/prototypes/diiid_stopping_rule/record_machine_geometry.py"
    frames = [(179, "limited"), (44, "limited"), (144, "diverted"), (137, "diverted")]
    for frame, branch in frames:
        output = run_root / "geometry" / f"geometry-{frame}-{branch}.npz"
        if output.exists():
            print(f"=== {frame} {branch} skipped", flush=True)
            continue
        print(f"=== {frame} {branch}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(recorder),
                "--frame",
                str(frame),
                "--branch",
                branch,
                "--artifact",
                str(run_root / f"probe-artifacts2/frame-{frame}-{branch}-default.npz"),
                "--record",
                str(run_root / f"probe-parts2/probe-frame-{frame}.json"),
                "--output",
                str(output),
            ],
            cwd=target,
            check=False,
        )
        print(f"EXIT_{frame}_{branch.upper()}={completed.returncode}", flush=True)


if __name__ == "__main__":
    main()