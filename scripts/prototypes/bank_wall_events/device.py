"""Capture both revisions in one device allocation, with separate fresh processes."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_wall_events.suite import archive


def execute(tree, revision, command, log, environment):
    with log.open("w") as handle:
        handle.write(
            f"# revision: {revision} tree: {tree} command: {shlex.join(command)}\n"
        )
        handle.flush()
        result = subprocess.run(
            command, cwd=tree, env=environment, stdout=handle, stderr=subprocess.STDOUT
        )
        handle.write(f"EXIT={result.returncode}\n")
    print("PROCESS_COMPLETE", log, result.returncode, flush=True)
    assert result.returncode == 0, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="wall-device-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        archive(args.root, args.base, base)
        for arm, tree, revision in (
            ("base", base, args.base),
            ("head", args.root, head),
        ):
            destination = args.run / arm
            destination.mkdir(exist_ok=True)
            environment = dict(
                os.environ,
                PYTHONPATH=str(tree),
                JAX_PLATFORMS="cuda",
                TMPDIR=str(scratch),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="2",
            )
            command = [
                sys.executable,
                "-P",
                str(args.root / "scripts/prototypes/bank_wall_events/capture.py"),
                "--run",
                str(destination),
                "--revision",
                revision,
            ]
            execute(
                tree, revision, command, args.run / f"device-{arm}.log", environment
            )
            command = [
                sys.executable,
                "-P",
                "-m",
                "scripts.prototypes.bank_solve_trace.analyse",
                str(destination),
            ]
            execute(
                tree,
                revision,
                command,
                args.run / f"attribution-{arm}.log",
                environment,
            )
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
