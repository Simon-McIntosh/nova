"""Run physical wall checks and fresh-process trajectory pairs in one GPU allocation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_wall_events.device import execute
from scripts.prototypes.bank_wall_events.suite import archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    args.run.mkdir(exist_ok=True)
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="wall-precision-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        archive(args.root, args.base, base)
        environment = dict(
            os.environ,
            PYTHONPATH=str(args.root),
            JAX_PLATFORMS="cuda,cpu",
            TMPDIR=str(scratch),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="2",
        )
        probe = [
            sys.executable,
            "-P",
            "-c",
            "from scripts.prototypes.bank_wall_events.trajectory import "
            "callback_control; callback_control()",
        ]
        execute(args.root, head, probe, args.run / "callback-control.log", environment)
        command = [
            sys.executable,
            "-P",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.prototypes.bank_wall_events.memory",
            f"--basetemp={scratch / 'physical'}",
            str(args.root / "tests/test_wall_events_batched.py"),
            "-m",
            "slow or not slow",
            "-vv",
            "-s",
        ]
        execute(args.root, head, command, args.run / "physical.log", environment)
        for repeat in range(2):
            for arm, tree, revision in (
                ("base", base, args.base),
                ("head", args.root, head),
            ):
                environment["PYTHONPATH"] = str(tree)
                name = f"{arm}-{repeat}"
                command = [
                    sys.executable,
                    "-P",
                    str(
                        args.root / "scripts/prototypes/bank_wall_events/trajectory.py"
                    ),
                    "--output",
                    str(args.run / name),
                    "--revision",
                    revision,
                    "--arm",
                    arm,
                    "--repeat",
                    str(repeat),
                ]
                execute(
                    tree, revision, command, args.run / (name + ".log"), environment
                )
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
