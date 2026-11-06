"""Warm each revision once, then measure two fresh processes in one allocation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_wall_events.suite import archive
from scripts.prototypes.bank_wall_events.device import execute


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
        prefix="candidate-device-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        archive(args.root, args.base, base)
        for arm, tree, revision in (
            ("base", base, args.base),
            ("head", args.root, head),
        ):
            environment = dict(os.environ, PYTHONPATH=str(tree), TMPDIR=str(scratch))
            for process in range(3):
                destination = args.run / f"{arm}-{process}"
                destination.mkdir(exist_ok=True)
                command = [
                    "bash",
                    str(
                        args.root
                        / "scripts/prototypes/bank_booking_determinism/measurement.sh"
                    ),
                    str(args.run / (arm + "-cache")),
                    sys.executable,
                    "-P",
                    str(
                        args.root / "scripts/prototypes/bank_candidate_loop/capture.py"
                    ),
                    "--run",
                    str(destination),
                    "--revision",
                    revision,
                ]
                execute(
                    tree,
                    revision,
                    command,
                    args.run / f"device-{arm}-{process}.log",
                    environment,
                )
                if process == 1:
                    execute(
                        tree,
                        revision,
                        [
                            sys.executable,
                            "-P",
                            "-m",
                            "scripts.prototypes.bank_solve_trace.analyse",
                            str(destination),
                        ],
                        args.run / f"attribution-{arm}.log",
                        environment,
                    )
        environment = dict(
            os.environ,
            PYTHONPATH=str(args.root),
            JAX_PLATFORMS="cuda",
            TMPDIR=str(scratch),
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
        )
        environment.pop("XLA_FLAGS", None)
        command = [
            sys.executable,
            "-P",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--basetemp=" + str(scratch / "physical"),
            str(args.root / "tests/test_candidate_inspection_batched.py"),
            "-m",
            "slow",
            "-vv",
            "-s",
        ]
        # The test process prints provenance before pytest starts.
        command = [
            sys.executable,
            "-P",
            "-c",
            "import nova, nova.equilibrium.topology as m; "
            "from pathlib import Path; import pytest,sys; "
            "print('nova',nova.__file__,'module',m.__file__,"
            "'cwd',Path.cwd().resolve(),flush=True); "
            "raise SystemExit(pytest.main(sys.argv[1:]))",
            *command[4:],
        ]
        execute(args.root, head, command, args.run / "physical.log", environment)
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
