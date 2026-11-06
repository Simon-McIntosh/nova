"""Capture the imported solve instrument in fresh processes sharing one cache."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_wall_events.device import execute


def main():
    root, run = map(Path, sys.argv[1:])
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="solve-trace-", dir=os.environ["TMPDIR"]
    ) as scratch:
        environment = dict(
            os.environ,
            PYTHONPATH=str(root),
            TMPDIR=scratch,
            JAX_DEBUG_LOG_MODULES="jax._src.compiler",
        )
        for process in range(3):
            destination = run / f"process-{process}"
            destination.mkdir(exist_ok=True)
            command = [
                "bash",
                str(
                    root / "scripts/prototypes/bank_booking_determinism/measurement.sh"
                ),
                str(run / "compilation-cache"),
                sys.executable,
                "-P",
                "-c",
                "from scripts.prototypes.bank_solve_trace.capture import main; main()",
                "--run",
                str(destination),
            ]
            execute(
                root, revision, command, run / f"capture-{process}.log", environment
            )
            if process:
                execute(
                    root,
                    revision,
                    [
                        sys.executable,
                        "-P",
                        "-c",
                        "from scripts.prototypes.bank_solve_trace.analyse "
                        "import main; main()",
                        str(destination),
                    ],
                    run / f"attribution-{process}.log",
                    environment,
                )
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
