"""Supply the optional paired matrix receipt to both CPU booking suites."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_booking_primal_buckets.suite import run_file
from scripts.prototypes.bank_wall_events.device import execute
from scripts.prototypes.bank_wall_events.suite import archive


def main():
    root, run = map(Path, sys.argv[1:3])
    base_revision = sys.argv[3]
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="booking-matrices-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        archive(root, base_revision, base)
        execute(
            base,
            base_revision,
            [
                sys.executable,
                "-P",
                "-m",
                "scripts.prototypes.bank_step_booking_batch.measure",
                "--run",
                str(run),
                "--arm",
                "matrix-base",
                "--revision",
                base_revision,
                "--matrices",
            ],
            run / "matrix-build.log",
            dict(os.environ, PYTHONPATH=str(base), TMPDIR=str(scratch)),
        )
        os.environ["BOOKING_BASE_MATRICES"] = str(run / "matrix-base-matrices.npz")
        results = [
            run_file(
                tree, revision, "test_bank_step_booking_batch", label, run, scratch
            )
            for label, tree, revision in (
                ("matrix-base", base, base_revision),
                ("matrix-head", root, head_revision),
            )
        ]
        print("MATRICES", results, flush=True)
        print("Removing scratch", scratch, flush=True)
        return int(any(results))


if __name__ == "__main__":
    raise SystemExit(main())
