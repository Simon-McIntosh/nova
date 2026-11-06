"""Verify the final tests and repeat the declared mutation at that revision."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_booking_primal_buckets.suite import FILES, run_file
from scripts.prototypes.bank_wall_events.suite import archive


def main():
    root, run = map(Path, sys.argv[1:])
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="booking-verify-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        results = [
            run_file(root, revision, name, "verified", run, scratch)
            for name in ("test_book_primal_buckets", *FILES)
        ]
        negative = scratch / "negative"
        archive(root, revision, negative)
        path = negative / "nova/equilibrium/current.py"
        source = path.read_text()
        target = "    normal = cut & read.normal_form_cells\n"
        assert source.count(target) == 2
        path.write_text(
            source.replace(
                target,
                target + "    normal = normal.at[jnp.argmax(normal)].set(False)\n",
                1,
            )
        )
        failed = run_file(
            negative,
            revision + "+misrouted-normal",
            "test_book_primal_buckets",
            "verified-negative",
            run,
            scratch,
        )
        print("VERIFIED", results, "NEGATIVE", failed, flush=True)
        print("Removing scratch", scratch, flush=True)
        return int(any(results) or failed == 0)


if __name__ == "__main__":
    raise SystemExit(main())
