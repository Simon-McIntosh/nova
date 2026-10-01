"""Print whether any of the named SLURM jobs are still in the queue.

Last line is IN_QUEUE while at least one named job is still queued or running,
and LEFT_QUEUE once every one of them has left the queue (completed, failed or
cancelled).  Used as a wait probe: the token is the state, the rows above it are
the detail.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    job_ids = [value for value in sys.argv[1:] if value.strip()]
    if not job_ids:
        print("LEFT_QUEUE")
        return 0
    listing = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids)],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = [line for line in listing.stdout.splitlines() if line.strip()]
    for row in rows:
        print(row)
    print("IN_QUEUE" if rows else "LEFT_QUEUE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())