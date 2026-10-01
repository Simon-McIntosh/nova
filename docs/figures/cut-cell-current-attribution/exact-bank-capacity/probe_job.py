"""Report whether a scheduler job remains in the queue."""

from __future__ import annotations

import subprocess
import sys


job_id = sys.argv[1]
result = subprocess.run(
    ["squeue", "-h", "-j", job_id, "-o", "%i"],
    check=False,
    capture_output=True,
    text=True,
)
print("IN_QUEUE" if result.stdout.strip() else "LEFT_QUEUE")
