"""Probe whether this node's H200 measurement has reached a deliverable state.

Read by the crew resume-ready watcher as the wait probe: the manifest's
``wait_terminal`` names the two terminal tokens, so a watcher re-invokes the
run only after the job it started has finished and the receipt it writes has
either finalised (``exit_marker 0`` with width-one members and baselines
present) or failed hard (job gone, receipt not finalised).  Everything the
probe needs resolves from the file's own location and from the receipt, so it
runs unchanged from any checkout or job id.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RECEIPT = ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/width-one-compiled.json"


def _job_terminal(job_id: str) -> bool:
    probe = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    state = probe.stdout.strip().upper()
    if not state:
        return True
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
    }
    return state in terminal


def main() -> int:
    if not RECEIPT.exists():
        print("running")
        return 0
    try:
        payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("running")
        return 0
    execution = payload.get("execution") or {}
    job_id = str(payload.get("assignment", {}).get("job_id") or "")
    members = payload.get("width_one", {}).get("members") or []
    banked_members = [
        row
        for row in members
        if row.get("failure") is not None or row.get("compiled") is not None
    ]
    complete = (
        execution.get("exit_marker") == 0
        and bool(banked_members)
        and payload.get("baseline_summary") is not None
    )
    if complete:
        print("measurement-done")
        return 0
    if job_id and _job_terminal(job_id):
        print("measurement-failed")
        return 0
    print("running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
