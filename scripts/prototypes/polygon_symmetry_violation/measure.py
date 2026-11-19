"""One job: reproduce the zero-entry split, then measure the derivative.

The split is recomputed from the receipt and the cell geometry, so the
counts and maxima this run reports are recomputed rather than copied, and
``compare.py`` then measures the booked dpsi/dZ against a psi central
difference at the group-A violations and the worst group-B entries.

Prints a header naming the tree the payload imports, the split's summary,
and a per-entry table; the full record is the JSON ``compare.py`` writes.

Run with ``PYTHONPATH=. /home/ITER/mcintos/Code/nova/.venv/bin/python
scripts/prototypes/polygon_symmetry_violation/measure.py --worktree
<checkout> --output <directory>``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
RECEIPT = (
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261117T110237355670-iter-bank-recertify/receipts/zero-entries.json"
)
GEOMETRY = "/home/ITER/mcintos/.cache/nova-hessian-diagnosis/h200/geometry.npz"


def run(command, *, cwd):
    """Run one child with the tree on its path and echo both its streams."""
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=dict(os.environ, PYTHONPATH=str(cwd)),
        capture_output=True,
        text=True,
    )
    print("$ " + " ".join(command), flush=True)
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"child exited {completed.returncode}: {command[1]}")
    return completed


def table(payload):
    """Per-entry rows: each tree's booked value, the psi limit, the band sweep."""
    print("\nGROUP A — booked dpsi/dZ against the psi central difference")
    print("node cell moment repaired base nocorner noband psi_limit gap")
    for row in payload["group_a"]:
        direct = row["direct"]
        print(
            "%d %d %d %.3e %.3e %.3e %.3e %.3e %s"
            % (
                row["node"],
                row["cell"],
                row["moment"],
                direct["current"],
                direct["base"],
                direct["nocorner"],
                direct["noband"],
                row["finest"],
                "%.2e" % row["largest_gap"]
                if row["largest_gap"] is not None
                else "n/a",
            )
        )
    print("\nGROUP A — booked value across the corner band")
    offsets = " ".join("%+.0e" % offset for offset, _ in payload["group_a"][0]["sweep"])
    print("node " + offsets)
    for row in payload["group_a"]:
        values = " ".join("%.2e" % value for _, value in row["sweep"])
        print("%d " % row["node"] + values)
    print("\nGROUP B — booked dpsi/dZ against the psi central difference")
    print("node cell moment repaired base psi_limit gap")
    for row in payload["group_b"]:
        direct = row["direct"]
        print(
            "%d %d %d %.3e %.3e %.3e %s"
            % (
                row["node"],
                row["cell"],
                row["moment"],
                direct["current"],
                direct["base"],
                row["finest"],
                "%.2e" % row["largest_gap"]
                if row["largest_gap"] is not None
                else "n/a",
            )
        )
    print("\nSCALE — booked dpsi/dZ at each violating cell's own radius, off plane")
    for row in payload["scale"]:
        print(
            "node %d cell %d moment %d dz %+.2f repaired %.3e base %.3e"
            % (
                row["node"],
                row["cell"],
                row["moment"],
                row["dz"],
                row["direct"]["current"],
                row["direct"]["base"],
            )
        )


def main():
    argv = sys.argv[1:]
    worktree = output = None
    batch = "2"
    for index, value in enumerate(argv):
        if value == "--worktree":
            worktree = argv[index + 1]
        elif value == "--output":
            output = argv[index + 1]
        elif value == "--batch":
            batch = argv[index + 1]
    if worktree is None or output is None:
        raise SystemExit("--worktree and --output are required")

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "-C", worktree, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print("# revision: " + revision)
    print("# tree: " + str(ROOT))
    print("# receipt: " + RECEIPT)
    print("# geometry: " + GEOMETRY)

    split = destination / "split.json"
    run(
        [
            sys.executable,
            str(ROOT / "scripts/prototypes/polygon_symmetry_violation/split.py"),
            "--receipt",
            RECEIPT,
            "--geometry",
            GEOMETRY,
            "--output",
            str(split),
        ],
        cwd=ROOT,
    )
    compare = destination / "compare.json"
    run(
        [
            sys.executable,
            str(ROOT / "scripts/prototypes/polygon_symmetry_violation/compare.py"),
            "--split",
            str(split),
            "--geometry",
            GEOMETRY,
            "--head-tree",
            str(ROOT),
            "--worktree",
            worktree,
            "--directory",
            str(destination / "scratch"),
            "--output",
            str(compare),
            "--batch",
            batch,
        ],
        cwd=ROOT,
    )
    with open(compare) as stream:
        payload = json.load(stream)
    print(
        "PSI_IDENTITY max_abs_difference %.3e" % payload["psi_current_vs_base_max_abs"]
    )
    table(payload)


if __name__ == "__main__":
    main()
