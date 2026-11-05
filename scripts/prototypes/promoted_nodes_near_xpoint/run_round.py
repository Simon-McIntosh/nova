"""Measure the promoted nodes and draw the panel in one lane payload.

Each step is followed by a check that the artifact it exists to produce is on
disk, because a step that returns success without writing its output reports a
measurement nobody can read and a job that reads as passed.
"""

import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
EVIDENCE = Path("/work/projects/imas_gpu/sophelio/evidence/promoted-nodes-near-xpoint")
RECEIPT = EVIDENCE / "measure-5.json"
FIELDS = EVIDENCE / "measure-5-fields.npz"
FIGURE = Path(
    "docs/figures/separatrix-spline-extraction/"
    "promoted-nodes-near-xpoint/promoted-nodes-panel.svg"
)
PANEL = EVIDENCE / "promoted-nodes-panel-check.png"

STEPS = (
    (
        [
            "scripts/prototypes/promoted_nodes_near_xpoint/measure_promoted_nodes.py",
            "--edit-index",
            "12",
            "--receipt",
            str(RECEIPT),
            "--fields",
            str(FIELDS),
        ],
        (RECEIPT, FIELDS),
    ),
    (
        [
            "scripts/prototypes/promoted_nodes_near_xpoint/render_panel.py",
            "--fields",
            str(FIELDS),
            "--receipt",
            str(RECEIPT),
            "--svg",
            str(FIGURE),
            "--png",
            str(PANEL),
        ],
        (FIGURE, PANEL),
    ),
)

# A stale artifact from an earlier attempt would satisfy the check below, so
# clear the evidence-directory ones and date the in-tree figure instead of
# deleting it, which would leave the worktree dirty if the render fails.
for path in (RECEIPT, FIELDS, PANEL):
    if Path(path).is_file():
        print(f"REMOVE-STALE {path}", flush=True)
        Path(path).unlink()
started = time.time()

for step, produced in STEPS:
    print("STEP " + " ".join(step), flush=True)
    completed = subprocess.run([PYTHON, *step], check=False)
    if completed.returncode:
        print(f"STEP-FAILED exit {completed.returncode} {step[0]}", flush=True)
        sys.exit(completed.returncode)
    missing = [
        str(path)
        for path in produced
        if not Path(path).is_file() or Path(path).stat().st_mtime < started
    ]
    if missing:
        print(f"STEP-WROTE-NOTHING {step[0]} missing {missing}", flush=True)
        sys.exit(1)
    for path in produced:
        print(f"PRODUCED {path} {Path(path).stat().st_size} bytes", flush=True)
print("ROUND-OK", flush=True)
