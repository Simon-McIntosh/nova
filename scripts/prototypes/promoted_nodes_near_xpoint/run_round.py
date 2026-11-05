"""Measure the promoted nodes and draw the panel in one lane payload."""

import subprocess
import sys

PYTHON = sys.executable
EVIDENCE = "/work/projects/imas_gpu/sophelio/evidence/promoted-nodes-near-xpoint"
FIGURE = (
    "docs/figures/separatrix-spline-extraction/"
    "promoted-nodes-near-xpoint/promoted-nodes-panel.svg"
)
STEPS = (
    [
        "scripts/prototypes/promoted_nodes_near_xpoint/measure_promoted_nodes.py",
        "--edit-index",
        "12",
        "--receipt",
        f"{EVIDENCE}/measure-5.json",
        "--fields",
        f"{EVIDENCE}/measure-5-fields.npz",
    ],
    [
        "scripts/prototypes/promoted_nodes_near_xpoint/render_panel.py",
        "--fields",
        f"{EVIDENCE}/measure-5-fields.npz",
        "--receipt",
        f"{EVIDENCE}/measure-5.json",
        "--svg",
        FIGURE,
        "--png",
        f"{EVIDENCE}/promoted-nodes-panel-check.png",
    ],
)
for step in STEPS:
    print("STEP " + " ".join(step), flush=True)
    completed = subprocess.run([PYTHON, *step], check=False)
    if completed.returncode:
        sys.exit(completed.returncode)
print("ROUND-OK", flush=True)
