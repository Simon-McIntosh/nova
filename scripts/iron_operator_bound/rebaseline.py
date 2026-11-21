"""Rewrite the shell operator control to a repaired revision.

The pin compares the default-argument operator at the working tree against the
same function loaded from ``control["revision"]``.  Re-baselining needs a
revision that already carries the repair, because that is the operator the
working tree now reproduces, and the control cannot name the commit that
contains it.  This script assembles the control's three shell cells at the
given revision, checks that the working tree reproduces them, and writes both
the revision and the operator into the control.

    python scripts/iron_operator_bound/rebaseline.py --revision <sha>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_linear_magnetisation.controls import shell  # noqa: E402
from scripts.iron_linear_magnetisation.kernel_cost import load_owner  # noqa: E402

assert jax.config.jax_enable_x64

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--revision", required=True)
parser.add_argument("--write", action="store_true")
args = parser.parse_args()
root = Path(__file__).parents[2]
path = root / "scripts" / "iron_linear_magnetisation" / "operator-control.json"
control = json.loads(path.read_text())
cells = iron._mesh(shell(), 0.04)
assert len(cells.polygons) == control["shell_cells"] == 228
polygons = tuple(cells.polygons[index] for index in control["selected_cells"])
with tempfile.TemporaryDirectory() as directory:
    reference = load_owner(args.revision, "rebaseline_owner", directory=Path(directory))
    expected, _ = reference.linear_galerkin_cell_interaction(polygons, plan=iron._PLAN)
actual, blocks = iron.linear_galerkin_cell_interaction(polygons, plan=iron._PLAN)
difference = np.asarray(actual) - np.asarray(expected)
relative = float(np.linalg.norm(difference) / np.linalg.norm(expected))
print(
    json.dumps(
        dict(
            revision=args.revision,
            relative=relative,
            previous=control["revision"],
            stored_relative_to_working=float(
                np.linalg.norm(np.asarray(actual) - np.asarray(control["operator"]))
                / np.linalg.norm(control["operator"])
            ),
        )
    ),
    flush=True,
)
assert relative <= 1e-14, relative
if args.write:
    control["revision"] = args.revision
    control["operator"] = [[float(value) for value in row] for row in actual]
    path.write_text(json.dumps(control, indent=2) + "\n")
    print("written", path, flush=True)
