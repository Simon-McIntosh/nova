"""Evaluate the true-outline model with fill fraction 1 and show the guard fails.

The mutation is applied to the measurement's own artifacts: the true model's
coarse field is replaced by the unit-fill arm's field, which is what the
measurement records when the true model is built with fill fraction 1. The
landed comparator is untouched.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--scratch", type=Path, required=True)
args = parser.parse_args()

args.scratch.mkdir(parents=True, exist_ok=True)
summary = json.loads((args.directory / "summary.json").read_text())
coarse = max(summary["sizes_m"])
for name in (coarse, min(summary["sizes_m"])):
    shutil.copyfile(
        args.directory / f"west-true-{name:g}.npz",
        args.scratch / f"west-true-{name:g}.npz",
    )
shutil.copyfile(
    args.directory / "west-true-fill-one.npz", args.scratch / "west-true-fill-one.npz"
)
true = dict(np.load(args.scratch / f"west-true-{coarse:g}.npz"))
unit = np.load(args.scratch / "west-true-fill-one.npz")
true["field"] = unit["field"]
np.savez(args.scratch / f"west-true-{coarse:g}.npz", **true)
(args.scratch / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
raise SystemExit(
    pytest.main(
        [
            "-p",
            "no:cacheprovider", "-q",
            "tests/test_magnetisation.py",
            "-k",
            "true_outline_fill_fraction_exceeds_its_mesh_pair_difference",
        ]
    )
)