"""Solve one DIII-D row on the declared profile-extent arm.

The solve, the branch seeding and the shadow-node imaging are the established
cohort's; this driver changes only which knots the prescribed p'/FF' tables
carry, and records the tables the solve was actually handed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_xpoint_held_nodes"))
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_xpoint_profile_extent"))

import arms  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "held_measure", ROOT / "scripts/prototypes/diiid_xpoint_held_nodes/measure.py"
)
held_measure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = held_measure
SPEC.loader.exec_module(held_measure)

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument("--arm", choices=arms.ARMS, required=True)
    args = parser.parse_args()
    assert Path(bench.__file__).resolve().is_relative_to(ROOT), bench.__file__
    args.output.mkdir(parents=True, exist_ok=True)
    arms.install(bench, args.arm)
    arms.write(
        args.output / "arm.json",
        {**arms.describe(args.arm), "revision": args.revision},
    )
    print("PROFILE_ARM " + json.dumps(arms.describe(args.arm)), flush=True)
    # The held cohort installs its extra image read in a block that only runs
    # when its module is the process entry point, and that block never executes
    # when the module is loaded as a library. Install it here, so a row writes
    # the image receipt the batch driver's retention check requires.
    held_measure.core.install_analysis = held_measure.install_analysis
    sys.argv = [
        sys.argv[0],
        "--tree",
        str(args.tree),
        "--revision",
        args.revision,
        "--frame",
        str(args.frame),
        "--branch",
        args.branch,
        "--output",
        str(args.output),
        "--trace-helper",
        str(args.trace_helper),
    ]
    held_measure.core.main()


if __name__ == "__main__":
    main()
