"""Solve one DIII-D row on a measured-current arm.

The solve, the branch seeding and the shadow-node image read are the established
cohort's. This driver changes only the measured conductor currents the build is
handed, and records the currents the solve was actually given.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_xpoint_held_nodes"))

SPEC = importlib.util.spec_from_file_location(
    "held_measure", ROOT / "scripts/prototypes/diiid_xpoint_held_nodes/measure.py"
)
held_measure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = held_measure
SPEC.loader.exec_module(held_measure)

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402


def install(correction, output):
    """Subtract each declared conductor error from its measured current.

    `correction` maps a conductor name to an equivalent error in ampere-turns of
    the fitted forward-minus-EFIT field, so the arm's current is the measured one
    with that error removed. A conductor the correction does not name keeps its
    measured current exactly. The applied currents and the fraction each one
    changed by are written beside the solve so the arm is a receipt.
    """

    original = bench._source_current

    def applied(row, description, names, time_ms):
        measured = original(row, description, names, time_ms)
        delta = [correction.get(name, 0.0) for name in names]
        currents = [float(item) - item_delta for item, item_delta in zip(measured, delta)]
        output.mkdir(parents=True, exist_ok=True)
        (output / "current-inputs.json").write_text(
            json.dumps(
                {
                    "time_ms": float(time_ms),
                    "names": list(names),
                    "measured_a_turn": [float(item) for item in measured],
                    "correction_a_turn": [float(item) for item in delta],
                    "applied_a_turn": currents,
                    "relative_correction": [
                        0.0 if item == 0 else item_delta / item
                        for item, item_delta in zip(measured, delta)
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        return currents

    bench._source_current = applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument(
        "--correction",
        type=Path,
        help="JSON mapping conductor name to an equivalent error in ampere-turns",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert Path(bench.__file__).resolve().is_relative_to(ROOT), bench.__file__
    correction = (
        json.loads(args.correction.read_text()) if args.correction else {}
    )
    install(correction, args.output)
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