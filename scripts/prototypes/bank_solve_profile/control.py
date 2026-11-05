"""Negative control for the closure check: does it read the stages at all?

The closure ratio is the stage sum over the whole solve's per-iterate wall. A
control that recomputes that arithmetic itself cannot test it, because the
answer is the same expression rearranged. So this drives the production
function in ``receipts`` from both ends: double the stages and the ratio must
double, halve the wall and it must double again, and remove a stage record and
the split must refuse rather than close.

The same three probes then run against a deliberately inert closure — one that
returns a constant — and it must fail them. A control whose bad input also
passes is measuring nothing, which is what the constant here exists to catch.
"""

import argparse
import json
import tempfile
from pathlib import Path

from scripts.prototypes.bank_solve_profile import receipts


def inert_closure(values, iterate):
    """A closure that never read its stages: the defect the checks must catch."""
    return 2.075


def probes(closure, values, iterate):
    """The three ways a closure can be tested from outside."""
    total = closure(values, iterate)
    doubled = closure({name: 2.0 * value for name, value in values.items()}, iterate)
    halved_wall = closure(values, 0.5 * iterate)
    return {
        "staged": abs(doubled - 2.0 * total) <= 1e-9,
        "walled": abs(halved_wall - 2.0 * total) <= 1e-9,
        "nonzero": total > 0.0,
    }


def missing_stage_refused(log):
    """A split with a stage record deleted must refuse, not close quietly."""
    lines = Path(log).read_text().splitlines()
    kept = [
        line
        for line in lines
        if not (line.startswith("STAGE ") and '"name": "assembly"' in line)
    ]
    assert len(kept) == len(lines) - 2, "control could not remove both assembly records"
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as handle:
        handle.write("\n".join(kept) + "\n")
        doctored = Path(handle.name)
    try:
        medians, _ = receipts.read(doctored)
        try:
            receipts.stages_at(medians, 16)
        except ValueError as error:
            return True, str(error)
        return False, "a split missing a stage returned a closure"
    finally:
        doctored.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    medians, solves = receipts.read(args.log)

    ok = True
    for lanes in (1, 16):
        values = receipts.stages_at(medians, lanes)
        iterate = receipts.iterate(medians, solves, lanes)
        for label, closure in (
            ("receipts", receipts.closure),
            ("inert", inert_closure),
        ):
            result = probes(closure, values, iterate)
            passed = all(result.values())
            print(
                f"PROBE closure={label} lanes={lanes} "
                f"staged={result['staged']} walled={result['walled']} "
                f"nonzero={result['nonzero']} verdict={'pass' if passed else 'fail'}",
                flush=True,
            )
            if label == "inert":
                ok = ok and not passed
            else:
                ok = ok and passed
        without = {
            name: (sum(values.values()) - values[name]) / iterate
            for name in receipts.NAMES
        }
        biggest = min(without, key=lambda name: without[name])
        print(
            f"DOMINANT lanes={lanes} stage={biggest!r} "
            f"closure_without_it={without[biggest]:.3f}",
            flush=True,
        )

    refused, detail = missing_stage_refused(args.log)
    print(f"FAIL_CLOSED refused={refused} detail={detail!r}", flush=True)
    ok = ok and refused
    print(json.dumps({"ok": ok}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
