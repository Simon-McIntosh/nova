"""Withdraw the promotion counts the probe did not measure.

The stored arms carry ``attempted_promotions = accepted_promotions = 24`` on
every solved record. Those numbers were read from the length of the fixed-shape
inner-iteration trace, which the solver pads with ``NOT_EXECUTED`` (-1) decisions
and nan applied factors, so they measure the padding rather than any work: the
trace in every stored arm is nothing but padding, and an all-padding trace reads
as 24 of 24 accepted.

The counts the receipt was supposed to carry are the solver's own accumulated
``attempted_newton_promotions`` and ``accepted_newton_promotions`` on the fixed
point result. ``measure_branches`` now reads them from there, but the solves
that produced this record ran in processes that have exited, so the values
cannot be recovered from what was written down. This script therefore withdraws
them for the stored arms and replaces them with what the stored trace does
support: the number of executed rows, the accepted and refused codes among them,
and the damped steps.
"""

import json
import sys
from pathlib import Path
from typing import Any

ACCEPTED = frozenset({1, 2, 3, 4})
REFUSED = frozenset({5, 6})
NOT_EXECUTED = -1

NOTE = (
    "withdrawn: the stored counts were the length of the padded inner-iteration "
    "trace, not the solver's promotion tally; the trace in this arm is entirely "
    "NOT_EXECUTED padding, so acceptance is unmeasured here. measure_branches "
    "now reads attempted_newton_promotions and accepted_newton_promotions from "
    "the fixed point result."
)


def _tally(arm: dict[str, Any]) -> dict[str, Any]:
    decisions = arm.get("globalisation_decisions") or []
    factors = arm.get("globalisation_factors") or []
    codes = [int(value) for value in decisions]
    executed = [code for code in codes if code != NOT_EXECUTED]
    damped = sum(
        1 for value in factors if value is not None and abs(float(value) - 1.0) > 0.0
    )
    return {
        "slots": len(codes),
        "executed": len(executed),
        "accepted": sum(code in ACCEPTED for code in executed),
        "refused": sum(code in REFUSED for code in executed),
        "damped": damped,
    }


def _repair(arm: dict[str, Any]) -> bool:
    if "globalisation_decisions" not in arm:
        return False
    tally = _tally(arm)
    arm["attempted_promotions"] = None
    arm["accepted_promotions"] = None
    arm["inner_step_trace"] = tally
    arm["promotion_counts_measured"] = tally["executed"] > 0
    arm["promotion_counts_note"] = NOTE
    return True


def walk(node: Any) -> int:
    repaired = 0
    if isinstance(node, dict):
        if _repair(node):
            repaired += 1
        for value in node.values():
            repaired += walk(value)
    elif isinstance(node, list):
        for value in node:
            repaired += walk(value)
    return repaired


def main(argv: list[str]) -> int:
    probe = Path(argv[1])
    document = json.loads(probe.read_text())
    repaired = walk(document)
    probe.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    parts = sorted((probe.parent / "parts").glob("probe-frame-*.json"))
    for part in parts:
        piece = json.loads(part.read_text())
        walk(piece)
        part.write_text(json.dumps(piece, indent=2, sort_keys=True) + "\n")
    print(f"arms repaired: {repaired}; parts rewritten: {len(parts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
