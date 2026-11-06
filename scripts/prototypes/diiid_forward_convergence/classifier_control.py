"""Check the terminal classifier against the input that separates the readings.

A period-2 cycle means the descent alternates between two levels.  A descent
that repeats one level four times alternates between nothing, so it is a flat
plateau and classifies as a contraction stall.  This script feeds that input to
the classifier this node ships and requires the stall reading, then feeds the
same input to the earlier classifier, whose cycle test compared the alternating
levels for equality rather than for the size of their separation, and records
that it reports a cycle.  The second reading is the negative control: it shows
the check can fail, and it names what the earlier rule got wrong.

Exit status is 0 only when the shipped classifier reads the plateau as a stall.
"""

from __future__ import annotations

import json

from terminal_class import classify_terminal

PLATEAU = [0.5, 0.5, 0.5, 0.5]


def earlier_classifier(residuals: list[float], terminal: float) -> str:
    """Return the terminal class the earlier cycle test produces.

    The earlier test compared the two alternating pairs for equality and read
    equality as the signature of a cycle, so an exactly flat tail satisfied it.
    """

    finite = list(residuals)
    if abs(finite[-1] - terminal) > 1e-12 * max(1.0, abs(terminal)):
        finite = finite + [terminal]
    tail = finite[-4:]
    if len(tail) == 4:
        even_gap = abs(tail[2] - tail[0]) / max(abs(tail[2]), 1e-300)
        odd_gap = abs(tail[3] - tail[1]) / max(abs(tail[3]), 1e-300)
        if even_gap < 1e-3 and odd_gap < 1e-3:
            return "period-2 cycle"
    return "contraction stall"


def main() -> None:
    shipped = classify_terminal(list(PLATEAU), PLATEAU[-1])
    earlier = earlier_classifier(list(PLATEAU), PLATEAU[-1])
    result = {
        "input": PLATEAU,
        "shipped_class": shipped["class"],
        "shipped_evidence": shipped.get("evidence"),
        "earlier_class": earlier,
        "requirement": "contraction stall",
        "shipped_meets_requirement": shipped["class"] == "contraction stall",
        "negative_control_fires": earlier != shipped["class"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["shipped_meets_requirement"]:
        print("FAIL: the shipped classifier reads a flat plateau as a cycle")
        raise SystemExit(1)
    if not result["negative_control_fires"]:
        print("FAIL: the negative control does not distinguish the two readings")
        raise SystemExit(1)
    print("PASS: plateau is a stall; the earlier rule reads it as a cycle")


if __name__ == "__main__":
    main()