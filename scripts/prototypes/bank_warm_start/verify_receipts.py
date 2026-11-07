"""Check the fragment's coarse-step counts against the run receipts.

The fragment states two counts that come straight out of the receipts: how many
coarse-step slices run every lane to the update ceiling, and how many warm
slices converge in every lane. Both are recomputed here and compared with the
text, so a fragment that drifts from its own table fails this check rather than
being read as fact.

``--fragment`` and ``--source`` point at the files to check, so the same script
judges a revision other than the one in the tree.
"""

import argparse
import json
from pathlib import Path

WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine",
}


def counts(report, step):
    table = [row for row in report["arms"]["runner1"]["table"] if row["slice"] > 0]
    coarse = [row for row in table if row["step"] == step]
    stalled = [
        row for row in coarse if row["updates_max"] == 60 and row["converged"] == 0
    ]
    full = [row for row in table if row["converged"] == row["lanes"]]
    return {
        "coarse_slices": len(coarse),
        "stalled_slices": [row["slice"] for row in stalled],
        "warm_slices": len(table),
        "fully_converged": len(full),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    found = counts(json.loads(args.report.read_text()), 1e-2)
    text = args.fragment.read_text()
    failures = []

    stalled = len(found["stalled_slices"])
    stalled_forms = (
        f"{stalled} of the {found['coarse_slices']} coarse-step slices".lower(),
        f"{WORDS[stalled]} of the {WORDS[found['coarse_slices']]} coarse-step slices".lower(),
    )
    if not any(form in text.lower() for form in stalled_forms):
        failures.append(
            "fragment does not state that "
            f"{stalled} of the {found['coarse_slices']} coarse-step slices stall"
        )
    if "21 converged warm slices" in text:
        failures.append("fragment claims all warm slices converged")
    if (
        f"{found['fully_converged']} of the {found['warm_slices']} warm slices"
        not in text
    ):
        failures.append(
            f"fragment does not state {found['fully_converged']} of "
            f"{found['warm_slices']} warm slices fully converge"
        )

    reuse = {
        "plot imports the ink trace style": "trace_axes" in args.source.read_text()
    }
    for name, held in reuse.items():
        if not held:
            failures.append(f"{name}: missing")
    for failure in failures:
        print(f"FAIL {failure}")
    print(json.dumps({"receipts": found, "failures": len(failures)}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()