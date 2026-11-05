"""Negative control: drop one stage from the split and watch the closure break.

The closure check compares the per-iterate stage sum against the whole solve's
own per-iterate wall. If that check were inert — a fixed ratio, or a sum that
never saw the stages — removing a stage would leave it where it was. This
zeroes each stage in turn and prints the ratio each time.
"""

import argparse
import json
from pathlib import Path

NAMES = [
    "read census",
    "read decision",
    "read geometry",
    "step residual",
    "step assembly",
    "factor",
    "back-solve",
]


def stages_at(medians, lanes):
    return {
        "read census": medians[(lanes, "census")],
        "read decision": medians[(lanes, "topology")]
        - medians[(lanes, "support_geometry")],
        "read geometry": medians[(lanes, "fit")] + medians[(lanes, "support_geometry")],
        "step residual": 2.0 * medians[(lanes, "residual")],
        "step assembly": medians[(lanes, "assembly")],
        "factor": medians[(lanes, "factor")],
        "back-solve": medians[(lanes, "backsolve")],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    medians = {}
    solves = {}
    for line in args.log.read_text().splitlines():
        if line.startswith("STAGE "):
            row = json.loads(line.partition(" ")[2])
            medians[(row["lanes"], row["name"])] = row["median_ms"]
        elif line.startswith("SOLVE "):
            row = json.loads(line.partition(" ")[2])
            solves[row["lanes"]] = row
    for lanes in (1, 16):
        values = stages_at(medians, lanes)
        iterate = medians[(lanes, "whole_solve")] / solves[lanes]["max_steps"]
        whole = sum(values[name] for name in NAMES) / iterate
        print(f"CLOSURE lanes={lanes} all_stages={whole:.3f}", flush=True)
        for dropped in NAMES:
            kept = sum(values[name] for name in NAMES if name != dropped)
            print(
                f"DROPPED lanes={lanes} stage={dropped!r} "
                f"closure={kept / iterate:.3f} "
                f"shift={100.0 * (kept / iterate - whole):.1f}%",
                flush=True,
            )
        shifts = {
            dropped: (
                sum(values[name] for name in NAMES if name != dropped)
                - sum(values[name] for name in NAMES)
            )
            / iterate
            for dropped in NAMES
        }
        biggest = max(shifts, key=lambda name: abs(shifts[name]))
        print(
            f"CONTROL lanes={lanes} largest_shift_stage={biggest!r} "
            f"shift={100.0 * shifts[biggest]:.1f}% "
            "immaterial="
            f"{sorted(name for name, shift in shifts.items() if abs(shift) < 0.15)}",
            flush=True,
        )
        assert abs(shifts[biggest]) > 0.15, "no stage moved the closure check"


if __name__ == "__main__":
    main()
