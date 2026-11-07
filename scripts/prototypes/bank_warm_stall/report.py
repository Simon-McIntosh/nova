"""Line up the observed events into one record per arm, update and trial.

The driver traces one loop body, so every callback in it carries the same
trace-time identity and only its execution order distinguishes one update
from the next. The order within one update is fixed by the driver: the linear
solve once per lane, then per trial the fresh read that scored it followed by
the trial's own scale and norm, then the scale the search retained, then the
carried update. Parsing that order is all this module adds; the arithmetic is
the solver's own.
"""

import argparse
import json
from pathlib import Path

DECISION_FIELDS = (
    "valid",
    "reason",
    "boundary_class",
    "cut_cells",
    "normal_cells",
    "fragments",
    "fragment_selected",
    "admitted_saddles",
    "qualified",
    "axis_x",
    "axis_y",
    "axis_flux",
    "boundary_flux",
)
KINDS = ("READ", "STEP", "TRIAL", "SEARCH", "UPDATE")


def parse(path):
    events = []
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            body = json.loads(line)
            events.append((body["kind"], body))
        else:
            label, _, body = line.partition(" ")
            events.append((label, json.loads(body) if body else {}))
    return events


def describe(body, index):
    values = body["lanes"]
    return {
        name: values[position][index] for position, name in enumerate(DECISION_FIELDS)
    }


def arm(path, index):
    events = parse(path)
    header = next(body for label, body in events if label == "HEADER")
    summary = next((body for label, body in events if label == "SUMMARY"), None)
    if summary is None:
        raise SystemExit(f"no SUMMARY in {path}: the solve did not finish")
    stream = [(label, body) for label, body in events if label in KINDS]
    reads = [body for label, body in stream if label == "READ"]
    seed = reads[0]
    cursor = 1
    table = [
        {
            "update": -1,
            "residual": summary["fresh_before"][index],
            "decision": describe(seed, index),
        }
    ]
    steps, trials, search = [], [], None
    for label, body in stream:
        if label == "STEP":
            steps.append(body["lanes"])
        elif label == "READ":
            pass
        elif label == "TRIAL":
            lanes = body["lanes"]
            trials.append(
                {
                    "scale": lanes[0][index],
                    "residual": lanes[1][index],
                    "valid": bool(lanes[2][index]),
                    "decision": describe(reads[cursor], index),
                }
            )
            cursor += 1
        elif label == "SEARCH":
            lanes = body["lanes"]
            search = {
                "incumbent": lanes[0][index],
                "accepted": lanes[1][index],
                "scale": lanes[2][index],
            }
        elif label == "UPDATE":
            lanes = body["lanes"]
            accepted_at = (
                min(
                    range(len(trials)),
                    key=lambda slot: abs(trials[slot]["scale"] - search["scale"]),
                    default=None,
                )
                if trials
                else None
            )
            linear = steps[index] if index < len(steps) else [None] * 6
            table.append(
                {
                    "update": len(table) - 1,
                    "residual": search["accepted"],
                    "incumbent": search["incumbent"],
                    "accepted_scale": search["scale"],
                    "trials": list(trials),
                    "accepted_decision": (
                        trials[accepted_at]["decision"]
                        if accepted_at is not None
                        else trials[0]["decision"]
                        if trials
                        else None
                    ),
                    "refactored": linear[0],
                    "krylov_condition": linear[1],
                    "krylov_residual": linear[2],
                    "krylov_iterations": linear[3],
                    "linear_residual": linear[4],
                    "linear_ok": linear[5],
                    "reads": lanes[3][index],
                    "refactors": lanes[2][index],
                    "active_after": lanes[6][index],
                    "lane_steps": len(steps),
                }
            )
            steps, trials, search = [], [], None
    return {"header": header, "summary": summary, "lane": index, "table": table}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = {
        name: arm(args.run / f"bank-warm-stall-{name}.jsonl", args.lane)
        for name in ("warm", "cold")
    }
    args.out.write_text(json.dumps(record, indent=1))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
