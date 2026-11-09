"""Reduce the observed record to the fields a reader checks, without the masks.

The event logs carry every per-lane array, including the 550-cell fragment and
membership masks that no reader re-derives by eye; they stay in the run
directory. What a reader needs is the scalar path: the residual at each
update, every trial's scale and residual, the retained scale, and the counts
that describe the discrete decision.
"""

import argparse
import json
from pathlib import Path


def decision(record):
    return {
        key: record[key]
        for key in (
            "valid",
            "reason",
            "boundary_class",
            "cut_cells",
            "normal_cells",
            "fragments",
            "admitted_saddles",
            "qualified",
        )
    } | {
        "axis": [record["axis_x"], record["axis_y"]],
        "axis_flux": record["axis_flux"],
        "boundary_flux": record["boundary_flux"],
    }


def lane(arm, index):
    return {
        "revision": arm["header"].get("revision"),
        "amplitude": arm["header"].get("amplitude"),
        "lane": index,
        "seed_residual": arm["summary"]["fresh_before"][index],
        "terminal_residual": arm["summary"]["terminal_residual"][index],
        "reads": arm["summary"]["reads"],
        "updates": arm["summary"]["updates"],
        "converged": arm["summary"]["converged"],
        "residual_path": [arm["summary"]["fresh_before"][index]]
        + [row["residual"] for row in arm["table"] if row["update"] >= 0],
        "table": [
            {
                "update": row["update"],
                "incumbent": row.get("incumbent"),
                "accepted_scale": row["accepted_scale"],
                "accepted_residual": row["residual"],
                "trials": [
                    {
                        "scale": trial["scale"],
                        "residual": trial["residual"],
                        "valid": trial["valid"],
                    }
                    for trial in row["trials"]
                ],
                "refactored": row["refactored"],
                "krylov_condition": row["krylov_condition"],
                "krylov_iterations": row["krylov_iterations"],
                "cumulative_reads": row["reads"],
                "decision": decision(row["accepted_decision"]),
            }
            for row in arm["table"]
            if row["update"] >= 0
        ],
        "seed_decision": decision(arm["table"][0]["decision"]),
    }


def reference(path):
    """The amplitude-1.0 control arm, from the summary its own job emitted."""
    for line in path.read_text().splitlines():
        if line.startswith("SUMMARY "):
            summary = json.loads(line.split(" ", 1)[1])
            return {
                "revision": None,
                "amplitude": summary["amplitude"],
                "seed_residual": summary["fresh_before"][0],
                "terminal_residual": max(summary["terminal_residual"]),
                "reads": summary["reads"][0],
                "updates": summary["updates"][0],
                "converged": sum(summary["converged"]),
                "lanes": len(summary["converged"]),
            }
    raise SystemExit(f"no SUMMARY line in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--unity", type=Path)
    args = parser.parse_args()
    record = json.loads(args.report.read_text())
    reduced = {name: lane(record[name], args.lane) for name in ("warm", "cold")}
    if args.unity is not None:
        reduced["reference"] = reference(args.unity)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reduced, indent=1))
    print(f"WROTE {args.out} {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
