"""Tables and oracle verdicts from the slice receipts.

Reads the builder's cold slice 0 and the two fresh runner processes, and
reports per step and slice the batch reads, updates, refactors, the warm-start
fresh residual before the first update, the whole-slice wall and the wall per
slice, with the cold slice-0 figures beside them. Every converged slice is
judged against the locked precision budget on its own drive amplitude: the
analytic oracle at unit drive, scaled by the slice's amplitude.
"""

import argparse
import json
from pathlib import Path

BUDGET_GEOMETRY = 5e-4
BUDGET_FLUX = 1e-4
BUDGET_RESIDUAL = 0.0  # terminal residual is already held to 1e-10 by the solve


def load(path):
    receipts = {"HEADER": [], "SLICE": [], "SCALING": []}
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        label, payload = line.split(" ", 1)
        receipts.setdefault(label, []).append(json.loads(payload))
    return receipts


def summarise(rows, minor_radius):
    """One row per (step, slice) over the stated process's lanes."""
    table = []
    for row in rows:
        reads = row["reads"]
        steps = row["updates"]
        residual = row["terminal_residual"]
        converged = row["converged"]
        budget = BUDGET_GEOMETRY * minor_radius
        geometry_ok = max(
            max(row["axis_error"]),
            max(row["saddle_error"]),
        ) <= budget
        flux_ok = max(abs(value) for value in row["boundary_error"]) <= (
            BUDGET_FLUX * abs(row["span"][0])
        )
        residual_ok = max(residual[index] for index in range(len(residual))) <= 1e-10
        table.append(
            {
                "step": row["step"],
                "slice": row["slice"],
                "amplitude": row["amplitude"],
                "reads_mean": sum(reads) / len(reads),
                "reads_max": max(reads),
                "reads": reads,
                "updates_mean": sum(steps) / len(steps),
                "updates_max": max(steps),
                "updates": steps,
                "refactors_mean": sum(row["refactors"]) / len(row["refactors"]),
                "fresh_before_max": max(row["fresh_before"]),
                "fresh_before": row["fresh_before"],
                "wall_ms": row["wall_ms"],
                "terminal_residual_max": max(residual),
                "converged": sum(1 for value in converged if value),
                "lanes": len(converged),
                "axis_error_max": max(row["axis_error"]),
                "saddle_error_max": max(row["saddle_error"]),
                "boundary_error_max": max(abs(v) for v in row["boundary_error"]),
                "span": row["span"],
                "geometry_ok": geometry_ok,
                "flux_ok": flux_ok,
                "residual_ok": residual_ok,
            }
        )
    return table


def arm_summary(table):
    """Mean and maximum over the warm slices, and the cold slice beside them."""
    cold = [row for row in table if row["slice"] == 0]
    warm = [row for row in table if row["slice"] > 0]
    summary = {}
    if warm:
        summary["reads_mean"] = sum(row["reads_mean"] for row in warm) / len(warm)
        summary["reads_max"] = max(row["reads_max"] for row in warm)
        summary["updates_mean"] = sum(row["updates_mean"] for row in warm) / len(warm)
        summary["updates_max"] = max(row["updates_max"] for row in warm)
        summary["refactors_mean"] = sum(row["refactors_mean"] for row in warm) / len(warm)
        summary["wall_ms_mean"] = sum(row["wall_ms"] for row in warm) / len(warm)
        summary["wall_ms_max"] = max(row["wall_ms"] for row in warm)
        summary["fresh_before_max"] = max(row["fresh_before_max"] for row in warm)
        summary["warm_slices"] = len(warm)
        summary["warm_slices_converged"] = sum(
            1 for row in warm if row["converged"] == row["lanes"]
        )
        summary["warm_slices_in_budget"] = sum(
            1
            for row in warm
            if row["geometry_ok"] and row["flux_ok"] and row["residual_ok"]
        )
        summary["warm_slices_one_update"] = sum(
            1 for row in warm if row["updates_max"] <= 1
        )
        summary["warm_slices_zero_update"] = sum(
            1 for row in warm if row["updates_max"] == 0
        )
    if cold:
        summary["cold"] = {
            "reads_mean": cold[0]["reads_mean"],
            "reads_max": cold[0]["reads_max"],
            "updates_mean": cold[0]["updates_mean"],
            "wall_ms": cold[0]["wall_ms"],
            "terminal_residual_max": cold[0]["terminal_residual_max"],
            "in_budget": cold[0]["geometry_ok"]
            and cold[0]["flux_ok"]
            and cold[0]["residual_ok"],
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for name in ("builder", "runner1", "runner2"):
        receipts = load(args.run / f"warm-start-{name}.jsonl")
        header = receipts["HEADER"][0] if receipts["HEADER"] else {}
        rows = receipts["SLICE"]
        table = summarise(rows, rows[0]["minor_radius"] if rows else 1.0)
        arms[name] = {
            "revision": header.get("revision"),
            "pid": header.get("pid"),
            "slices": len(table),
            "table": table,
            "summary": arm_summary(table),
        }
        if receipts["SCALING"]:
            arms[name]["scaling"] = receipts["SCALING"]
    attempted = [
        {"step": row["step"], "slice": row["slice"], **row}
        for row in arms["runner1"]["table"]
    ]
    summary = {"arms": arms, "runner1_rows": attempted}
    args.out.write_text(json.dumps(summary, indent=1, sort_keys=True))

    for name, arm in arms.items():
        print(f"== {name} revision={arm['revision']} pid={arm['pid']}")
        for row in arm["table"]:
            print(
                f"  step={row['step']:.0e} slice={row['slice']} amp={row['amplitude']:.6f} "
                f"reads={row['reads_mean']:.2f} (max {row['reads_max']}) "
                f"updates={row['updates_mean']:.2f} (max {row['updates_max']}) "
                f"refactors={row['refactors_mean']:.2f} "
                f"fresh_before={row['fresh_before_max']:.3e} "
                f"residual={row['terminal_residual_max']:.3e} "
                f"wall={row['wall_ms']:.3f} ms "
                f"axis={row['axis_error_max']:.3e} saddle={row['saddle_error_max']:.3e} "
                f"flux={row['boundary_error_max']:.3e} "
                f"converged={row['converged']}/{row['lanes']} "
                f"budget={'pass' if row['geometry_ok'] and row['flux_ok'] and row['residual_ok'] else 'FAIL'}"
            )
        print("  summary", json.dumps(arm["summary"], sort_keys=True))
        if "scaling" in arm:
            print("  scaling", json.dumps(arm["scaling"], sort_keys=True))


if __name__ == "__main__":
    main()