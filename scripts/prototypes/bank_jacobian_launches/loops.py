"""Resolve Jacobian launches to nested loop owners using compiler frames."""

from collections import defaultdict
import json
from pathlib import Path
import sys

from scripts.prototypes.bank_solve_trace.provenance import metadata


def owner(frames):
    names = {frame["function"] for frame in frames}
    if any(".propagate" in name for name in names):
        return "conic strip propagation"
    if "_conic_ring_tangents" in names:
        return "channel-vmapped ring tangents"
    if any(name.startswith("_integrate_live_rings") for name in names):
        return "live-ring groups"
    if "_conic_rings" in names:
        return "conic ring construction"
    if any(name.startswith("_integrate_bucket") for name in names):
        return "cut-cell tiles and normal rings"
    return "other Jacobian work"


def analyse(run):
    result = {}
    for lanes in (1, 16):
        instructions = metadata(run / f"solve-{lanes}-0.hlo.pb")
        kernels = json.loads((run / f"kernels-{lanes}.json").read_text())
        capture = json.loads((run / "capture.json").read_text())[str(lanes)]
        calls = max(capture["steps"])
        totals = defaultdict(lambda: dict(launches=0, device_ms=0.0))
        detail = []
        for row in kernels:
            if row["group"] != "step_jacobian":
                continue
            names = row["candidates"]
            if row["method"] == "enclosing HLO loop":
                names = row["enclosing"][-1:]
            labels = {owner(instructions[name]["stack"]) for name in names}
            label = (
                next(iter(labels))
                if len(labels) == 1
                else "shared: " + " / ".join(sorted(labels))
            )
            total = totals[label]
            total["launches"] += row["launches"]
            total["device_ms"] += row["device_ms"]
            detail.append(dict(row, loop_owner=label))
        assert calls > 0 and sum(row["launches"] for row in totals.values()) > 0
        result[str(lanes)] = dict(calls=calls, groups=dict(totals))
        for name, row in totals.items():
            print(
                "LOOP",
                lanes,
                name,
                "launches_per_call",
                row["launches"] / calls,
                "device_ms_per_call",
                row["device_ms"] / calls,
                flush=True,
            )
        (run / f"loop-kernels-{lanes}.json").write_text(json.dumps(detail, indent=2))
    (run / "loops.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    analyse(Path(sys.argv[1]))
