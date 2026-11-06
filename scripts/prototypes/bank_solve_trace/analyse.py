"""Reconcile device events with host windows and their compiler source owners."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import jax.profiler

from scripts.prototypes.bank_solve_trace.groups import GROUPS, group
from scripts.prototypes.bank_solve_trace.provenance import metadata


def events(directory):
    paths = list(directory.rglob("*.xplane.pb"))
    assert len(paths) == 1, paths
    data = jax.profiler.ProfileData.from_file(str(paths[0]))
    device, host = [], []
    inventory = []
    for plane in data.planes:
        for line in plane.lines:
            rows = [
                dict(
                    name=e.name,
                    start=e.start_ns,
                    end=e.end_ns,
                    duration=e.duration_ns,
                    stats=dict(e.stats),
                )
                for e in line.events
            ]
            inventory.append((plane.name, line.name, len(rows)))
            if plane.name.startswith("/device:GPU") and line.name.startswith("Stream"):
                device.extend(rows)
            elif plane.name == "/host:CPU":
                host.append(rows)
    return device, host, inventory


def contexts(host, instructions):
    result = {}
    for rows in host:
        stack = []
        for event in sorted(rows, key=lambda e: (e["start"], -e["duration"])):
            while stack and event["end"] > stack[-1]["end"]:
                stack.pop()
            if event["name"].startswith("command_buffer_"):
                owners = [e["name"] for e in stack if e["name"] in instructions]
                result[event["name"]] = owners
            stack.append(event)
    return result


def prefix(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def copy_bytes(event):
    match = re.search(r"\bsize:(\d+)", event["stats"].get("memcpy_details", ""))
    if event["name"].startswith("Memcpy"):
        assert match, event
    return int(match[1]) if match else 0


def analyse(run, lanes):
    instructions = metadata(run / f"solve-{lanes}-0.hlo.pb")
    aliases = defaultdict(set)
    for name, item in instructions.items():
        if item["opcode"] in ("fusion", "custom-call", "sort", "copy"):
            for alias in (name, item["deduplicated_name"]):
                if alias:
                    aliases[re.sub(r"[^a-zA-Z0-9_]", "_", alias)].add(name)
    device, host, inventory = events(run / f"trace-{lanes}")
    windows = [e for rows in host for e in rows if e["name"] == f"solve_{lanes}_0"]
    assert len(windows) == 1 and device
    window = windows[0]
    graph_context = contexts(host, instructions)
    device = [e for e in device if window["start"] <= e["start"] < window["end"]]
    assert all(e["end"] <= window["end"] for e in device)
    totals = {name: dict(device_ms=0.0, launches=0) for name in GROUPS}
    ledger = {}
    copies = defaultdict(lambda: dict(events=0, bytes=0, device_ms=0.0))
    for event in device:
        stats = event["stats"]
        if "kernel_details" not in stats:
            row = copies[event["name"]]
            row["events"] += 1
            row["bytes"] += copy_bytes(event)
            row["device_ms"] += event["duration"] / 1e6
            continue
        key = (event["name"], stats.get("hlo_op", ""))
        if key not in ledger:
            hlo = key[1]
            owners = graph_context.get(hlo, [])
            candidates = [hlo] if hlo in instructions else sorted(aliases[key[0]])
            groups = {group(instructions[name]) for name in candidates}
            method = "hlo_op" if hlo in instructions else "kernel symbol"
            if hlo not in instructions and owners:
                context_group = group(instructions[owners[-1]])
                contained = {
                    "read census",
                    "read wall events",
                    "read fragment selection",
                    "step_jacobian",
                    "step_residual",
                }
                if context_group in contained or (
                    context_group == "read candidate loop"
                    and not groups
                    & {
                        "read curvature stencil",
                        "read normal form + set",
                        "read fragment selection",
                    }
                ):
                    groups = {context_group}
                    method = "enclosing HLO loop"
            if len(groups) != 1 or groups == {"unbound compiler/library"}:
                if owners:
                    context = instructions[owners[-1]]
                    scores = {
                        name: prefix(instructions[name]["stack"], context["stack"])
                        for name in candidates
                    }
                    if scores and max(scores.values()) > 0:
                        candidates = [
                            name
                            for name in candidates
                            if scores[name] == max(scores.values())
                        ]
                        groups = {group(instructions[name]) for name in candidates}
                        method = "kernel symbol + enclosing HLO loop"
                    if not groups or groups == {"unbound compiler/library"}:
                        groups = {group(context)}
                        method = "enclosing HLO loop"
            label = (
                next(iter(groups))
                if len(groups) == 1
                else "shared: " + " / ".join(sorted(groups))
            )
            if not groups:
                label = "unbound compiler/library"
            ledger[key] = dict(
                kernel=key[0],
                hlo_op=hlo,
                group=label,
                method=method,
                candidates=candidates,
                enclosing=owners,
                launches=0,
                device_ms=0.0,
            )
            binding_names = (
                owners[-1:] if method == "enclosing HLO loop" else candidates
            )
            ledger[key]["source_bindings"] = sorted(
                {
                    f"{frame['file'].split('/nova/equilibrium/')[-1]}:"
                    f"{frame['line']} ({frame['function']})"
                    for name in binding_names
                    for frame in instructions[name]["stack"][-1:]
                }
            )
        row = ledger[key]
        row["launches"] += 1
        row["device_ms"] += event["duration"] / 1e6
        total = totals.setdefault(row["group"], dict(device_ms=0.0, launches=0))
        total["launches"] += 1
        total["device_ms"] += event["duration"] / 1e6
    intervals = sorted((e["start"], e["end"]) for e in device)
    gaps, overlap, cursor = [], 0.0, window["start"]
    for start, end in intervals:
        if start > cursor:
            gaps.append(start - cursor)
        overlap += max(0.0, min(end, cursor) - start)
        cursor = max(cursor, end)
    suffix = window["end"] - cursor
    gaps.append(suffix)
    device_ms = sum(e["duration"] for e in device) / 1e6
    receipt = json.loads((run / "capture.json").read_text())[str(lanes)]
    wall = receipt["profiled_ms"][0]
    accounted = device_ms + sum(gaps) / 1e6 - overlap / 1e6
    row = dict(
        lanes=lanes,
        groups=totals,
        copies=dict(copies),
        device_ms=device_ms,
        kernel_launches=sum(v["launches"] for v in totals.values()),
        gap_ms=sum(gaps) / 1e6,
        prefix_ms=(intervals[0][0] - window["start"]) / 1e6,
        suffix_ms=suffix / 1e6,
        between_ms=sum(gaps[1:-1]) / 1e6,
        gap_count=len(gaps),
        overlap_ms=overlap / 1e6,
        profiled_ms=wall,
        annotation_ms=window["duration"] / 1e6,
        closure_residual_ms=accounted - wall,
        closure_fraction=(accounted - wall) / wall,
        inventory=inventory,
        attribution_methods=dict(Counter(v["method"] for v in ledger.values())),
    )
    validate_accounting(row)
    (run / f"kernels-{lanes}.json").write_text(
        json.dumps(list(ledger.values()), indent=2)
    )
    return row


def validate_accounting(row):
    grouped = sum(value["device_ms"] for value in row["groups"].values())
    grouped += sum(value["device_ms"] for value in row["copies"].values())
    assert abs(grouped - row["device_ms"]) < 1e-6, "device attribution incomplete"
    accounted = grouped + row["gap_ms"] - row["overlap_ms"]
    assert abs(accounted - row["profiled_ms"]) <= 0.1 * row["profiled_ms"], (
        "device plus observed gaps does not close",
        accounted,
        row["profiled_ms"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    device, _, inventory = events(args.run / "control")
    transferred = sum(copy_bytes(e) for e in device if e["name"] == "MemcpyH2D")
    kernels = [e for e in device if "kernel_details" in e["stats"]]
    assert transferred == 2097152 and kernels, (transferred, len(kernels))
    print("POSITIVE_CONTROL", transferred, len(kernels), flush=True)
    rows = [analyse(args.run, lanes) for lanes in (1, 16)]
    result = dict(
        control=dict(h2d_bytes=transferred, kernels=len(kernels), inventory=inventory),
        solves=rows,
    )
    (args.run / "attribution.json").write_text(json.dumps(result, indent=2))
    for row in rows:
        print(
            "SUMMARY",
            json.dumps(
                {k: v for k, v in row.items() if k not in ("inventory", "groups")}
            ),
            flush=True,
        )
        for name, values in sorted(
            row["groups"].items(), key=lambda item: -item[1]["device_ms"]
        ):
            print("GROUP", row["lanes"], name, values, flush=True)


if __name__ == "__main__":
    main()
