"""Compare complete trace inventories and separate read and update costs."""

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median
import sys

import matplotlib.pyplot as plt

import nova
from scripts.prototypes.bank_solve_trace_head.report import (
    groups,
    events,
    metadata,
    save_checked,
    table,
    validate_accounting,
)


def load(path):
    return json.loads(path.read_text())


def omit_largest(trace):
    damaged = copy.deepcopy(trace)
    removed = max(damaged["groups"], key=lambda k: damaged["groups"][k]["device_ms"])
    del damaged["groups"][removed]
    print("OMITTED_GROUP", removed, flush=True)
    return damaged, removed


def residual_split(run, lanes, reads, instructions):
    """Partition residual kernels by their compiler-recorded calling scope."""
    _, host, _ = events(run / f"trace-{lanes}")
    read_loops = {
        name
        for name, item in instructions.items()
        if item["opcode"] == "while"
        and any(f["function"] == "_wall_events" for f in item["stack"])
    }
    counts = Counter(
        e["name"] for line in host for e in line if e["name"] in read_loops
    )
    assert sum(counts.values()) == reads, counts
    owners = {}
    for name, item in instructions.items():
        names = {f["function"] for f in item["stack"]}
        owners[name] = (
            "update"
            if "newton_step" in names
            else ("read" if any(".magnitude" in f for f in names) else None)
        )
    totals = dict(read=0.0, update=0.0)
    for row in load(run / f"kernels-{lanes}.json"):
        if row["group"] != "step_residual":
            continue
        names = (
            row["enclosing"][-1:]
            if row["method"] == "enclosing HLO loop"
            else row["candidates"]
        )
        candidates = {owners[n] for n in names}
        if len(candidates) != 1 and row["enclosing"]:
            candidates = {owners[row["enclosing"][-1]]}
        assert len(candidates) == 1 and None not in candidates, row
        totals[candidates.pop()] += row["device_ms"]
    receipt = dict(wall_loop_counts=dict(counts), residual_ms=totals)
    (run / f"invocations-{lanes}.json").write_text(json.dumps(receipt, indent=2))
    print("INVOCATIONS", run.name, lanes, receipt, flush=True)
    return totals


def figure(traces, output, preview):
    values = []
    for pair in (traces[:2], traces[2:]):
        names = set().union(*(groups(row) for row in pair))
        value = {
            name: mean(groups(row).get(name, {}).get("device_ms", 0) for row in pair)
            for name in names
            if not name.startswith("shared:")
        }
        value["shared fused operations"] = mean(
            sum(
                v["device_ms"]
                for k, v in row["groups"].items()
                if k.startswith("shared:")
            )
            for row in pair
        )
        values.append(value)
    names = sorted(
        set().union(*values), key=lambda n: -max(v.get(n, 0) for v in values)
    )
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    for position, subset in zip(
        ((0.24, 0.14, 0.26, 0.74), (0.77, 0.14, 0.21, 0.74)),
        (names[:8], names[8:]),
        strict=True,
    ):
        ax = fig.add_axes(position)
        for offset, value, marker, colour, label in (
            (-0.16, values[0], "s", "#929292", "Previous trace"),
            (0.16, values[1], "o", "#1f5d92", "Four reads"),
        ):
            ax.plot(
                [value.get(n, 0) for n in subset],
                [i + offset for i in range(len(subset))],
                marker,
                color=colour,
                markersize=7,
                label=label,
                zorder=3 if offset > 0 else 2,
            )
        labels = [
            n.replace("lu_solve + update", "GMRES + update")
            .replace("equilibrate + lu_factor", "equilibration")
            .replace("step_", "step ")
            for n in subset
        ]
        ax.set_yticks(range(len(subset)), labels, fontsize=14)
        ax.invert_yaxis()
        ax.set_xlim(left=-0.025 * max(v.get(n, 0) for v in values for n in subset))
        ax.set_xticks([v for v in ax.get_xticks() if 0 <= v <= ax.get_xlim()[1]])
        ax.set_xlabel("Device time per solve [ms]", fontsize=16)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save_checked(fig, output)
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    fig.savefig(preview, dpi=100)
    plt.close(fig)


def main():
    run, previous, output = map(Path, sys.argv[1:])
    print("HEADER", nova.__file__, __file__, Path.cwd().resolve(), flush=True)
    current = [load(run / f"process-{p}/attribution.json") for p in (1, 2)]
    historic = [load(previous / f"process-{p}/attribution.json") for p in (1, 2)]
    captures = [load(run / f"process-{p}/capture.json") for p in (1, 2)]
    prior_captures = [load(previous / f"process-{p}/capture.json") for p in (1, 2)]
    digest = hashlib.sha256((run / "common-seeds.npz").read_bytes()).hexdigest()
    keys = []
    for process in (0, 1, 2):
        log = (run / f"capture-{process}.log").read_text()
        assert digest in log and "XLA_FLAGS=unset" in log and log.endswith("EXIT=0\n")
        hits = set(re.findall(r"cache hit for 'jit_solve' with key '([^']+)'", log))
        if process:
            assert len(hits) == 2, hits
            keys.append(hits)
    assert keys[0] == keys[1]
    for row in current:
        assert row["control"]["h2d_bytes"] == 2097152 and row["control"]["kernels"] > 0
    tables, summary, walls, units, closures, launch_rows = [], [], [], [], [], []
    for index, lanes in enumerate((1, 16)):
        traces = [r["solves"][index] for r in historic + current]
        inventories = [groups(t) for t in traces]
        names = list(dict.fromkeys(n for g in inventories for n in g))
        rows = []
        for name in names:
            cells = [name]
            for g, trace in zip(inventories, traces, strict=True):
                value = g.get(name, dict(device_ms=0, launches=0))
                cells.append(
                    f"{value['device_ms']:.3f} / {value['launches']} / "
                    f"{100 * value['device_ms'] / trace['device_ms']:.2f}%"
                )
            rows.append(cells)
        rows.append(
            [
                "All device activity",
                *(
                    f"{t['device_ms']:.3f} / {t['kernel_launches']} / 100%"
                    for t in traces
                ),
            ]
        )
        tables.append(
            f"<h3>{lanes} slices: device ms / kernel launches / device share</h3>\n"
            + table(
                [
                    "Source group",
                    "Previous p1",
                    "Previous p2",
                    "Current p1",
                    "Current p2",
                ],
                rows,
            )
        )
        for label, pair in (("Previous", prior_captures), ("Current", captures)):
            for p, capture in enumerate(pair, 1):
                receipt = capture[str(lanes)]
                wall = median(receipt["unprofiled_ms"])
                walls.append(
                    [
                        label,
                        lanes,
                        p,
                        receipt["reads"],
                        receipt["steps"],
                        f"{wall:.3f}",
                        f"{wall / lanes:.3f}",
                        f"{wall / max(receipt['reads']):.3f}",
                    ]
                )
        for p, (trace, capture) in enumerate(zip(traces[2:], captures, strict=True), 1):
            validate_accounting(trace)
            receipt = capture[str(lanes)]
            assert all(receipt["converged"]) and max(receipt["residual"]) <= 1e-10
            reads, updates = max(receipt["reads"]), max(receipt["steps"])
            assert reads == 4, receipt
            instructions = metadata(run / f"process-{p}/solve-{lanes}-0.hlo.pb")
            split = residual_split(run / f"process-{p}", lanes, reads, instructions)
            functions = {
                f["function"] for item in instructions.values() for f in item["stack"]
            }
            markers = {
                name: sum(name in f for f in functions)
                for name in ("curvature_coefficients", "_block_gmres")
            }
            assert all(markers.values()), markers
            print("COMPILED_MARKERS", lanes, p, markers, flush=True)
            g = trace["groups"]
            assert abs(sum(split.values()) - g["step_residual"]["device_ms"]) < 1e-6
            row = dict(
                lanes=lanes,
                process=p,
                reads=reads,
                updates=updates,
                read_ms=sum(
                    v["device_ms"] for k, v in g.items() if k.startswith("read ")
                )
                / reads,
                certification_ms=split["read"] / reads,
                residual_ms=split["update"] / updates,
                jacobian_ms=g["step_jacobian"]["device_ms"] / updates,
                linear_ms=g["lu_solve + update"]["device_ms"] / updates,
                scaling_ms=g["equilibrate + lu_factor"]["device_ms"] / updates,
            )
            row["update_ms"] = sum(
                row[k]
                for k in ("residual_ms", "jacobian_ms", "linear_ms", "scaling_ms")
            )
            row["floor_ms"] = (
                4 * (row["read_ms"] + row["certification_ms"])
                + updates * row["update_ms"]
            )
            row["remaining_device_ms"] = trace["device_ms"] - row["floor_ms"]
            row["wall_ms"] = median(receipt["unprofiled_ms"])
            row["max_residual"] = max(receipt["residual"])
            summary.append(row)
            units.append(
                [
                    lanes,
                    p,
                    reads,
                    updates,
                    *(
                        f"{row[k]:.3f}"
                        for k in (
                            "read_ms",
                            "certification_ms",
                            "residual_ms",
                            "jacobian_ms",
                            "linear_ms",
                            "scaling_ms",
                            "update_ms",
                            "floor_ms",
                        )
                    ),
                    f"{row['floor_ms'] / lanes:.3f}",
                ]
            )
            closures.append(
                [
                    lanes,
                    p,
                    *(
                        f"{v:.6f}"
                        for v in (
                            trace["device_ms"],
                            trace["overlap_ms"],
                            trace["gap_ms"],
                            trace["device_ms"] - trace["overlap_ms"] + trace["gap_ms"],
                            trace["profiled_ms"],
                            trace["closure_residual_ms"],
                            100 * trace["closure_fraction"],
                        )
                    ),
                ]
            )
            for name, calls in (
                ("step_jacobian", updates),
                ("step_residual", reads + updates),
                ("read candidate loop", reads),
            ):
                launch_rows.append(
                    [
                        lanes,
                        p,
                        name,
                        g[name]["launches"],
                        calls,
                        f"{g[name]['launches'] / calls:.3f}",
                    ]
                )
    for title, headers, rows in (
        (
            "Fresh-process counters and synchronized unprofiled wall",
            [
                "Trace",
                "Slices",
                "Process",
                "Reads per lane",
                "Updates per lane",
                "Batch ms",
                "ms/slice",
                "ms/batch read",
            ],
            walls,
        ),
        (
            "Read and update device costs",
            [
                "Slices",
                "Process",
                "Reads",
                "Updates",
                "Read ms/read",
                "Certification ms/read",
                "Residual ms/update",
                "Jacobian ms/update",
                "GMRES + update ms/update",
                "Scaling ms/update",
                "Total ms/update",
                "Four-read floor ms",
                "Floor ms/slice",
            ],
            units,
        ),
        (
            "Closure against the profiled wall",
            [
                "Slices",
                "Process",
                "Device ms",
                "Overlap ms",
                "Gaps ms",
                "Accounted ms",
                "Profiled ms",
                "Residual ms",
                "Residual %",
            ],
            closures,
        ),
        (
            "Launches per invocation",
            [
                "Slices",
                "Process",
                "Group",
                "Launches/solve",
                "Invocations",
                "Launches/invocation",
            ],
            launch_rows,
        ),
    ):
        tables.append(f"<h3>{title}</h3>\n" + table(headers, rows))
    damaged, removed = omit_largest(current[0]["solves"][1])
    try:
        validate_accounting(damaged)
    except AssertionError as error:
        print("OMISSION_REFUSED", removed, str(error), flush=True)
    else:
        raise AssertionError("omitted source group escaped accounting")
    (run / "tables.html").write_text("\n".join(tables))
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    figure(
        [r["solves"][1] for r in historic + current], output, run / "device-time.png"
    )
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--omit-largest"]:
        trace = load(Path(sys.argv[2]))["solves"][1]
        damaged, _ = omit_largest(trace)
        validate_accounting(damaged)
        print("OMISSION_ESCAPED", flush=True)
    else:
        main()
