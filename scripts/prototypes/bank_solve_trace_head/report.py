"""Compare source-bound device inventories and check repeated-process evidence."""

import copy
from collections import Counter
from html import escape
import json
from pathlib import Path
import statistics
import sys

import matplotlib.pyplot as plt

from scripts.prototypes.bank_solve_profile.plot import save_checked
from scripts.prototypes.bank_solve_trace.analyse import events, validate_accounting
from scripts.prototypes.bank_solve_trace.groups import GROUPS
from scripts.prototypes.bank_solve_trace.provenance import metadata


def table(headers, rows):
    def cells(tag, values):
        return (
            "<tr>"
            + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values)
            + "</tr>"
        )

    return "\n".join(
        [
            "<table><thead>" + cells("th", headers) + "</thead><tbody>",
            *(cells("td", row) for row in rows),
            "</tbody></table>",
        ]
    )


def groups(row):
    values = dict(row["groups"])
    values["Device copies and fills"] = dict(
        device_ms=sum(v["device_ms"] for v in row["copies"].values()),
        launches=0,
    )
    return values


def invocations(run, lanes, reads):
    instructions = metadata(run / f"solve-{lanes}-0.hlo.pb")
    _, host, _ = events(run / f"trace-{lanes}")
    wall_loops = {
        name
        for name, item in instructions.items()
        if item["opcode"] == "while"
        and any(f["function"] == "_wall_events" for f in item["stack"])
    }
    counts = Counter(e["name"] for row in host for e in row if e["name"] in wall_loops)
    assert sum(counts.values()) == reads
    totals = dict(read=0.0, update=0.0)

    def context(name):
        functions = [f["function"] for f in instructions[name]["stack"]]
        if "newton_step" in functions:
            return "update"
        assert any(".magnitude" in f for f in functions), functions
        return "read"

    ledger = json.loads((run / f"kernels-{lanes}.json").read_text())
    for row in ledger:
        if row["group"] != "step_residual":
            continue
        names = (
            row["enclosing"][-1:]
            if row["method"] == "enclosing HLO loop"
            else row["candidates"]
        )
        owners = {context(name) for name in names}
        if len(owners) != 1 and row["enclosing"]:
            owners = {context(row["enclosing"][-1])}
        assert len(owners) == 1, row
        totals[owners.pop()] += row["device_ms"]
    strips = {}
    for group in ("step_jacobian", "step_residual", "read candidate loop"):
        counts_by_symbol = Counter()
        for row in ledger:
            if (
                row["group"] == group
                and "dynamic_update_slice" in row["kernel"]
                and any("quadratic_cell_fragments" in b for b in row["source_bindings"])
            ):
                counts_by_symbol[row["kernel"]] += row["launches"]
        assert counts_by_symbol
        strips[group] = dict(counts_by_symbol)
    result = dict(
        wall_loop_counts=dict(counts), residual_ms=totals, strip_kernels=strips
    )
    (run / f"invocations-{lanes}.json").write_text(json.dumps(result, indent=2))
    print("INVOCATIONS", run.name, lanes, json.dumps(result), flush=True)
    return totals


def main():
    run, morning, output = map(Path, sys.argv[1:])
    historic = json.loads((morning / "attribution.json").read_text())["solves"]
    captures = [
        json.loads((run / f"process-{p}/capture.json").read_text()) for p in (1, 2)
    ]
    current = [
        json.loads((run / f"process-{p}/attribution.json").read_text()) for p in (1, 2)
    ]
    for process, row in enumerate(current, 1):
        assert row["control"]["h2d_bytes"] == 2097152 and row["control"]["kernels"] > 0
        log = (run / f"capture-{process}.log").read_text()
        assert log.count("Persistent compilation cache hit for 'jit_solve'") == 2
    tables, summary = [], []
    wall_rows, unit_rows = [], []
    for index, lanes in enumerate((1, 16)):
        traces = [c["solves"][index] for c in current]
        receipts = [c[str(lanes)] for c in captures]
        assert receipts[0]["reads"] == receipts[1]["reads"]
        assert receipts[0]["steps"] == receipts[1]["steps"]
        assert traces[0]["kernel_launches"] == traces[1]["kernel_launches"]
        values = [groups(historic[index]), *(groups(t) for t in traces)]
        names = list(dict.fromkeys([*GROUPS, *(n for v in values for n in v)]))
        rows = []
        for name in names:
            cells = [name]
            for value, trace in zip(values, [historic[index], *traces], strict=True):
                value = value.get(name, dict(device_ms=0, launches=0))
                cells += [
                    f"{value['device_ms']:.3f}",
                    value["launches"],
                    f"{100 * value['device_ms'] / trace['device_ms']:.2f}%",
                ]
            rows.append(cells)
        rows.append(
            [
                "All device activity",
                *(
                    v
                    for t in [historic[index], *traces]
                    for v in (f"{t['device_ms']:.3f}", t["kernel_launches"], "100.00%")
                ),
            ]
        )
        tables.append(
            f"<h3>{lanes} slices: device attribution</h3>\n"
            + table(
                [
                    "Source group",
                    "Morning ms",
                    "Launches",
                    "Share",
                    "Process 1 ms",
                    "Launches",
                    "Share",
                    "Process 2 ms",
                    "Launches",
                    "Share",
                ],
                rows,
            )
        )
        for process, (trace, receipt) in enumerate(
            zip(traces, receipts, strict=True), 1
        ):
            validate_accounting(trace)
            reads, updates = max(receipt["reads"]), max(receipt["steps"])
            wall = statistics.median(receipt["unprofiled_ms"])
            assert all(receipt["converged"]) and max(receipt["residual"]) <= 1e-10
            wall_rows.append(
                [
                    lanes,
                    process,
                    ", ".join(map(str, receipt["reads"])),
                    ", ".join(map(str, receipt["steps"])),
                    f"{wall:.3f}",
                    f"{wall / lanes:.3f}",
                    f"{wall / reads:.3f}",
                    f"{max(receipt['residual']):.6g}",
                ]
            )
            g = trace["groups"]
            read_ms = sum(v["device_ms"] for k, v in g.items() if k.startswith("read "))
            split = invocations(run / f"process-{process}", lanes, reads)
            assert abs(sum(split.values()) - g["step_residual"]["device_ms"]) < 1e-6
            residual = split["update"] / updates
            certification = split["read"] / reads
            jacobian = g["step_jacobian"]["device_ms"] / updates
            linear = g["lu_solve + update"]["device_ms"] / updates
            scaling = g["equilibrate + lu_factor"]["device_ms"] / updates
            unit_rows.append(
                [
                    lanes,
                    process,
                    reads,
                    updates,
                    f"{read_ms / reads:.3f}",
                    f"{certification:.3f}",
                    f"{residual:.3f}",
                    f"{jacobian:.3f}",
                    f"{linear:.3f}",
                    f"{scaling:.3f}",
                    f"{residual + jacobian + linear + scaling:.3f}",
                ]
            )
            summary.append(
                dict(
                    lanes=lanes,
                    process=process,
                    reads=receipt["reads"],
                    updates=updates,
                    wall_ms=wall,
                    device_ms=trace["device_ms"],
                    read_ms=read_ms / reads,
                    residual_ms=residual,
                    certification_ms=certification,
                    jacobian_ms=jacobian,
                    linear_ms=linear,
                    scaling_ms=scaling,
                )
            )
    tables.append(
        "<h3>Fresh-process counters and unprofiled wall</h3>\n"
        + table(
            [
                "Slices",
                "Process",
                "Reads per lane",
                "Updates per lane",
                "Batch ms",
                "ms/slice",
                "ms/batch read",
                "Maximum fresh residual",
            ],
            wall_rows,
        )
    )
    tables.append(
        "<h3>Attributed device cost per invocation</h3>\n"
        + table(
            [
                "Slices",
                "Process",
                "Batch reads",
                "Batch updates",
                "Read ms/read",
                "Fresh residual ms/read",
                "Step residual ms/update",
                "Jacobian ms/update",
                "GMRES + update ms/update",
                "Scaling ms/update",
                "Step sum ms/update",
            ],
            unit_rows,
        )
    )
    closure = []
    for process, current_row in enumerate(current, 1):
        for t in current_row["solves"]:
            closure.append(
                [
                    t["lanes"],
                    process,
                    *(
                        f"{v:.6f}"
                        for v in (
                            t["device_ms"],
                            t["overlap_ms"],
                            t["gap_ms"],
                            t["device_ms"] - t["overlap_ms"] + t["gap_ms"],
                            t["profiled_ms"],
                            t["closure_residual_ms"],
                            100 * t["closure_fraction"],
                        )
                    ),
                ]
            )
    tables.append(
        "<h3>Closure against the profiled wall</h3>\n"
        + table(
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
            closure,
        )
    )
    damaged = copy.deepcopy(current[0]["solves"][1])
    removed = max(damaged["groups"], key=lambda k: damaged["groups"][k]["device_ms"])
    del damaged["groups"][removed]
    try:
        validate_accounting(damaged)
    except AssertionError as error:
        print("OMISSION_REFUSED", removed, str(error), flush=True)
    else:
        raise AssertionError("omitted source group escaped accounting")
    (run / "tables.html").write_text("\n".join(tables))
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    values = []
    for row in (historic[1], current[0]["solves"][1], current[1]["solves"][1]):
        g = groups(row)
        v = {k: t["device_ms"] for k, t in g.items() if not k.startswith("shared:")}
        v["shared fused operations"] = sum(
            t["device_ms"] for k, t in g.items() if k.startswith("shared:")
        )
        values.append(v)
    names = sorted(values[0], key=lambda n: -max(v[n] for v in values))
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    for position, subset in zip(
        ((0.24, 0.14, 0.26, 0.74), (0.77, 0.14, 0.21, 0.74)),
        (names[:8], names[8:]),
        strict=True,
    ):
        ax = fig.add_axes(position)
        for offset, value, marker, colour, label in (
            (-0.16, values[0], "s", "#929292", "Morning"),
            (
                0.16,
                {n: statistics.mean(v[n] for v in values[1:]) for n in names},
                "o",
                "#1f5d92",
                "Current",
            ),
        ):
            ax.plot(
                [value[n] for n in subset],
                [i + offset for i in range(len(subset))],
                marker,
                color=colour,
                markersize=7,
                label=label,
                zorder=3 if offset > 0 else 2,
            )
        labels = [
            n.replace("lu_solve + update", "GMRES / update (LU before)")
            .replace("equilibrate + lu_factor", "equilibration / LU")
            .replace("step_", "step ")
            for n in subset
        ]
        ax.set_yticks(range(len(subset)), labels, fontsize=14)
        ax.invert_yaxis()
        ax.set_xlim(left=-0.025 * max(v[n] for v in values for n in subset))
        ax.set_xticks([v for v in ax.get_xticks() if 0 <= v <= ax.get_xlim()[1]])
        ax.set_xlabel("Device time per solve [ms]", fontsize=16)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save_checked(fig, output)
    fig.savefig(run / "device-time.png", dpi=100)
    plt.close(fig)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
