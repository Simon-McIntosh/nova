"""Audit completed profile receipts and render their per-slice stage costs."""

import argparse
import ast
from collections import defaultdict
from html import escape
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
from statistics import mean
import subprocess


@lru_cache(None)
def revision_source(revision):
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(
        ["git", "-C", str(root), "show", revision + ":nova/equilibrium/support_read.py"]
    )


def measured_source(record):
    """Require the receipt's retained module to match its measured revision."""
    source = Path(record["module"])
    actual = source.read_bytes()
    digest = hashlib.sha256(actual).hexdigest()
    if digest != record["module_sha256"] or actual != revision_source(
        record["revision"]
    ):
        raise ValueError("measured module differs from receipt revision or digest")
    inputs = Path(record["inputs_path"])
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != record["inputs_sha256"]:
        raise ValueError("retained measurement input differs from receipt digest")
    return source


def attributed(record, directory, program):
    """Resolve deduplicated symbols through their executed enclosing branch."""
    from scripts.prototypes.bank_solve_trace.provenance import metadata

    source = measured_source(record)
    module = ast.parse(source.read_text())
    read = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "read_batch"
    )
    carry = next(
        n for n in read.body if isinstance(n, ast.FunctionDef) and n.name == "carry"
    )
    terminal = carry.body[-1]
    terminal_line = None
    if (
        isinstance(terminal, ast.Return)
        and isinstance(terminal.value, ast.Call)
        and isinstance(terminal.value.func, ast.Attribute)
        and terminal.value.func.attr == "cond"
    ):
        assert terminal.value.args[1].body.func.id == "full_verdict"
        assert (
            isinstance(terminal.value.args[2].body, ast.Name)
            and terminal.value.args[2].body.id == "carried"
        )
        terminal_line = terminal.lineno
    width = record["width"]
    profile = record["programs"][program]["profile"]
    instructions = metadata(directory / program / f"solve-{width}-0.hlo.pb")
    compiled_modules = {
        frame["file"]
        for item in instructions.values()
        for frame in item["stack"]
        if frame["file"].endswith("/nova/equilibrium/support_read.py")
    }
    original = record.get("module_original", record["module"])
    if compiled_modules != {original}:
        raise ValueError("compiled module paths differ from receipt module identity")
    ledger = json.loads((directory / program / f"kernels-{width}.json").read_text())
    totals = defaultdict(float)
    mechanisms = defaultdict(float)
    for row in ledger:
        candidates = [instructions[n] for n in row["candidates"]]
        contexts = [instructions[n] for n in row["enclosing"]]
        frames = [f for item in candidates for f in item["stack"]]
        names = {f["function"].split(".<locals>")[0] for f in frames}
        branch = None
        for item in reversed(contexts):
            if "full_geometry" in item["op_name"]:
                branch = "full"
                break
            if "carried_geometry" in item["op_name"]:
                branch = "carried"
                break
            for frame in reversed(item["stack"]):
                function = frame["function"]
                # The false certificate branch returns existing values; its
                # executed inner kernels can only belong to the full verdict.
                if (
                    function == "read_batch.<locals>.carry"
                    and frame["line"] == terminal_line
                ):
                    branch = "full"
                    break
                if (
                    "read_batch.<locals>.full_verdict" in function
                    or "read_batch.<locals>.whole_mesh" in function
                ):
                    branch = "full"
                    break
                if (
                    "read_batch.<locals>.carry" in function
                    or "refresh_at_stationary_points" in function
                ):
                    branch = "carried"
                    break
            if branch is not None:
                break
        if names & {
            "step_jacobian",
            "_normal_geometry_booking_tangent",
            "_local_booking_sensitivities",
        } or row["group"] in ("assembly", "step_jacobian"):
            label = "assembly"
        elif "step_residual" in names or row["group"] in (
            "update residual",
            "read residual",
        ):
            label = (
                "update residual"
                if "newton_step" in names or row["group"] == "update residual"
                else "read residual"
            )
        elif "newton_step" in names or row["group"] == "block GMRES and scaling":
            label = "block GMRES and scaling"
        elif "stationary_stencil" in names or row["group"] == "read census":
            label = "tracking and qualification"
        elif branch == "carried" or (
            branch is None and row["group"].startswith("carried read:")
        ):
            label = "carried geometry"
        elif branch == "full" or row["group"].startswith("full read:"):
            label = "full geometry"
        else:
            label = "other and fused"
        # The carried-only capture asserts no rebuild before recording; a
        # full-source alias there is shared continuous support, not a full read.
        if program == "carried" and label == "full geometry":
            label = "carried geometry"
        totals[label] += row["device_ms"]
        if label == "carried geometry":
            if "carried_decision_changed" in names or "incidence guard" in row["group"]:
                mechanism = "incidence guard"
            elif names & {
                "_fragments_at_level",
                "quadratic_cell_fragments",
                "_positive_vertical_intervals",
                "_vertical_polygon_interval",
            }:
                mechanism = "fragment areas and edge intervals"
            elif names & {"_curvature_derivatives", "saddle_normal_form"}:
                mechanism = "curvature and normal form"
            elif names & {"nodal_jets", "field_from_jets", "sampled_coefficient"}:
                mechanism = "field coefficients"
            else:
                mechanism = "refresh qualification and shared operations"
            mechanisms[mechanism] += row["device_ms"]
        row["audited_group"] = label
        row["context_provenance"] = [
            dict(op_name=i["op_name"], stack=i["stack"]) for i in contexts
        ]
        row["executed_branch"] = branch
    assert sum(row["launches"] for row in ledger) == profile["kernel_launches"]
    kernel_ms = sum(row["device_ms"] for row in ledger)
    assert abs(sum(totals.values()) - kernel_ms) < 1e-7
    (directory / program / "audited-kernels.json").write_text(
        json.dumps(ledger, indent=2)
    )
    return dict(totals), dict(mechanisms)


def stages(record, directory):
    profile = record["programs"]["solve"]["profile"]
    classified, _ = attributed(record, directory, "solve")
    totals = defaultdict(float, classified)
    prediction = record["programs"]["predictor"]["profile"]
    totals["host scheduling gaps"] = sum(
        p["gap_ms"] - p["overlap_ms"] for p in (profile, prediction)
    )
    totals["transfers and fills"] = sum(
        v["device_ms"] for p in (profile, prediction) for v in p["copies"].values()
    )
    totals["predictor"] = sum(v["device_ms"] for v in prediction["groups"].values())
    expected = profile["profiled_ms"] + prediction["profiled_ms"]
    assert abs(sum(totals.values()) / expected - 1) <= 0.10
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    assert summary["complete"] and not summary["smoke"]
    current = defaultdict(list)
    historical = defaultdict(list)
    for arm in summary["arms"]:
        if arm["process"] == 0 or arm["exit_status"] == 77:
            continue
        record = json.loads(Path(arm["receipt"]).read_text())
        record["receipt_directory"] = str(Path(arm["receipt"]).parent)
        measured_source(record)
        assert record["complete"] and arm["exit_status"] == 0
        assert record["profiler_control"]["h2d_bytes"] == 2097152
        assert record["profiler_control"]["kernels"] > 0
        if arm["read_only"]:
            historical[arm["label"]].append(record)
        else:
            current[arm["width"]].append(record)
            assert max(record["fresh_residual"]) <= 1e-10
            for name, limit in (("axis", 5e-4), ("saddle", 5e-4), ("boundary", 1e-4)):
                assert len(record["precision"][name]) == arm["width"]
                assert max(record["precision"][name]) <= limit
    audited = []
    for width, records in sorted(current.items()):
        assert len(records) == 2
        walls = [r["end_to_end_ms"] for r in records]
        assert abs(walls[0] - walls[1]) / min(walls) <= 0.05
        profiles = [stages(r, Path(r["receipt_directory"])) for r in records]
        names = sorted(set().union(*profiles))
        audited.append(
            dict(
                width=width,
                batch_ms=mean(walls),
                per_slice_ms=mean(walls) / width,
                stages_ms={
                    name: mean(p.get(name, 0) for p in profiles) for name in names
                },
                kernels=[
                    r["programs"]["solve"]["profile"]["kernel_launches"]
                    for r in records
                ],
                residual=max(max(r["fresh_residual"]) for r in records),
            )
        )
    assert set(historical) == {"historical", "main"}
    paired = {}
    for label, records in historical.items():
        assert len(records) == 2
        values = [r["programs"]["carried"]["median_ms"] for r in records]
        assert abs(values[0] - values[1]) / min(values) <= 0.05
        attributed_rows = [
            attributed(r, Path(r["receipt_directory"]), "carried") for r in records
        ]
        paired[label] = dict(
            wall_ms=mean(values),
            groups={
                name: mean(groups.get(name, 0) for groups, _ in attributed_rows)
                for name in set().union(*(g for g, _ in attributed_rows))
            },
            mechanisms={
                name: mean(groups.get(name, 0) for _, groups in attributed_rows)
                for name in set().union(*(g for _, g in attributed_rows))
            },
        )
    (args.run / "audited.json").write_text(
        json.dumps(
            dict(
                widths=audited,
                carried=paired,
                concurrency_units="independent pulses or ensemble members",
                single_pulse_latency_ms=audited[0]["batch_ms"],
            ),
            indent=2,
        )
    )
    headers = [
        "Concurrent pulses / members",
        "Batch wall [ms]",
        "Throughput cost [ms/slice]",
        "Kernels",
        "Fresh residual",
    ]
    names = sorted(set().union(*(r["stages_ms"] for r in audited)))
    headers += [name + " [device ms/batch]" for name in names]
    rows = [
        [
            r["width"],
            f"{r['batch_ms']:.4f}",
            f"{r['per_slice_ms']:.4f}",
            str(r["kernels"]),
            f"{r['residual']:.4g}",
            *(f"{r['stages_ms'].get(name, 0):.4f}" for name in names),
        ]
        for r in audited
    ]

    def cells(tag, values):
        return (
            "<tr>"
            + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values)
            + "</tr>"
        )

    (args.run / "tables.html").write_text(
        "<table><thead>"
        + cells("th", headers)
        + "</thead><tbody>"
        + "".join(cells("td", r) for r in rows)
        + "</tbody></table>"
    )
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    rows_count = math.ceil((len(names) + 1) / 3)
    fig, axes = plt.subplots(
        rows_count, 3, figsize=(14, max(7.2, rows_count * 2.6)), dpi=100
    )
    widths = [r["width"] for r in audited]
    for ax in list(axes.flat)[len(names) + 1 :]:
        ax.set_axis_off()
    for ax, name in zip(
        list(axes.flat)[: len(names) + 1], ["whole slice", *names], strict=True
    ):
        values = [
            r["per_slice_ms"]
            if name == "whole slice"
            else r["stages_ms"][name] / r["width"]
            for r in audited
        ]
        native = name == "whole slice"
        ax.plot(
            widths,
            values,
            "s--" if native else "o-",
            color="#1f5d92",
            linewidth=3.0,
            zorder=3,
        )
        assert min(values) > 0
        lower, upper = min(values) / 1.7, max(values) * 1.7
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        ticks = [min(values), (min(values) * max(values)) ** 0.5, max(values)]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        margin = (math.log10(min(values)) - math.log10(lower)) / (
            math.log10(upper) - math.log10(lower)
        )
        assert margin >= 0.07, "smallest marker is too close to the axis"
        print(
            "PANEL_VISIBLE",
            name,
            "minimum_ms_per_slice",
            min(values),
            "axis_fraction",
            margin,
            flush=True,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [str(w) for w in widths], fontsize=11)
        ax.tick_params(labelsize=11)
        short = {
            "whole slice": "native throughput cost",
            "block GMRES and scaling": "GMRES / scaling",
            "tracking and qualification": "tracking",
            "host scheduling gaps": "host gaps",
            "transfers and fills": "transfer / fills",
            "other and fused": "other / fused",
        }.get(name, name)
        ax.set_ylabel(short + "\n[ms/slice]", fontsize=13)
        ax.set_xlabel("concurrent pulses / members", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.08, top=0.91, hspace=0.70, wspace=0.60
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#1f5d92",
                marker="s",
                linestyle="--",
                linewidth=3,
                label="native end-to-end throughput cost",
            ),
            Line2D(
                [],
                [],
                color="#1f5d92",
                marker="o",
                linestyle="-",
                linewidth=3,
                label="profiled exclusive stage cost",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    save_checked(fig, args.figure)
    import xml.etree.ElementTree as ET

    image = ET.parse(args.figure)
    image.getroot().set("height", f"{31.1 * fig.get_figheight() / 14:.3f}cm")
    image.write(args.figure, encoding="unicode", xml_declaration=True)
    fig.savefig(args.run / "per-slice-stages.png", dpi=100)
    plt.close(fig)
    print("SUMMARY", json.dumps(dict(widths=audited, carried=paired)), flush=True)


if __name__ == "__main__":
    main()
