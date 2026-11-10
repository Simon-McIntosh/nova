"""Render paired receipts with their per-lane census and precision checks."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_cycle.report import pair_gaps
from scripts.prototypes.bank_solve_profile.plot import save_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    parent_log = (args.run / "timing-job.log").read_text()
    assert parent_log.rstrip().endswith("EXIT=0"), "device job did not finish cleanly"

    def receipt(label):
        log = (args.run / f"{label}.log").read_text()
        assert log.rstrip().endswith("EXIT=0"), f"incomplete process {label}"
        assert "XLA_FLAGS=unset" in log and "P100" in log, label
        result = json.loads((args.run / f"{label}.json").read_text())
        assert result["wall_device"] == "titan P100, width 8", label
        assert all("P100" in kind for kind in result["device_kinds"]), label
        expected = {"cold"} | {
            f"warm_{step}_{index}" for step in (1e-5, 1e-3) for index in range(1, 8)
        }
        assert len(result["rows"]) == 15, label
        assert {row["key"] for row in result["rows"]} == expected, label
        return result

    builder = (args.run / "builder.log").read_text()
    assert builder.rstrip().endswith("EXIT=0") and "COMMON_INPUTS" in builder
    pairs = {
        process: {arm: receipt(f"{arm}-{process}") for arm in ("base", "head")}
        for process in (0, 1, 2)
    }
    hashes = {
        receipt["inputs_sha256"] for pair in pairs.values() for receipt in pair.values()
    }
    assert hashes == {
        hashlib.sha256((args.run / "common-inputs.npz").read_bytes()).hexdigest()
    }, "paired arms did not load the same seed archive"
    precision = []
    problems = []
    table = []
    for process, pair in pairs.items():
        for before, after in zip(
            pair["base"]["rows"], pair["head"]["rows"], strict=True
        ):
            assert (before["lanes"], before["key"]) == (after["lanes"], after["key"])
            assert before["lanes"] == 8
            gaps, limits = pair_gaps(before, after)
            maximum = {key: float(np.max(value)) for key, value in gaps.items()}
            precision.append(
                dict(
                    process=process,
                    lanes=before["lanes"],
                    key=before["key"],
                    gaps=maximum,
                    lane_gaps=gaps,
                    limits=limits,
                )
            )
            for name, value in maximum.items():
                if not np.isfinite(value) or value > limits[name]:
                    problems.append(
                        f"p{process} {before['lanes']} {before['key']} "
                        f"{name}: {value} > {limits[name]}"
                    )
            for arm, row in (("base", before), ("head", after)):
                residual = np.asarray(row["residual"])
                if (
                    not all(row["converged"])
                    or residual.size != row["lanes"]
                    or not np.all(np.isfinite(residual))
                    or np.max(residual) > 1e-10
                ):
                    problems.append(f"p{process} {arm} {row['key']}: fresh residual")
                if not row.get("census_matches"):
                    problems.append(
                        f"p{process} {arm} {row['key']}: diagnostic census differs"
                    )
                per_read = float(np.median(row["walls_ms"])) / max(row["reads"])
                allocator = row["peak_memory"] or {}
                peak = allocator.get("peak_bytes_in_use", allocator.get("bytes_in_use"))
                peak_gb = None if peak is None else peak / 1e9
                table.append(
                    f"<tr><td>{process}</td><td>{arm}</td><td>{row['lanes']}</td>"
                    f"<td>{escape(row['key'])}</td><td>{row.get('full_reads')}</td>"
                    f"<td>{row.get('carried_reads')}</td><td>{row['wall_per_slice_ms']:.5f}</td>"
                    f"<td>{per_read:.5f}</td><td>{peak_gb}</td>"
                    f"<td><details><summary>Lane order</summary>reads {row['reads']}; "
                    f"updates {row['updates']}; "
                    f"diagnostic reads {row.get('diagnostic_reads')}; "
                    f"diagnostic updates {row.get('diagnostic_updates')}"
                    f"</details></td></tr>"
                )
    (args.run / "precision.json").write_text(
        json.dumps(dict(rows=precision, problems=problems), indent=2)
    )
    plt.style.use("data-ink")
    widths = (8,)
    fig, axes = plt.subplots(1, len(widths), figsize=(14, 7.2), dpi=100, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.82, wspace=0.25)
    color = InkStyle().flux_color
    for ax, lanes in zip(axes.flat, widths, strict=True):
        for process, pair in pairs.items():
            if process == 0:
                continue
            for arm, ink, marker in (("base", "#929292", "s"), ("head", color, "o")):
                rows = [row for row in pair[arm]["rows"] if row["lanes"] == lanes]
                for index, step in enumerate((0.0, 1e-5, 1e-3)):
                    values = [
                        row["wall_per_slice_ms"] for row in rows if row["step"] == step
                    ]
                    assert len(values) == (1 if step == 0 else 7)
                    position = (
                        index
                        + (-0.12 if arm == "base" else 0.12)
                        + (-0.035 if process == 1 else 0.035)
                    )
                    ax.plot(
                        [position, position],
                        [min(values), max(values)],
                        color=ink,
                        lw=2.6,
                        zorder=2,
                    )
                    ax.plot(
                        position,
                        np.mean(values),
                        marker=marker,
                        color=ink,
                        ms=8,
                        zorder=3 if arm == "head" else 2,
                    )
        ax.set_xticks([0, 1, 2], ["cold", "warm\n1e-5 step", "warm\n1e-3 step"])
        ax.set_ylabel("titan P100, width 8 wall per slice [ms]")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(bottom=0)
        ax.text(0, 1.05, f"{lanes} slices", transform=ax.transAxes, fontsize=16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.legend(
        handles=[
            Line2D([], [], color="#929292", marker="s", lw=2.6, label="base"),
            Line2D([], [], color=color, marker="o", lw=3, label="carried decision"),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    save_checked(fig, args.figure)
    fig.savefig(args.run / "wall-per-slice.png", dpi=100)
    plt.close(fig)
    cost_rows = []
    for arm in ("base", "head"):
        for width, costs in pairs[1][arm].get("read_device_cost", {}).items():
            for kind, cost in costs.items():
                cost_rows.append(
                    f"<tr><td>{arm}</td><td>{width}</td><td>{kind}</td><td>{cost['device_ms']:.5f}</td><td>{cost['rebuilt']}</td></tr>"
                )
    pmax = {
        key: max(row["gaps"][key] for row in precision) for key in precision[0]["gaps"]
    }
    precision_table = "".join(
        f"<tr><td>{escape(key)}</td><td>{value:.8e}</td></tr>"
        for key, value in pmax.items()
    )
    verdict = (
        "All measured precision and census checks pass."
        if not problems
        else "Measured checks remain unmet: " + "; ".join(problems)
    )
    comparisons = []
    for step in (0.0, 1e-5, 1e-3):
        costs = {}
        for arm in ("base", "head"):
            values = [
                float(np.median(row["walls_ms"])) / max(row["reads"])
                for process in (1, 2)
                for row in pairs[process][arm]["rows"]
                if row["step"] == step
            ]
            costs[arm] = float(np.mean(values))
        comparisons.append(
            dict(step=step, **costs, ratio=costs["head"] / costs["base"])
        )
    comparison_table = "".join(
        f"<tr><td>{row['step']}</td><td>{row['base']:.5f}</td>"
        f"<td>{row['head']:.5f}</td><td>{row['ratio']:.5f}</td></tr>"
        for row in comparisons
    )
    width_note = (
        "Only the first eight lanes were run. The cold perturbations are the first "
        "eight entries of the unchanged sixteen-lane schedule. Base builds all "
        "cold and warm predecessor arrays once at width eight; both arms load "
        "that same archive. Widths sixteen and 128 were not attempted on titan."
    )
    residual_max = max(
        max(row["residual"])
        for pair in pairs.values()
        for arm in pair.values()
        for row in arm["rows"]
    )
    section = (
        '<section id="bank-carried-decision-device">'
        "<h2>Titan P100, width 8 paired measurement</h2>"
        "<p>" + escape(verdict) + " Every wall below is a titan P100, width 8 wall. "
        "The H200 width-16 headline remains pending a free card. The pending H200 "
        "allocation was cancelled by id under the revised device decision; "
        "the titan allocation uses one P100, four CPUs and 96 GB host memory.</p>"
        "<p>" + escape(width_note) + "</p>"
        '<figure id="bank-carried-decision-wall"><img '
        'src="/nova/figures/sub-millisecond-forward-path/'
        'bank-carried-decision/wall-per-slice.svg" '
        'alt="Titan P100, width 8 wall per slice, base and carried decision, '
        'cold and warm">'
        "<figcaption>Titan P100, width 8 wall per slice. Each mark is a fresh "
        "measured process after process zero warmed its revision cache. Cold is "
        "the median of three calls; warm is the mean of seven input medians, "
        "with whiskers spanning the seven inputs. Both measured processes are "
        "shown. Raw walls describe the workload; revision comparisons require "
        "equal per-lane read counts or the wall per trial read table. "
        "H200 width-16 headline pending a free card.</figcaption></figure>"
        "<h3>Read-normalised comparison</h3><p>Titan P100, width 8 wall per "
        "batch trial read, averaged over the two measured processes and inputs "
        "in each stream. A batch trial read uses the maximum per-lane read count; "
        "the detailed table retains every lane’s counts.</p>"
        "<table><thead><tr><th>Drive step</th>"
        "<th>Base titan P100, width 8 wall [ms/read]</th>"
        "<th>Head titan P100, width 8 wall [ms/read]</th>"
        "<th>Head/base</th></tr></thead><tbody>" + comparison_table + "</tbody></table>"
        "<h3>Provenance and precision</h3><p>All processes, including warmup "
        "process zero, load the same seed and predecessor arrays built once at base. "
        "Input SHA-256: <code>" + next(iter(hashes)) + "</code>. "
        "Each revision uses its own persistent compilation cache with XLA_FLAGS "
        "unset. The report reads each complete process log and checks its terminal "
        "EXIT=0, device identity and shared-input hash. The independent full-read "
        "residual maximum is " + f"{residual_max:.8e}" + " against 1e-10. "
        "Every diagnostic census must match the timed executable’s per-lane "
        "reads and updates. Process zero is included in the precision checks "
        "and tables, and excluded from the timing figure.</p>"
        "<p>pair_gaps checks every lane: head/base flux 1e-8 of span, axis and "
        "X-point 5e-4 of minor radius, boundary flux 1e-4 of span. Each arm is "
        "also checked against the independent analytic stationary-point oracle.</p>"
        "<table><thead><tr><th>Ratio</th><th>Maximum</th></tr></thead><tbody>"
        + precision_table
        + "</tbody></table>"
        "<h3>Fresh processes and lane census</h3>"
        "<p>All timing columns are titan P100, width 8 walls. Full and carried counts "
        "include the extra fresh certificate after tentative carried convergence. "
        "Peak device GB is the process allocator high-water mark through the row, "
        "including setup and diagnostics; it is not incremental solve memory.</p>"
        "<table><thead><tr><th>Process</th><th>Arm</th><th>Lanes</th><th>Input</th>"
        "<th>Full reads</th><th>Carried reads</th>"
        "<th>Titan P100, width 8 wall [ms/slice]</th>"
        "<th>Titan P100, width 8 wall [ms/batch trial read]</th>"
        "<th>Peak device [GB]</th>"
        "<th>Per-lane counters</th></tr></thead><tbody>"
        + "".join(table)
        + "</tbody></table>"
        "<h3>Standalone read device cost</h3><p>Process one profiles one warmed "
        "full and carried read at the cold input. Titan P100, width 8 device ms "
        "sum observed GPU stream activity including copies; this is not an inferred "
        "decomposition of solve wall.</p><table><thead><tr><th>Arm</th><th>Lanes</th>"
        "<th>Read</th><th>Titan P100, width 8 device [ms]</th>"
        "<th>Rebuilt by lane</th></tr>"
        "</thead><tbody>" + "".join(cost_rows) + "</tbody></table>"
        "<p>Complete logs, receipts, common inputs, profiler traces, precision.json: "
        "<code>" + escape(str(args.run)) + "</code>. The checked SVG uses the "
        "repository data-ink style; raster inspection is recorded in the run log.</p>"
        "</section>"
    )
    fragment = args.fragment.read_text()
    fragment = re.sub(
        r'<section id="bank-carried-decision-device">.*?</section>',
        "",
        fragment,
        flags=re.S,
    )
    fragment = fragment.replace("</main>", section + "\n</main>")
    args.fragment.write_text(fragment)
    print(
        "CHECKS",
        json.dumps(
            dict(
                wall_device="titan P100, width 8",
                problems=problems,
                precision_maxima=pmax,
                fresh_residual_max=residual_max,
                widths=widths,
                inputs_sha256=next(iter(hashes)),
                paired_rows=len(precision),
                read_normalised_wall=comparisons,
            )
        ),
        flush=True,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
