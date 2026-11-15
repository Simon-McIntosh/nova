"""Audit paired slice receipts and show native and device booking costs."""

import argparse
from html import escape
import json
from pathlib import Path
from statistics import mean
import subprocess

from scripts.prototypes.bank_h200_slice_profile.report import stages

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    assert summary["complete"]
    rows = []
    for comparison in summary["comparisons"]:
        width = comparison["width"]
        item = dict(
            width=width, verdict=comparison["verdict"], margin=comparison["margin"]
        )
        for label in ("off", "on"):
            pair = comparison[label]
            assert pair["spread"] <= 0.05
            records = pair["records"]
            assert len(records) == 2
            profiles = []
            for record in records:
                assert (
                    record["complete"] and "H200" in record["device"] and record["x64"]
                )
                assert record["inputs_sha256"] == summary["input_sha256"]
                assert max(record["fresh_residual"]) <= 1e-10
                for name, limit in (
                    ("axis", 5e-4),
                    ("saddle", 5e-4),
                    ("boundary", 1e-4),
                ):
                    assert len(record["precision"][name]) == width
                    assert max(record["precision"][name]) <= limit
                assert record["profiler_control"]["h2d_bytes"] == 2097152
                assert record["profiler_control"]["kernels"] > 0
                current = (
                    args.run
                    / "measured-source"
                    / record["revision"]
                    / "nova/equilibrium/current.py"
                )
                expected = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(ROOT),
                        "show",
                        record["revision"] + ":nova/equilibrium/current.py",
                    ]
                )
                assert current.read_bytes() == expected
                arm = next(
                    a
                    for a in summary["arms"]
                    if a["label"] == label
                    and a["width"] == width
                    and json.loads(Path(a["receipt"]).read_text()) == record
                )
                directory = Path(arm["receipt"]).parent
                profiles.append(stages(record, directory))
            device = [
                sum(
                    p.get(k, 0)
                    for k in ("assembly", "update residual", "read residual")
                )
                for p in profiles
            ]
            assert min(device) > 0, (
                "known-present booking/assembly kernels were not attributed"
            )
            item[label] = dict(
                wall_ms=pair["wall_ms"],
                per_slice_ms=pair["wall_ms"] / width,
                process_ms=[r["end_to_end_ms"] for r in records],
                spread=pair["spread"],
                booking_assembly_device_ms=mean(device),
                device_per_slice_ms=mean(device) / width,
                device_spread=abs(device[0] - device[1]) / min(device),
                compile_seconds=[
                    r["programs"]["solve"]["compile_seconds"] for r in records
                ],
                stages={
                    k: mean(p.get(k, 0) for p in profiles)
                    for k in set().union(*profiles)
                },
                maximum_residual=max(max(r["fresh_residual"]) for r in records),
                maximum_precision={
                    k: max(max(r["precision"][k]) for r in records)
                    for k in ("axis", "saddle", "boundary")
                },
            )
        item["wall_change"] = item["on"]["wall_ms"] / item["off"]["wall_ms"] - 1
        rows.append(item)
    audit = dict(
        widths=rows,
        complete=True,
        input_sha256=summary["input_sha256"],
        base=summary["base"],
        head=summary["head"],
    )
    (args.run / "audit.json").write_text(json.dumps(audit, indent=2))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    widths = [r["width"] for r in rows]
    for ax, metric, ylabel in zip(
        axes,
        ("per_slice_ms", "device_per_slice_ms"),
        ("native throughput cost [ms/slice]", "booking + assembly device [ms/slice]"),
        strict=True,
    ):
        for label, colour, style, linewidth in (
            ("off", "#929292", "s--", 2.6),
            ("on", "#1f5d92", "o-", 3.0),
        ):
            ax.plot(
                widths,
                [r[label][metric] for r in rows],
                style,
                color=colour,
                linewidth=linewidth,
                zorder=3 if label == "on" else 2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [str(w) for w in widths])
        ax.set_xlabel("concurrent independent pulses")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.set_ylim(bottom=0)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#929292",
                linestyle="--",
                marker="s",
                linewidth=2.6,
                label="separate passes",
            ),
            Line2D(
                [], [], color="#1f5d92", marker="o", linewidth=3, label="fused pass"
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.85, wspace=0.30)
    save_checked(fig, args.figure)
    fig.savefig(args.run / "booking-assembly.png", dpi=100)
    plt.close(fig)

    cells = []
    for row in rows:
        cells.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    row["width"],
                    " / ".join(f"{v:.4f}" for v in row["off"]["process_ms"]),
                    " / ".join(f"{v:.4f}" for v in row["on"]["process_ms"]),
                    f"{row['off']['per_slice_ms']:.4f} → "
                    f"{row['on']['per_slice_ms']:.4f}",
                    f"{row['off']['booking_assembly_device_ms']:.4f} → "
                    f"{row['on']['booking_assembly_device_ms']:.4f}",
                    f"{100 * row['wall_change']:+.2f}%",
                    f"{100 * row['margin']:.2f}%",
                    row["verdict"],
                )
            )
            + "</tr>"
        )
    relative = args.figure.relative_to(ROOT / "docs")
    fragment = f"""<section id="fused-booking-assembly-landing">
<h2>Primal booking and local assembly channels share one cell traversal</h2>
<p>The Newton correction evaluates primal moments beside local derivative
channels on the same conic rings and quadrature points. The scalar conic tile
remains 128 cells by default; normal-form cells retain their independent
eight-cell buckets. Global current normalization follows the cell reduction.
Factor-reuse corrections retain their booking-only path.</p>
<table><thead><tr><th>Independent pulses</th>
<th>Separate process walls [ms/batch]</th><th>Fused process walls [ms/batch]</th>
<th>Native cost [ms/slice]</th><th>Booking + assembly device [ms/batch]</th>
<th>Wall change</th><th>Decision margin</th><th>Verdict</th></tr></thead><tbody>
{"".join(cells)}
</tbody></table>
<p>The paired H200 job uses base <code>{summary["base"]}</code>
and fused revision <code>{summary["head"]}</code>.
Each arm has a warming process and two fresh measured processes; every selected
pair agrees within 5%. The decision margin is the larger of 3% and either
process spread. Compiles are excluded from synchronized warm walls and
retained separately in each receipt. The common input SHA256 is
<code>{summary["input_sha256"]}</code>.</p>
<figure id="fused-booking-assembly-figure"><img src="/nova/{relative}"
alt="Separate and fused booking arms across independent pulse widths,
showing native throughput and device booking plus assembly cost.">
<figcaption>Each point is the mean of two fresh-process medians. The left panel
is unprofiled secant-predicted flattop throughput; the right panel sums the
profiled assembly, read-residual booking and update-residual booking kernels.
Profiled device costs are not substituted for native wall savings. Width one
is causal single-pulse latency; widths 16 and 64 advance independent pulses
and do not batch sequential TORAX-coupled slices.</figcaption></figure>
<p>All selected lanes pass the independent fresh residual (at most 1e-10),
axis and X-point (at most 5e-4 of minor radius), and boundary-flux (at most
1e-4 of flux span) budgets. The existing slice profiler supplies the solve,
predictor, precision oracle and raw device capture; its retained source-aware
report audits kernel attribution and closes device stages against profiled
walls. Before each capture, the profiler detects the known-present
2097152-byte transfer and matrix kernel. Metadata-free lowered program digests
differ between arms, and feeding the identical on-program receipt to the
comparison guard is refused.</p>
<p>Receipts, exact measured product sources, compile walls, per-kernel ledgers,
profiler controls, program digests and negative-control logs are retained at
<code>{args.run}</code>. The figure passes the shared text-boundary and
collision checker; the producer inspects its rendered SVG before delivery.
Correctness and missing-moment mutation results are recorded in the worker
manifest.</p>
</section>
"""
    args.fragment.write_text(fragment)
    print(
        "SUMMARY",
        json.dumps(
            [
                dict(
                    width=r["width"],
                    verdict=r["verdict"],
                    separate_ms=r["off"]["per_slice_ms"],
                    fused_ms=r["on"]["per_slice_ms"],
                    change=r["wall_change"],
                    margin=r["margin"],
                )
                for r in rows
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
