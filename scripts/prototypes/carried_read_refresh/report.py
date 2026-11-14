"""Render paired refresh receipts without mixing latency and throughput."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from scripts.prototypes.carried_read_refresh.run import compare


def cpu_evidence(run):
    """Keep completed compatibility and mutation checks beside the timing."""
    audit = json.loads((run / "per-file/audit.json").read_text())
    assert not audit["added_head_failures"]
    assert audit["totals"]["baseline"] == audit["totals"]["after"]
    scoped = json.loads((run / "default-gate/receipts.json").read_text())
    after, negative = scoped["after"], scoped["negative"]
    assert after["completed"] and after["exit_status"] == 0
    assert negative["completed"] and negative["exit_status"] != 0
    assert set(negative["failure_ids"]) - set(after["failure_ids"])
    after_summary = Path(after["log_path"]).read_text().splitlines()[-2]
    negative_summary = Path(negative["log_path"]).read_text().splitlines()[-2]
    body = [
        '<h3 id="carried-read-refresh-correctness">CPU comparison</h3>',
        "<p>Each compatibility file ran in a separate fresh process with "
        "a unique temporary directory. The baseline is "
        f"<code>{audit['baseline_revision']}</code>; the measured head is "
        f"<code>{audit['measured_head']}</code>. All twenty file logs were "
        "audited against their receipts.</p>",
        "<table><thead><tr><th>Arm</th><th>Completed files</th>"
        "<th>Passed</th><th>Failed</th><th>Skipped</th>"
        "<th>Incomplete files</th></tr></thead><tbody>",
    ]
    for arm in ("baseline", "after"):
        totals = audit["totals"][arm]
        body.append(
            f"<tr><td>{arm}</td><td>{totals['completed_files']}</td>"
            f"<td>{totals['passed']}</td><td>{totals['failed']}</td>"
            f"<td>{totals['skipped']}</td>"
            f"<td>{totals['incomplete_files']}</td></tr>"
        )
    body += [
        "</tbody></table>",
        "<p>No added head failures were observed. The three failing IDs match; "
        "both batch files terminate with <code>EXIT=-9</code> in both arms. "
        "The runner completed its loop, but SLURM records "
        "<code>OUT_OF_MEMORY</code> and four OOM kills. These shared failures "
        "and incomplete files are retained under the authorized delta rule; "
        "this is not a fully passing suite.</p>",
        f"<p>The full-refresh default is pinned at <code>{after['revision']}"
        f"</code>. Its scoped suite reports <code>{escape(after_summary)}</code>. "
        "Corrupting the reuse mask reports "
        f"<code>{escape(negative_summary)}</code> and fails head-passing tests. "
        "The base "
        "has no scoped test file and its arm is explicitly unarmed. "
        f"Receipts: <code>{escape(str(run / 'default-gate'))}</code>; "
        f"per-file audit: <code>{escape(str(run / 'per-file/audit.json'))}</code>."
        "</p>",
    ]
    return body


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    measurements = args.measurements or args.run / "h200"
    summary = json.loads((measurements / "summary.json").read_text())
    assert summary["complete"]
    audit = json.loads((measurements / "audit.json").read_text())
    assert audit["complete"] and audit["revision"] == summary["revision"]
    assert audit["receipt_count"] == 36
    records = {}
    for arm in summary["arms"]:
        path = Path(arm["log"]).parent / "receipt.json"
        record = json.loads(path.read_text())
        assert record["complete"] and record["revision"] == summary["revision"]
        assert record["x64"] and "H200" in record["device"]
        assert Path(arm["log"]).read_text().splitlines()[-1] == "EXIT=0"
        inputs = args.run / "inputs/common-streams.npz"
        assert (
            hashlib.sha256(inputs.read_bytes()).hexdigest() == record["inputs_sha256"]
        )
        if arm["kind"] == "slice":
            assert max(record["fresh_residual"]) <= 1e-10
            for key, limit in (("axis", 5e-4), ("saddle", 5e-4), ("boundary", 1e-4)):
                assert len(record["precision"][key]) == arm["width"]
                assert max(record["precision"][key]) <= limit
        records[arm["kind"], arm["width"], arm["arm"], arm["process"]] = record
    assert len(records) == 36
    for kind in ("read", "slice"):
        for width in (1, 16, 64):
            compare(
                records[kind, width, "off", 1],
                records[kind, width, "on", 1],
                "carried" if kind == "read" else "solve",
            )
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    for axis, kind, label in zip(
        axes,
        ("read", "slice"),
        ("Carried read [ms/slice]", "Forward slice [ms/slice]"),
        strict=True,
    ):
        rows = sorted(
            (row for row in summary["comparisons"] if row["kind"] == kind),
            key=lambda row: row["width"],
        )
        for arm, color, style, marker in (
            ("off", "#929292", "--", "s"),
            ("on", "#1f5d92", "-", "o"),
        ):
            values = [row["batch_ms"][arm] / row["width"] for row in rows]
            axis.plot(
                [row["width"] for row in rows],
                values,
                color=color,
                linestyle=style,
                marker=marker,
                linewidth=3 if arm == "on" else 2.6,
                label="selective refresh" if arm == "on" else "full refresh",
                zorder=3 if arm == "on" else 2,
            )
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(FixedLocator([1, 16, 64]))
        axis.set_xticklabels(["1", "16", "64"])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Independent pulses")
        axis.set_ylabel(label)
        axis.set_xlim(0.8, 80)
        axis.margins(y=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.84, wspace=0.25)
    save_checked(fig, args.figure)
    plt.close(fig)

    body = [
        '<section id="carried-read-refresh-result">',
        "<h2>Faster carried reads; full refresh retained by default</h2>",
    ]
    regressions = [
        row
        for row in summary["comparisons"]
        if row["kind"] == "slice" and row["gain"] < -row["margin"]
    ]
    assert regressions, "the default decision requires a resolved slice regression"
    single = next(row for row in regressions if row["width"] == 1)
    body.append(
        "<p>Selective refresh improves the isolated carried read at every "
        "measured width, but increases complete single-pulse forward-slice wall "
        f"by {-single['gain']:.2%}, beyond the {single['margin']:.0%} decision "
        "margin. Widths 16 and 64 have "
        "unresolved complete-slice differences. The support read therefore "
        "selects full refresh by default; selective refresh remains available "
        "through <code>topology.refresh_at_stationary_points(reuse=True)</code>."
        " The table records the measured revision before that default change; "
        "it does not claim a second timing run at the changed default.</p>"
    )
    body += [
        "<p>The carried read retains conic fragments only when their signed "
        "coefficients are unchanged, or a conservative polynomial bound certifies "
        "unchanged full or empty support and the edge slots agree. Saddle fragments "
        "are retained only when the entire normal form agrees. Changed conic and "
        "saddle populations run in separate device loops; uncertain cut geometry "
        "is always recomputed. Stationary roots and curvature are still evaluated "
        "at the current state.</p>",
        "<p>The base reproduction recorded "
        "<code>valid=True rebuilt=False recomputed_cells=50 total_cells=50</code> "
        "for an unchanged carried state. Its log is "
        f"<code>{escape(str(args.run / 'reproduce.log'))}</code>. "
        "The selective test executes zero; changing one cut cell executes "
        "one; the boundary-motion fixture executes 39 of 50. All compare with "
        "forced full refresh at the existing tolerances. The stale-mask mutation "
        "deliberately reuses changing cut cells and must fail the head-passing "
        "tests.</p>",
    ]
    maximum_spread = max(
        value for row in summary["comparisons"] for value in row["spread"].values()
    )
    maxima = audit["maxima"]
    body.append(
        "<p>One paired H200 job completed all 36 fresh processes. The largest "
        f"measured-process spread is {maximum_spread:.3%}, below the 5% bound. "
        f"The worst fresh residual is {maxima['fresh_residual']:.6g} "
        "(budget 1e-10); maximum axis and saddle position errors are "
        f"{maxima['axis']:.6g} and {maxima['saddle']:.6g} of the minor radius "
        "(budget 5e-4), and boundary error is "
        f"{maxima['boundary']:.6g} of the flux span (budget 1e-4). "
        "The profiler's positive control observes a 2 MiB transfer and a "
        "kernel in every process. The independent receipt audit is "
        f"<code>{escape(str(measurements / 'audit.json'))}</code>.</p>"
    )
    for kind, title in (
        ("read", "Carried read"),
        ("slice", "Forward slice with secant prediction"),
    ):
        body.append(f'<h3 id="carried-read-refresh-{kind}">{title}</h3>')
        body.append(
            "<table><thead><tr><th>Width</th><th>Refresh</th>"
            "<th>Process 1 [ms/batch]</th><th>Process 2 [ms/batch]</th>"
            "<th>Mean [ms/slice]</th><th>Spread</th>"
            "<th>Optimized operations</th></tr></thead><tbody>"
        )
        for width in (1, 16, 64):
            for arm in ("off", "on"):

                def wall(process):
                    receipt = records[kind, width, arm, process]
                    return (
                        receipt["programs"]["carried"]["median_ms"]
                        if kind == "read"
                        else receipt["end_to_end_ms"]
                    )

                a, b = wall(1), wall(2)
                spread = abs(a - b) / min(a, b)
                assert spread <= 0.05
                program = "carried" if kind == "read" else "solve"
                operations = records[kind, width, arm, 1]["programs"][program][
                    "optimized_operations"
                ]
                assert (
                    operations
                    == records[kind, width, arm, 2]["programs"][program][
                        "optimized_operations"
                    ]
                )
                body.append(
                    f"<tr><td>{width}</td>"
                    f"<td>{'full' if arm == 'off' else 'selective'}</td>"
                    f"<td>{a:.4f}</td><td>{b:.4f}</td>"
                    f"<td>{(a + b) / 2 / width:.4f}</td><td>{spread:.3%}</td>"
                    f"<td>{operations}</td></tr>"
                )
        body.append("</tbody></table>")
        for row in summary["comparisons"]:
            if row["kind"] == kind:
                verdict = (
                    "resolved improvement"
                    if row["gain"] > row["margin"]
                    else "resolved regression"
                    if row["gain"] < -row["margin"]
                    else "unresolved difference"
                )
                direction = "lower" if row["gain"] >= 0 else "higher"
                body.append(
                    f"<p>Width {row['width']}: {abs(row['gain']):.2%} "
                    f"{direction} wall; "
                    f"margin {row['margin']:.2%}; {verdict}.</p>"
                )
    source = (
        "/nova/figures/sub-millisecond-forward-path/"
        "carried-read-refresh/paired-walls.svg"
    )
    body.append(
        f'<figure id="carried-read-refresh-figure"><img src="{source}" '
        'alt="Full and selective refresh wall per slice across independent pulse '
        'widths 1, 16 and 64."><figcaption>Each point is the mean of two '
        "fresh-process medians of eleven synchronized warm calls, following one "
        "warming process per arm. Width one is single-pulse latency. Wider values "
        "divide concurrent independent-pulse batch wall by width; slices within "
        "one TORAX-coupled pulse remain sequential. Grey dashed squares show full "
        "refresh; blue solid circles show selective refresh.</figcaption></figure>"
    )
    body.append(
        f"<p>Measured revision <code>{escape(summary['revision'])}</code>. "
        "The unchanged slice-profile instrument supplies compile and execute "
        "walls separately, full fresh certificates and the analytic precision "
        "checks. All 36 receipts and logs are retained under "
        f"<code>{escape(str(measurements))}</code>. Off/on optimized operation "
        "counts differ, and comparing an arm with itself is refused as an inert "
        f"lever. Common input <code>{escape(str(inputs))}</code> has SHA256 "
        "<code>52f8d4fb9e60dbdbe1530e8b64a171d3bb6470fb5cf73a1ddcf760895d09a0e9"
        "</code>.</p>"
    )
    body.extend(cpu_evidence(args.run))
    body.append("</section>")
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text("\n".join(body) + "\n")
    print("SUMMARY", json.dumps(summary["comparisons"]), flush=True)


if __name__ == "__main__":
    main()
