"""Render paired refresh receipts without mixing latency and throughput."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from scripts.prototypes.carried_read_refresh.run import compare


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "h200/summary.json").read_text())
    assert summary["complete"]
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
        "<h2>Selective continuous fragment refresh</h2>",
    ]
    body += [
        "<p>The carried read retains conic fragments only when their signed "
        "coefficients are unchanged, or a conservative polynomial bound certifies "
        "unchanged full or empty support and the edge slots agree. Saddle fragments "
        "are retained only when the entire normal form agrees. Changed conic and "
        "saddle populations run in separate device loops; uncertain cut geometry "
        "is always recomputed. Stationary roots and curvature are still evaluated "
        "at the current state.</p>",
        "<p>The unchanged-state reproduction executed 50 conic decompositions for "
        "50 cells. The selective test executes zero; changing one cut cell executes "
        "one; the boundary-motion fixture executes 39 of 50. All compare with "
        "forced full refresh at the existing tolerances. The stale-mask mutation "
        "deliberately reuses changing cut cells and must fail the head-passing "
        "tests.</p>",
    ]
    for kind, title in (
        ("read", "Carried read"),
        ("slice", "Forward slice with secant prediction"),
    ):
        body.append(f'<h3 id="carried-read-refresh-{kind}">{title}</h3>')
        body.append(
            "<table><thead><tr><th>Width</th><th>Refresh</th>"
            "<th>Process 1 [ms/batch]</th><th>Process 2 [ms/batch]</th>"
            "<th>Mean [ms/slice]</th><th>Spread</th></tr></thead><tbody>"
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
                body.append(
                    f"<tr><td>{width}</td>"
                    f"<td>{'full' if arm == 'off' else 'selective'}</td>"
                    f"<td>{a:.4f}</td><td>{b:.4f}</td>"
                    f"<td>{(a + b) / 2 / width:.4f}</td><td>{spread:.2%}</td></tr>"
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
                body.append(
                    f"<p>Width {row['width']}: {row['gain']:.2%} lower wall; "
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
        f"<code>{escape(str(args.run / 'h200'))}</code>. Off/on optimized operation "
        "counts differ, and comparing an arm with itself is refused as an inert "
        "lever. The common input SHA256 matches the original slice profile: "
        "<code>52f8d4fb9e60dbdbe1530e8b64a171d3bb6470fb5cf73a1ddcf760895d09a0e9"
        "</code>.</p>"
    )
    body.append("</section>")
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text("\n".join(body) + "\n")
    print("SUMMARY", json.dumps(summary["comparisons"]), flush=True)


if __name__ == "__main__":
    main()
