"""Publish selected fresh-process walls without merging disagreeing pairs."""

import argparse
import html
import json
from pathlib import Path
import statistics


def load(run):
    paths = [run / "h200-levers-corrected/summary.json"]
    inventory = json.loads((run / "measurement-jobs.json").read_text())
    paths += [Path(job["output"]) / "summary.json" for job in inventory["jobs"]]
    paths += sorted(run.glob("measurement-repair-*/summary.json"))
    comparisons = {}
    headline = None
    for path in paths:
        value = json.loads(path.read_text())
        if "repair" in path.parent.name and not value["completed"]:
            continue
        for row in value["comparisons"]:
            key = (row["name"], row["width"], row.get("step"), row.get("population"))
            comparisons[key] = row
        if "headline" in value:
            headline = value["headline"]
    return list(comparisons.values()), headline, inventory


def arms(comparison):
    return comparison.get("tiles") or [comparison["off"], comparison["on"]]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run, root = args.run.resolve(), args.root.resolve()
    comparisons, headline, inventory = load(run)
    table = []
    compile_table = []
    mismatches = set()
    max_spread = 0.0
    residual = 0.0
    for comparison in comparisons:
        for arm in arms(comparison):
            records = arm["records"]
            assert len(records) == 2
            first, second = records
            reference = arms(comparison)[0]["records"]
            if any(
                a["input_sha256"] != b["input_sha256"]
                for a, b in zip(reference, records, strict=True)
            ):
                mismatches.add(comparison["name"])
            rows = {row["name"]: row for row in second["rows"]}
            for row in first["rows"]:
                other = rows[row["name"]]
                identity = [
                    comparison["name"],
                    str(comparison["width"]),
                    str(comparison.get("step", "")),
                    comparison.get("population", ""),
                    str(arm["arm"]),
                    row["name"],
                ]
                if "compile_seconds" in row:
                    compile_table.append(
                        identity
                        + [
                            f"{row['compile_seconds']:.6f}",
                            f"{other['compile_seconds']:.6f}",
                        ]
                    )
                if "median_ms" not in row:
                    continue
                spread = arm["spreads"][row["name"]]
                assert spread <= 0.05
                max_spread = max(max_spread, spread)
                residual = max(
                    residual, *row.get("residual", [0]), *other.get("residual", [0])
                )
                table.append(
                    identity
                    + [
                        f"{row['median_ms']:.6f}",
                        f"{other['median_ms']:.6f}",
                        f"{100 * spread:.3f}%",
                    ]
                )
    per_slice = [
        [
            row["per_slice_ms"]
            for row in record["rows"]
            if row["name"].startswith("slice-") and row["name"] != "slice-1"
        ]
        for record in headline["records"]
    ]
    warm = [statistics.median(values) for values in per_slice]
    jobs = ", ".join(job["job_id"] for job in inventory["jobs"])

    def render_table(identifier, headers, values):
        return (
            f'<table id="h200-levers-{identifier}"><thead><tr>'
            + "".join(f"<th>{html.escape(x)}</th>" for x in headers)
            + "</tr></thead><tbody>\n"
            + "\n".join(
                "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
                for row in values
            )
            + "\n</tbody></table>\n"
        )

    text = f"""<section id="h200-levers-measurements">
<h2>H200 walls and unresolved default decisions</h2>
<p>Seven H200 jobs ({jobs}) completed successfully. Selected fresh-process pairs
agree within {100 * max_spread:.3f}%. Every original completed receipt, including both
carried-decision widths and booking-off width 1, remains unchanged. Disagreeing
earlier pairs are excluded rather than averaged. CPU smoke data contain one
measured warm sample per arm; the driver now says so explicitly.</p>
<p>The inventory audit refuses unequal input fingerprints for
{html.escape(", ".join(sorted(mismatches)))}. These timing observations remain
provisional. Their seed moments were rebuilt through revision-specific booking
code. The committed repair exports the historical fixture and requires its main
arm to reproduce the retained input fingerprint exactly. Completed historical
arms are retained; only unresolved main arms need another measurement.</p>
<p>The expanded audit checked 80 paired process geometry records against the
analytic plasma minor radius and the locked 5e-4 position and 1e-4 flux-span
limits. These are paired differences, not a new independent analytic oracle
qualification. Solved residuals are at most {residual:.6g}. All seven jobs use
x64 on NVIDIA H200 NVL. Compile/cache-load wall is separate from execution.</p>
<p>The historical carried branch returns a frozen reading. Its wall is
attribution only. Carried decisions compare the historical full read with the
current refreshed carried read. The negative control admits distinct program
fingerprints and then refuses identical programs and executed-work counts with
<code>ValueError: inert lever: identical lowered programs and execution counts</code>
and <code>EXIT=1</code>. Each measured pair also records its own program or
executed-work distinction; a synthetic refusal does not replace that check.</p>
<table id="h200-levers-verdicts"><thead><tr>
<th>Lever</th>
<th>Measured implication</th>
<th>Default decision</th></tr></thead><tbody>
<tr>
<td>Carried decisions</td>
<td>Width 16: 30.101346 to 9.301983 ms, first processes</td>
<td>Provisional on; input provenance unresolved</td></tr>
<tr>
<td>Booking cost</td>
<td>Width 1 assembly: 7.586610 to 6.017247 ms; width 16 difference inside 3%</td>
<td>Provisional retain on; input provenance unresolved</td></tr>
<tr>
<td>Factor carry</td>
<td>Flattop and ramp slower at both widths</td>
<td>Retain off; explicit reuse_factor opt-in remains</td></tr>
<tr>
<td>Scalar tile</td>
<td>128 beats 32 at both widths; width-16 full 29.135708 to 27.803048 ms and carried
17.774370 to 16.525416 ms</td>
<td>128 recommended beyond 3% margin; default remains 32 pending permission to update
its pinned test</td></tr>
<tr>
<td>Per-lane drive</td>
<td>Width 1: 173.763677 to 226.640278 ms; width 16: 805.932002 to 997.578302 ms</td>
<td>Retain shared scalar route; independent lanes remain an explicit array-input
choice</td></tr>
<tr>
<td>Live candidates</td>
<td>Width-16 carried 10.423839 to 9.310935 ms</td>
<td>Provisional retain on; input provenance unresolved</td></tr>
<tr>
<td>Diverted wall skip</td>
<td>Width-16 all-diverted carried 8.440215 to 7.178316 ms; mixed 8.400809 to 8.397005
ms</td>
<td>Retain guarded skip; mixed difference unresolved inside 3%</td></tr>
</tbody></table>
<p>Each verdict uses the larger of 3% and the larger arm process spread.
Inside that margin the incumbent stays. No product default has changed yet,
and no paired product suite is claimed. Completion still requires resolving
the refused input comparisons and changing the scalar default with its parity
test and the required sirius base/head suite pair.</p>
<figure id="h200-levers-walls-figure">
<img
 src="/nova/figures/sub-millisecond-forward-path/h200-levers/arm-walls.svg"
alt="Per-lever H200 walls for widths one and sixteen, with a secant flattop trace."/>
<figcaption>First selected process in each arm; second-process walls appear in
the tables. Stars identify input-provenance refusals. Read panels show carried
reads, except the decision-free reference uses its full read. Booking shows
assembly; factor carry shows the median across warm flattop slices 2–7 within
one process; wall skip shows the all-diverted batch. The final panel is per-slice
width-16 wall, including prediction. Other panels are whole-batch walls.
Line colours consistently identify batch width. No disagreeing process pair
contributes to a plotted point.</figcaption></figure>
<p>Width-16 flattop with the secant predictor: warm per-slice medians over slices
2–7 are {warm[0]:.6f} and {warm[1]:.6f} ms in the two fresh processes.
The 1 ms target is not met. Slice 1, before a two-slice secant history is
available, is shown separately below.</p>
"""
    headers = ["Lever", "Width", "Drive step", "Population", "Arm", "Stage"]
    text += render_table(
        "execute-walls", headers + ["Process 1 [ms]", "Process 2 [ms]", "Spread"], table
    )
    text += render_table(
        "compile-walls",
        headers + ["Process 1 compile/load [s]", "Process 2 compile/load [s]"],
        compile_table,
    )
    headline_rows = []
    for left, right in zip(
        headline["records"][0]["rows"], headline["records"][1]["rows"], strict=True
    ):
        if "per_slice_ms" in left:
            headline_rows.append(
                [
                    left["name"],
                    f"{left['per_slice_ms']:.6f}",
                    f"{right['per_slice_ms']:.6f}",
                ]
            )
    text += render_table(
        "headline",
        ["Slice", "Process 1 [ms/slice]", "Process 2 [ms/slice]"],
        headline_rows,
    )
    text += f"""<p>Durable inputs, receipts, lowered fingerprints and logs:
<code>{run}</code>. Inventory: <code>measurement-jobs.json</code>;
preserved original hashes: <code>retained-receipts.json</code>;
audit diagnostics: <code>precision-audit.json</code> and
<code>input-provenance-errors.json</code>. The audit exits nonzero
while input provenance is unresolved.</p></section>"""
    fragment = (
        root / "docs/evidence/fragments/sub-millisecond-forward-path/h200-levers.html"
    )
    fragment.write_text(text)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), dpi=100)
    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.12, top=0.89, wspace=0.57, hspace=0.65
    )
    names = [
        ("carried-decisions", "Carried decisions*", "carried"),
        ("booking", "Booking*", "assembly"),
        ("factor-carry", "Factor carry", None),
        ("scalar-tile", "Scalar tile", "carried"),
        ("lane-drive", "Lane drive", "mixed-drive"),
        ("live-candidates", "Live candidates*", "carried"),
        ("wall-skip", "Wall skip", "carried"),
    ]
    for ax, (name, label, stage) in zip(axes.flat, names):
        for width, colour in [(1, "#929292"), (16, "#1f5d92")]:
            c = next(
                c
                for c in comparisons
                if c["name"] == name
                and c["width"] == width
                and c.get("step", 1e-5) == 1e-5
                and c.get("population", "diverted") == "diverted"
            )
            values = []
            labels = []
            for arm in arms(c):
                metrics = arm["metrics"]
                value = (
                    statistics.median(
                        v
                        for k, v in metrics.items()
                        if k.startswith("slice-") and k != "slice-1"
                    )
                    if stage is None
                    else metrics[
                        "full"
                        if name == "carried-decisions" and arm["arm"] == "off"
                        else stage
                    ]
                )
                values.append(value)
                labels.append(str(arm["arm"]))
            ax.plot(
                range(len(values)),
                values,
                "o-",
                color=colour,
                linewidth=2.6,
                label=f"Width {width}",
                zorder=3 if width == 16 else 2,
            )
            ax.set_xticks(range(len(values)), labels)
        ax.set_ylabel(label + "\nBatch wall [ms]", fontsize=11)
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.18, y=0.15)
    ax = axes.flat[-1]
    record = headline["records"][0]
    values = [r["per_slice_ms"] for r in record["rows"] if "per_slice_ms" in r]
    ax.plot(
        range(1, len(values) + 1), values, "o-", color="#1f5d92", linewidth=3, zorder=3
    )
    ax.set(
        xlabel="Stream slice",
        ylabel="Secant flattop\nWall [ms/slice]",
        xticks=[1, 4, 7],
    )
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    output = (
        root / "docs/figures/sub-millisecond-forward-path/h200-levers/arm-walls.svg"
    )
    for axis in axes.flat:
        lower, upper = axis.get_ylim()
        axis.set_yticks([tick for tick in axis.get_yticks() if lower <= tick <= upper])
    save_checked(fig, output)
    fig.savefig(run / "arm-walls.png", dpi=100)
    print(
        f"SUMMARY {len(table)} selected execution rows; "
        f"warm per-slice {warm[0]:.6f}, {warm[1]:.6f} ms; "
        f"unresolved input families {len(mismatches)}"
    )


if __name__ == "__main__":
    main()
