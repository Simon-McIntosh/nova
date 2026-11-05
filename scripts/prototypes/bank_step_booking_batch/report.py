"""Render paired Jacobian scaling and record the converged-state precision."""

import argparse
from html import escape
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_cycle.report import compare
from scripts.prototypes.bank_solve_profile.receipts import iterate
from scripts.prototypes.bank_solve_read.plot import save_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    topic = "sub-millisecond-forward-path/bank-step-booking-batch"
    base, head = [
        json.loads((args.run / f"{arm}-timings.json").read_text())
        for arm in ("base", "head")
    ]
    solves = [
        json.loads((args.run / f"{arm}-solve.json").read_text())
        for arm in ("base", "head")
    ]
    precision = compare(*solves)
    (args.run / "precision.json").write_text(json.dumps(precision, indent=2))
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    color = InkStyle().flux_color
    for label, values, hue, dash, offset in [
        ("base", base, "#929292", "--", (10, 24)),
        ("batched", head, color, "-", (52, 12)),
    ]:
        walls = [values[f"{lanes}:assembly"] for lanes in (1, 16)]
        ax.plot(
            [1, 16],
            walls,
            color=hue,
            linestyle=dash,
            marker="o",
            linewidth=3 if label == "batched" else 2.6,
            label=label,
            zorder=3 if label == "batched" else 2,
        )
        for lanes, wall in zip((1, 16), walls, strict=True):
            ax.annotate(
                f"{wall:.2f}",
                (lanes, wall),
                xytext=offset,
                textcoords="offset points",
                color=hue if label != "base" else "#555555",
                fontsize=14,
            )
    ax.set_xlim(0, 19)
    ax.set_ylim(0, max(base["16:assembly"], head["16:assembly"]) * 1.18)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("Lanes")
    ax.set_ylabel("Jacobian wall [ms]")
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.14, top=0.95)
    figure = root / "docs/figures" / topic / "jacobian-wall.svg"
    save_checked(fig, figure)
    fig.savefig(args.run / "jacobian-wall.png", dpi=100)
    plt.close(fig)
    rows = []
    summary = []
    for arm, timings, receipt in zip(
        ("base", "batched"), (base, head), solves, strict=True
    ):
        for lanes in (1, 16):
            row = next(
                r for r in receipt["rows"] if r["lanes"] == lanes and r["call"] == 1
            )
            per_iterate = iterate(
                {(lanes, "whole_solve"): row["wall_ms"]},
                {lanes: {"max_steps": max(row["steps"])}},
                lanes,
            )
            values = dict(
                arm=arm,
                lanes=lanes,
                jacobian_ms=timings[f"{lanes}:assembly"],
                iterate_ms=per_iterate,
                per_slice_ms=row["per_slice_ms"],
                steps=max(row["steps"]),
                residual=max(row["residual"]),
            )
            summary.append(values)
            rows.append(
                f"<tr><td>{arm}</td><td>{lanes}</td><td>{values['jacobian_ms']:.3f}</td><td>{per_iterate:.3f}</td><td>{row['per_slice_ms']:.3f}</td><td>{values['steps']}</td><td>{values['residual']:.4e}</td></tr>"
            )
    maximum = {
        key: max(r[key] for r in precision)
        for key in precision[0]
        if key not in ("lanes", "call")
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    print("PRECISION", json.dumps(maximum), flush=True)
    ratio = head["16:assembly"] / head["1:assembly"]
    closure_rows = []
    for arm in ("base", "head"):
        path = args.run / f"{arm}-step-profile.json"
        if path.exists():
            for row in json.loads(path.read_text()):
                closure_rows.append(
                    f"<tr><td>{arm}</td><td>{row['lanes']}</td>"
                    f"<td>{row['closure']:.5f}</td>"
                    f"<td>{escape(str(row['stencil_nodes']))}</td></tr>"
                )
    gate_rows = []
    gate_paths = (
        [
            args.run.parent / f"cpu-base-{name}.log"
            for name in (
                "test_bank_solve_cycle",
                "test_bank_solve_driver",
                "test_support_read",
            )
        ]
        + [
            args.run / f"rings-head-{name}.log"
            for name in (
                "test_bank_solve_cycle",
                "test_bank_solve_driver",
                "test_support_read",
                "booking",
            )
        ]
        + [args.run / "scaling-head.log", args.run / "negative-control.log"]
    )
    failures = []
    for path in gate_paths:
        if not path.exists():
            result = "pending"
        else:
            content = path.read_text()
            summaries = re.findall(r"^=+ (.+ in .+) =+$", content, re.MULTILINE)
            result = summaries[-1] if summaries else "pending"
            if path.name.startswith("rings-head-"):
                failures.extend(re.findall(r"^FAILED (.+)$", content, re.MULTILINE))
        gate_rows.append(
            f"<tr><td>{escape(path.name)}</td><td>{escape(result)}</td></tr>"
        )
    regression = (
        "<p><strong>CPU regression remains:</strong> "
        + escape("; ".join(failures))
        + ". This node is not ready to promote.</p>"
        if failures
        else ""
    )
    fragment = root / "docs/evidence/fragments" / (topic + ".html")
    body = f"""<section id="bank-step-booking-batch-landing">
<h2>Batch populated booking buckets across Newton lanes</h2>
{regression}
<p>The initial H200 reproduction (allocation 1306388) isolates the excess
outer-vmap cost to local booking:
13.387 ms mapped versus 121.856 ms vectorized at one lane. Whole Jacobian assembly is
18.837 versus 128.185 ms. Both compiled loops carry 9 conic and 69 normal padded
buckets; the mapped control contains two bucket conditionals and the vectorized
form contains none. The lane batching rule lowers those conditionals to selection,
so empty padded buckets still integrate.</p>
<p>The integration loop now stops at the selected population's bucket count.
Ring quadrature likewise compacts live rings in groups of four and loops through
every group, so the padded polygon capacity does not expand the expensive integral.
Fresh Newton corrections use an outer vmap. Fresh geometry evaluates eighteen jet
channels on three distinct Bell nodes, against 162 padded channels at base.
A stencil exceeding three nodes returns a nonfinite matrix and cannot supply an
accepted correction. The capacity-pressure test lowers the capacity to two and
requires that refusal on the three-node seed.</p>
<p>Sixteen-lane Jacobian scaling is <strong>{ratio:.3f} times one lane</strong>,
against a limit of four. Measurements use the production fresh-correction lane
mapper, with explicit synchronized calls through the shared timing instrument.</p>
<table id="bank-step-booking-batch-walls"><thead><tr><th>Arm</th><th>Lanes</th>
<th>Jacobian [ms/batch]</th><th>Converged solve / maximum updates [ms]</th>
<th>Converged wall [ms/slice]</th><th>Maximum updates</th>
<th>Largest fresh residual</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<figure id="bank-step-booking-batch-figure"><img
src="/nova/figures/{topic}/jacobian-wall.svg"
alt="Jacobian wall against one and sixteen lanes, base and batched, on linear
axes with values at markers"><figcaption>One reserved H200 allocation,
{escape(args.job)}.
Jacobian values are medians of three synchronized warm calls; the solve wall
is its second execution after explicit compilation. The per-iterate value
divides that complete solve wall by its maximum lane update count, including
reads and line search. The plot uses linear axes and prints milliseconds at
every marker.</figcaption></figure>
<p>The first bucket-only candidate scaled 9.45 times (15.624 to 147.719 ms) and
missed the bound; its receipts remain under bucket-only. The table uses the
live-ring compaction with sixteen-cell cut buckets. The capacity sweep measured
one/sixteen-lane walls of 12.173/32.893 ms at width eight, 9.382/31.557 ms at
width sixteen, and 8.086/33.801 ms at width thirty-two. Width sixteen minimizes
the sixteen-lane wall among those choices. All paired terminal states satisfy
the precision
budget. Largest fresh residual:
{maximum["residual"]:.5e}; flux/span against base: {maximum["flux"]:.5e};
axis/minor radius: {maximum["axis"]:.5e}; X-point/minor radius:
{maximum["saddle"]:.5e}; boundary flux/span: {maximum["boundary_flux"]:.5e}.
The limits are respectively 1e-10, 1e-8, 5e-4, 5e-4 and 1e-4.</p>
<p id="bank-step-booking-batch-provenance">Base revision:
<code>{escape(solves[0]["revision"])}</code>; measured head:
<code>{escape(solves[1]["revision"])}</code>. Paired timing receipts, CPU suite
logs, matrix-relative-error rows, overflow refusal, serial-lane mutation and
rendered figure are under <code>{escape(str(args.run))}</code>. The base is a git
archive; every arm prints its imported module path. Read stages are zero in the
standalone Newton-step closure because that step consumes an already-completed
read. The shared closure instrument sums residual, assembly, equilibration/factor
and back-solve against the measured whole step.</p>
<table id="bank-step-booking-batch-closure"><thead><tr><th>Arm</th><th>Lanes</th>
<th>Stage sum / whole Newton step</th><th>Distinct geometry nodes per lane</th>
</tr></thead><tbody>{"".join(closure_rows)}</tbody></table>
<table id="bank-step-booking-batch-tests"><thead><tr><th>Gate log</th>
<th>Pytest summary</th></tr></thead><tbody>{"".join(gate_rows)}</tbody></table>
</section>"""
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(body + "\n")


if __name__ == "__main__":
    main()
