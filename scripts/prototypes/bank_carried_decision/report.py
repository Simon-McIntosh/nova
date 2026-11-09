"""Render paired receipts with their per-lane census and precision checks."""

import argparse
from html import escape
import json
from pathlib import Path

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
    pairs = {
        process: {
            arm: json.loads((args.run / f"{arm}-{process}.json").read_text())
            for arm in ("base", "head")
        }
        for process in (1, 2)
    }
    hashes = {
        receipt["inputs_sha256"] for pair in pairs.values() for receipt in pair.values()
    }
    assert len(hashes) == 1, "paired arms did not load the same seed archive"
    precision = []
    problems = []
    table = []
    for process, pair in pairs.items():
        for before, after in zip(
            pair["base"]["rows"], pair["head"]["rows"], strict=True
        ):
            assert (before["lanes"], before["key"]) == (after["lanes"], after["key"])
            gaps, limits = pair_gaps(before, after)
            maximum = {key: float(np.max(value)) for key, value in gaps.items()}
            precision.append(
                dict(
                    process=process,
                    lanes=before["lanes"],
                    key=before["key"],
                    gaps=maximum,
                )
            )
            for name, value in maximum.items():
                if value > limits[name]:
                    problems.append(
                        f"p{process} {before['lanes']} {before['key']} "
                        f"{name}: {value} > {limits[name]}"
                    )
            for arm, row in (("base", before), ("head", after)):
                if not all(row["converged"]) or max(row["residual"]) > 1e-10:
                    problems.append(f"p{process} {arm} {row['key']}: fresh residual")
                if not row.get("census_matches"):
                    problems.append(
                        f"p{process} {arm} {row['key']}: diagnostic census differs"
                    )
                per_read = float(np.median(row["walls_ms"])) / max(row["reads"])
                memory = row["peak_memory"] or {}
                peak = memory.get("peak_bytes_in_use", memory.get("bytes_in_use"))
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.82, wspace=0.25)
    color = InkStyle().flux_color
    for ax, lanes in zip(axes, (16, 128), strict=True):
        for process, pair in pairs.items():
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
        ax.set_ylabel("wall per slice [ms]")
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
    args.fragment.write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence">
<meta name="plan-evidence-for" content="sub-millisecond-forward-path">
<meta name="plan-verifies" content="sub-millisecond-forward-path#s2">
<title>Carried support decision | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css"><link rel="stylesheet"
href="/_shared/dashboard.css">
</head><body><main class="plan-doc"><section id="bank-carried-decision-result">
<h1>Carried support decision with fresh geometry</h1>
<p>Fragment ownership, component and wall labels, admitted saddle slots and
represented normal-form cells are carried while the stationary cells and saddle
liveness remain unchanged. Nodal jets and own-cell coefficients, stationary positions
and fluxes, saddle curvature, fragment areas, edge intervals and fractional membership
are recomputed. A tentative converged residual triggers an unchanged full read before
acceptance; a capped search also returns a full read.</p>
<p>"""
        + escape(verdict)
        + """</p>
<p>The moved-state test agrees with the full read within 1e-12 relative per continuous
quantity. At base it fails on axis with relative error 2.9542256489575536e-7. Carrying
the preceding edge intervals in scratch fails the same test with edge_interval error
7.391647964927195e-6. The certificate tests force a full read and refuse an invalid
fresh certificate.</p>
<figure id="bank-carried-decision-wall"><img
src="/nova/figures/sub-millisecond-forward-path/bank-carried-decision/wall-per-slice.svg"
alt="Wall per slice at sixteen and 128 lanes, base beside carried decisions, cold and
two warm streams"><figcaption>One reserved H200 allocation. Each marker is one fresh
process: the cold input's median of three calls, or the mean over seven warm input
medians. Whiskers span those seven inputs; two processes per revision follow one
warmup. Both revisions load the same cold and preceding warm-state arrays built at
base. The 128-lane rows repeat the sixteen physical seed lanes eight times. Panels use
independent ordinate ranges. These raw solve walls carry the stated read counts;
revision comparisons use equal-read rows or wall per read from the
table.</figcaption></figure>
<h2>Fresh processes and lane telemetry</h2>
<p>The read census comes from an untimed callback executable in the same process. It
is admitted beside a timing only when its per-lane trial and update counts match the
timed executable; every disagreement is listed above. Full plus carried counts include
the extra full read following a tentative carried certificate. Peak bytes are the
process's device allocator high-water mark through that row, including setup, earlier
solves and diagnostics.</p>
<table><thead><tr><th>Process</th><th>Arm</th><th>Lanes</th><th>Input</th><th>Full
reads</th><th>Carried reads</th><th>ms/slice</th><th>ms/batch trial read</th><th>Peak
device bytes</th><th>Per-lane counters</th></tr></thead><tbody>"""
        + "".join(table)
        + """</tbody></table>
<h2>Standalone read device cost</h2><p>Process one profiles one warmed full read and
one carried read at the cold input, per width. Device milliseconds sum observed GPU
stream activity and include copies; these are standalone read costs, not an inferred
decomposition of solve wall.</p>
<table><thead><tr><th>Arm</th><th>Lanes</th><th>Read</th><th>Device ms</th><th>Rebuilt
by lane</th></tr></thead><tbody>"""
        + "".join(cost_rows)
        + """</tbody></table>
<h2>Precision maxima</h2><p>pair_gaps supplies the head/base flux budget 1e-8 of span,
position budgets 5e-4 of minor radius and boundary-flux budget 1e-4 of span. Both arms
are checked against the independent analytic stationary-point oracle. Each receipt
also contains a separately recomputed full-read residual, with bound
1e-10.</p><table><thead><tr><th>Ratio</th><th>Maximum</th></tr></thead><tbody>"""
        + precision_table
        + """</tbody></table>
<p>Receipts, common input arrays, complete logs, profiler traces and precision.json
are retained at <code>"""
        + escape(str(args.run))
        + """</code>. The runner, controls and scheduler recipes are under
<code>scripts/prototypes/bank_carried_decision</code>. The figure uses the
repository's data-ink style and checked SVG saver. Its rendered PNG requires
inspection before delivery.</p>
</section></main></body></html>"""
    )
    print(
        "CHECKS", json.dumps(dict(problems=problems, precision_maxima=pmax)), flush=True
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
