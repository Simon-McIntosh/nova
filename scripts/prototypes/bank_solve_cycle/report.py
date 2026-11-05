"""Compare paired device receipts and draw all fresh-residual trajectories."""

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_profile.plot import save_checked


def compare(base, head):
    gaps = []
    for before, after in zip(base["rows"], head["rows"], strict=True):
        assert (before["lanes"], before["call"]) == (after["lanes"], after["call"])
        row = dict(lanes=after["lanes"], call=after["call"])
        for key, unit in [
            ("flux", before["span"]),
            ("axis", before["minor_radius"]),
            ("saddle", before["minor_radius"]),
            ("boundary_flux", before["span"]),
        ]:
            difference = np.asarray(after[key]) - np.asarray(before[key])
            if key in ("axis", "saddle"):
                difference = np.linalg.norm(difference, axis=-1)
            row[key] = float(np.max(np.abs(difference))) / unit
        row["residual"] = max(after["residual"])
        oracle_roots = np.asarray(after["oracle_roots"])
        row["oracle_axis"] = (
            float(
                np.max(
                    np.linalg.norm(np.asarray(after["axis"]) - oracle_roots[0], axis=-1)
                )
            )
            / after["minor_radius"]
        )
        row["oracle_saddle"] = (
            float(
                np.max(
                    np.linalg.norm(
                        np.asarray(after["saddle"]) - oracle_roots[1], axis=-1
                    )
                )
            )
            / after["minor_radius"]
        )
        row["oracle_boundary"] = (
            after["oracle"]["boundary_ratio"]
            + float(
                np.max(
                    np.abs(
                        np.asarray(after["boundary_flux"]) - after["boundary_flux"][0]
                    )
                )
            )
            / after["span"]
        )
        gaps.append(row)
        assert all(after["converged"]) and row["residual"] <= 1e-10
        assert row["flux"] <= 1e-8
        assert (
            max(row["axis"], row["saddle"], row["oracle_axis"], row["oracle_saddle"])
            <= 5e-4
        )
        assert row["boundary_flux"] <= 1e-4 and row["oracle_boundary"] <= 1e-4
    return gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    pairs = [
        (
            json.loads((args.run / f"{label}-base.json").read_text()),
            json.loads((args.run / f"{label}-head.json").read_text()),
        )
        for label in ("pair-one", "pair-two")
    ]
    gaps = [compare(*pair) for pair in pairs]
    (args.run / "precision.json").write_text(json.dumps(gaps, indent=2))
    plt.style.use("data-ink")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), dpi=100, sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.11, top=0.86, hspace=0.30, wspace=0.12
    )
    color = InkStyle().flux_color
    for allocation, pair in enumerate(pairs):
        for arm, receipt in enumerate(pair):
            row = receipt["rows"][-1]
            ax = axes[allocation, arm]
            worst = int(np.argmax(row["steps"]))
            for lane, (steps, values) in enumerate(
                zip(row["steps"], row["residual_path"], strict=True)
            ):
                ax.plot(
                    np.arange(steps + 1),
                    values[: steps + 1],
                    color=color if lane == worst else "#929292",
                    linewidth=3 if lane == worst else 1.2,
                    zorder=3 if lane == worst else 2,
                )
            ax.axhline(1e-10, color="#666666", linestyle=":", linewidth=1.2, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(1e-13, 1)
            arm_name = "base" if arm == 0 else "backtracking"
            ax.set_title(
                f"Allocation {allocation + 1} · {arm_name}",
                loc="left",
                fontsize=13,
            )
            if arm == 0:
                ax.set_ylabel("fresh residual")
            if allocation == 1:
                ax.set_xlabel("Newton iterate")
    fig.legend(
        handles=[
            Line2D([], [], color="#929292", lw=1.2, label="other lanes"),
            Line2D([], [], color=color, lw=3, label="slowest lane in each arm"),
            Line2D([], [], color="#666666", lw=1.2, ls=":", label="acceptance limit"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
    )
    save_checked(fig, args.figure)
    fig.savefig(args.run / "residual-history.png", dpi=100)
    tables = []
    for allocation, pair in enumerate(pairs):
        rows = []
        for arm, receipt in zip(("base", "backtracking"), pair, strict=True):
            for lanes in (1, 16):
                calls = [r for r in receipt["rows"] if r["lanes"] == lanes]
                steps = ", ".join(map(str, calls[1]["steps"]))
                assert calls[0]["steps"] == calls[1]["steps"]
                rows.append(
                    f"<tr><td>{arm}</td><td>{lanes}</td><td>{calls[0]['per_slice_ms']:.2f}</td><td>{calls[1]['per_slice_ms']:.2f}</td><td>{steps}</td><td>{max(calls[1]['residual']):.3e}</td></tr>"
                )
        tables.append(
            f'<h2 id="bank-solve-cycle-allocation-{allocation + 1}">'
            f"Allocation {allocation + 1}</h2><table><thead><tr>"
            "<th>Arm</th><th>Slices</th><th>Call 1 [ms/slice]</th>"
            "<th>Call 2 [ms/slice]</th><th>Steps per lane, in lane order</th>"
            "<th>Largest fresh residual</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    maximum = {
        key: max(row[key] for pair in gaps for row in pair)
        for key in gaps[0][0]
        if key not in ("call", "lanes")
    }
    print("PRECISION", json.dumps(maximum), flush=True)
    print(
        "TABLES",
        json.dumps(
            [
                {
                    "base_steps": max(pair[0]["rows"][-1]["steps"]),
                    "head_steps": max(pair[1]["rows"][-1]["steps"]),
                    "base_ms": pair[0]["rows"][-1]["per_slice_ms"],
                    "head_ms": pair[1]["rows"][-1]["per_slice_ms"],
                }
                for pair in pairs
            ]
        ),
        flush=True,
    )
    text = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence">
<meta name="plan-evidence-for" content="sub-millisecond-forward-path">
<meta name="plan-verifies" content="sub-millisecond-forward-path#s3">
<title>Bank residual cycle | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css"><link rel="stylesheet"
href="/_shared/dashboard.css">
</head><body><main class="plan-doc"><h1 id="bank-solve-cycle-landing">Bank residual
cycle</h1>
<p>Releasing a frozen active set on non-contraction alone took 45 steps at sixteen
lanes, against 22 at the base. Candidate halvings are therefore checked against a fresh
read, in a bounded device loop with at most eight trials. Each lane retains its best
valid trial; fresh validity and the original residual budget still govern
acceptance.</p>
<p>The base trace saw a residual jump from 1.644200958e-7 to 3.187730151e-10 with 174
edge-interval entries changing by at most 6.45276e-9 and 81 membership fractions by at
most 2.07998e-9. Fragment selection and normal-form selection did not change. This
identifies moving continuous geometry, but does not isolate one formula as the cause.
The callback trace itself changes numerical ordering and has a 14-step tail; it is
excluded from all timing claims.</p>
"""
        + "".join(tables)
        + """
<p>Each allocation ran both arms at one and sixteen slices, with two executable calls
after explicit compilation. Call 1 includes first-execution device initialization; Call
2 is the repeated converged wall used for the comparison. Both calls have identical lane
counts within each arm. The seed, analytic oracle and solver are imported from the
existing bank measurement module. The base is a git archive, with module paths and
revisions printed in each log.</p>
<figure id="bank-solve-cycle-history"><img
src="/nova/figures/sub-millisecond-forward-path/bank-solve-cycle/residual-history.svg"
alt="All lane fresh residuals at each Newton iterate, base beside backtracking in two
allocations, on shared logarithmic axes"><figcaption>Sixteen lanes from the diverted
550-cell seed, multipliers 2e-5 through 6e-5. The blue line is the slowest lane in each
arm; all other lanes are grey. Histories come directly from SolveResult.residual_path
and stop at each lane's acceptance. The dotted line is the unchanged 1e-10 residual
limit. Calls shown are the second calls in each allocation. Text bounds and overlaps
were checked and the PNG rendering was inspected.</figcaption></figure>
<p>Largest head/base and oracle ratios across both allocations and calls: <code>"""
        + escape(json.dumps(maximum))
        + """
</code>. Flux differs by at most the stated ratio of analytic span (budget 1e-8); axis
and X-point by a ratio of minor radius (budget 5e-4); boundary flux by a ratio of span
(budget 1e-4). The oracle boundary ratio is a conservative all-lane upper bound: the
first lane's oracle error plus the largest difference from that lane.</p>
<p>Raw receipts and gate logs: <code>"""
        + escape(str(args.run))
        + """
</code>. The controlled continuous-kink test starts at its exact two-cycle; the physical
test restarts all sixteen certificate lanes after two production updates, below the
freeze threshold, and requires fresh acceptance within eight further updates.</p>
</main></body></html>"""
    )
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(text)


if __name__ == "__main__":
    main()
