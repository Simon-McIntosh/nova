"""Compare paired device receipts and draw all fresh-residual trajectories."""

import argparse
from html import escape
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_profile.plot import save_checked


PAIR_LIMITS = {"flux": 1e-8, "axis": 5e-4, "saddle": 5e-4, "boundary_flux": 1e-4}
ORACLE_LIMIT = 1e-4


def pair_gaps(before, after):
    """Per-lane precision gaps between a paired base and head row, and their bounds.

    Every value is normalised by the unit it is judged against, so a lane's gap
    and its limit are directly comparable. Both arms' oracle gaps are reported
    because a base arm outside the budget is as much a failure as a head arm.
    """
    gaps = {}
    for key, unit in (
        ("flux", before["span"]),
        ("axis", before["minor_radius"]),
        ("saddle", before["minor_radius"]),
        ("boundary_flux", before["span"]),
    ):
        difference = np.asarray(after[key]) - np.asarray(before[key])
        if key in ("axis", "saddle"):
            difference = np.linalg.norm(difference, axis=-1)
        if key == "flux":
            difference = np.max(np.abs(difference), axis=-1)
        gaps[key] = (np.abs(difference) / unit).tolist()
    for arm, row in (("base", before), ("head", after)):
        roots = np.asarray(row["oracle_roots"])
        for key, point in (("axis", 0), ("saddle", 1)):
            gaps[f"{arm}_oracle_{key}"] = (
                np.linalg.norm(np.asarray(row[key]) - roots[point], axis=-1)
                / row["minor_radius"]
            ).tolist()
        # The first-lane absolute oracle error plus its companion difference
        # bounds every lane without cancellation between signed errors.
        gaps[f"{arm}_oracle_boundary"] = (
            row["oracle"]["boundary_ratio"]
            + np.abs(np.asarray(row["boundary_flux"]) - row["boundary_flux"][0])
            / row["span"]
        ).tolist()
    limits = dict(PAIR_LIMITS)
    limits.update(
        {
            key: ORACLE_LIMIT if key.endswith("boundary") else 5e-4
            for key in gaps
            if "oracle" in key
        }
    )
    return gaps, limits


def compare(base, head):
    gaps = []
    for before, after in zip(base["rows"], head["rows"], strict=True):
        assert (before["lanes"], before["call"]) == (after["lanes"], after["call"])
        values, _ = pair_gaps(before, after)
        row = dict(lanes=after["lanes"], call=after["call"])
        for key in ("flux", "axis", "saddle", "boundary_flux"):
            row[key] = float(np.max(values[key]))
        row["residual"] = max(after["residual"])
        for key in ("axis", "saddle", "boundary"):
            row[f"oracle_{key}"] = float(np.max(values[f"head_oracle_{key}"]))
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
    parser.add_argument("--pairs", nargs=2, default=["contract-one", "contract-two"])
    args = parser.parse_args()
    pairs = [
        (
            json.loads((args.run / f"{label}-base.json").read_text()),
            json.loads((args.run / f"{label}-head.json").read_text()),
        )
        for label in args.pairs
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
    last_iterate = max(max(arm["rows"][-1]["steps"]) for pair in pairs for arm in pair)
    upper_iterate = max(5, 5 * math.ceil(last_iterate / 5))
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
                    linewidth=3 if lane == worst else 2.6,
                    zorder=3 if lane == worst else 2,
                )
            ax.axhline(1e-10, color="#666666", linestyle=":", linewidth=1.2, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(1e-13, 1)
            ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
            ax.set_xlim(0, upper_iterate)
            ax.set_xticks(np.arange(0, upper_iterate + 1, 5))
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
            Line2D([], [], color="#929292", lw=2.6, label="other lanes"),
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
    provenance = []
    for allocation, pair in enumerate(pairs):
        label = args.pairs[allocation]
        provenance.append(
            f"Allocation {allocation + 1}: base {pair[0]['revision']}; "
            f"head {pair[1]['revision']}; logs {label}-base.log and {label}-head.log."
        )
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
    summaries = []
    for number, pair in enumerate(pairs, 1):
        before, after = (arm["rows"][-1] for arm in pair)
        summaries.append(
            f"Allocation {number}: slowest lane {max(before['steps'])} to "
            f"{max(after['steps'])} steps; repeated per-slice wall "
            f"{before['per_slice_ms']:.2f} to {after['per_slice_ms']:.2f} ms."
        )
    summary = " ".join(summaries)
    precision_rows = []
    for label, key, limit in [
        ("Fresh residual", "residual", 1e-10),
        ("Head/base flux / span", "flux", 1e-8),
        ("Head/base axis / minor radius", "axis", 5e-4),
        ("Head/base X-point / minor radius", "saddle", 5e-4),
        ("Head/base boundary flux / span", "boundary_flux", 1e-4),
        ("Oracle axis / minor radius", "oracle_axis", 5e-4),
        ("Oracle X-point / minor radius", "oracle_saddle", 5e-4),
        ("Oracle boundary flux / span, upper bound", "oracle_boundary", 1e-4),
    ]:
        precision_rows.append(
            f"<tr><td>{label}</td><td>{maximum[key]:.4e}</td><td>{limit:.1e}</td></tr>"
        )
    precision_table = (
        "<table><thead><tr><th>Quantity</th><th>Largest ratio</th>"
        "<th>Budget</th></tr></thead><tbody>"
        + "".join(precision_rows)
        + "</tbody></table>"
    )
    gate_rows = []
    for name in (
        "cycle-seed-base-test_bank_solve_cycle",
        "cpu-suite-head-test_bank_solve_cycle",
        "contract-suite-head-test_bank_solve_cycle",
        "stall-base-test_bank_solve_cycle",
        "stall-head-test_bank_solve_cycle",
        "cpu-suite-base-test_bank_solve_driver",
        "cpu-suite-head-test_bank_solve_driver",
        "contract-suite-head-test_bank_solve_driver",
        "cpu-suite-base-test_support_read",
        "cpu-suite-head-test_support_read",
        "contract-suite-head-test_support_read",
    ):
        path = args.run / (name + ".log")
        lines = path.read_text().splitlines() if path.exists() else []
        summaries = [
            line.strip("= ")
            for line in lines
            if line.startswith("===") and " in " in line
        ]
        complete = bool(lines and lines[-1].startswith("EXIT="))
        outcome = summaries[-1] if summaries and complete else "running or pending"
        gate_rows.append(f"<tr><td>{name}.log</td><td>{escape(outcome)}</td></tr>")
    gates = (
        '<h2 id="bank-solve-cycle-tests">CPU tests on sirius</h2>'
        "<table><thead><tr><th>Log</th><th>Recorded result</th></tr>"
        "</thead><tbody>" + "".join(gate_rows) + "</tbody></table>"
    )
    text = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="bank-solve-cycle">
<meta name="plan-evidence-for" content="sub-millisecond-forward-path">
<meta name="plan-verifies" content="sub-millisecond-forward-path#s3">
<title>Bank residual cycle | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css"><link rel="stylesheet"
href="/_shared/dashboard.css">
</head><body><main class="plan-doc"><h1 id="bank-solve-cycle-landing">Bank residual
cycle</h1>
<p id="bank-solve-cycle-cpu-blocker">CPU restart diagnostics are recorded in
restart-base.log and restart-head.log; run restart_report.py after this report
to reconcile their completed histories and insert the measured walls.</p>
"""
        + "<p>"
        + escape(summary)
        + "</p>"
        + """
<p>Releasing a frozen active set on non-contraction alone took 45 steps at sixteen
lanes and 1691.63 ms per slice in an exploratory allocation. A separate base
allocation took 22 steps and 833.69 ms per slice; these are not a paired timing
comparison. Candidate halvings are therefore checked against a fresh
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
        + "<p>"
        + escape(" ".join(provenance))
        + "</p>"
        + """
<p id="bank-solve-cycle-single-slice">The 345.10 to 707.94 ms single-slice
regression in allocation 2 accompanies <strong>two additional accepted Newton
updates</strong>: five at head versus three at base, and one lock release versus
zero. After update 2, base residuals go from 1.613e-9 directly to 1.153e-11;
head takes 1.603e-9 → 7.922e-10 → 5.839e-10 → 5.201e-12. Its accepted
update magnitudes over those last three updates are 2.706e-10, 6.683e-11 and
2.017e-10. These are contracting accepted states, with no return to the
1.6e-7 cycle level. The smaller accepted updates are consistent with damping
near the root. The receipts retain accepted residual and update paths, but no
rejected trial history or read counter; the exact number of trial halvings in
that H200 call is therefore not recoverable. Two is the measured increase in
accepted updates, not a claimed count of rejected backtracking trials.</p>
<p>The preliminary backtracking arm required a twofold decrease on each trial.
It reduced the sixteen-slice tails from 18 to 7 and 29 to 9, but regressed the
single-slice wall from 344.81 to 4250.59 ms and 346.80 to 1026.64 ms.
Those receipts remain in pair-one and pair-two. The tables above use the
revised rule: halve only when the fresh trial residual fails to decrease,
while retaining the 0.5 contraction threshold for lock release and factor reuse.</p>
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
<p>Head/base and oracle ratios across both allocations and calls:</p>"""
        + precision_table
        + """
<p>Flux differs by at most the stated ratio of analytic span (budget 1e-8); axis
and X-point by a ratio of minor radius (budget 5e-4); boundary flux by a ratio of span
(budget 1e-4). The oracle boundary ratio is a conservative all-lane upper bound: the
first lane's oracle error plus the largest difference from that lane.</p>
<p>Raw receipts and gate logs: <code>"""
        + escape(str(args.run))
        + """
</code>. The controlled continuous-kink test starts at its exact two-cycle; the physical
test restarts all sixteen certificate lanes after two production updates, below the
freeze threshold, and requires fresh acceptance within eight further updates.</p>
"""
        + gates
        + """
<p>The exact continuous-kink cycle is the step-count negative control:
base runs all six permitted updates and fails the two-update bound; head
accepts in one. The unchanged-selection stall also fails at base (zero
unfreezes) and passes at head (two). The physical CPU restart is an additional
nonregression check: base already passes its eight-update budget, with a
maximum of seven, so it is not claimed as the discriminating test.</p>
</main></body></html>"""
    )
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(text)


if __name__ == "__main__":
    main()
