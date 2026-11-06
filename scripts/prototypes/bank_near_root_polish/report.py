"""Check common-seed precision and render batch-read sensitivity from receipts."""

import argparse
from html import escape
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_cycle.report import pair_gaps
from scripts.prototypes.bank_solve_read.plot import save_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    arms = {
        (arm, process): json.loads((args.run / f"{arm}-{process}.json").read_text())
        for arm in ("base", "head")
        for process in (0, 1, 2)
    }
    common_digest = hashlib.sha256(
        (args.run / "common-seeds.npz").read_bytes()
    ).hexdigest()
    for (arm, process), receipt in arms.items():
        assert len(receipt["rows"]) == 6
        assert [row["variant"] for row in receipt["rows"]] == list(range(6))
        log = (args.run / f"{arm}-{process}.log").read_text().splitlines()
        assert log[-1] == "EXIT=0"
        hashes = [
            json.loads(line.partition(" ")[2])["sha256"]
            for line in log
            if line.startswith("COMMON_SEEDS ")
        ]
        assert hashes == [common_digest]
    suite_rows = []
    suite_receipts = {}
    for arm in ("base", "head"):
        logs, failures, statuses, outcomes = [], [], [], []
        for name in (
            "test_bank_solve_cycle",
            "test_bank_solve_driver",
            "test_support_read",
            "test_bank_newton_step",
            "test_equilibrium_solve",
        ):
            path = args.run / f"cpu-{arm}-{name}.log"
            content = path.read_text()
            lines = content.splitlines()
            assert lines[-1].startswith("EXIT="), path
            summaries = [
                line.strip("= ")
                for line in lines
                if line.startswith("===") and " in " in line
            ]
            assert summaries, path
            status = int(lines[-1].split("=")[1])
            statuses.append(status)
            outcomes.append(summaries[-1])
            failures.extend(re.findall(r"^(?:FAILED|ERROR) (\S+)", content, re.M))
            logs.append(content)
            suite_rows.append((arm, name, summaries[-1], status))
        combined = args.run / f"{arm}-suite.log"
        combined.write_text("\n".join(logs) + f"\nEXIT={max(statuses)}\n")
        suite_receipts[arm] = dict(
            revision=arms[arm, 1]["revision"],
            command=(args.run / "suite-command.txt").read_text().strip(),
            exit_status=max(statuses),
            log_path=str(combined),
            completed=True,
            failure_count=len(set(failures)),
            failure_ids=sorted(set(failures)),
            summaries=outcomes,
        )
    added_failures = sorted(
        set(suite_receipts["head"]["failure_ids"])
        - set(suite_receipts["base"]["failure_ids"])
    )
    (args.run / "suite-receipts.json").write_text(json.dumps(suite_receipts, indent=2))
    perturbations = []
    with np.load(args.run / "common-seeds.npz") as seeds:
        for variant in range(1, 6):
            relative = [
                float(
                    np.max(
                        np.abs(seeds[f"state_{variant}_{c}"] - seeds[f"state_0_{c}"])
                    )
                    / np.max(np.abs(seeds[f"state_0_{c}"]))
                )
                for c in range(3)
            ]
            assert all(0 < value <= 1.1e-14 for value in relative)
            perturbations.append(relative)
    comparisons = []
    rows = []
    certification = []
    for process in (0, 1, 2):
        for before, after in zip(
            arms["base", process]["rows"], arms["head", process]["rows"], strict=True
        ):
            assert before["variant"] == after["variant"]
            gaps, limits = pair_gaps(before, after)
            passed = {
                key: bool(
                    np.all(np.isfinite(value))
                    and np.all(np.asarray(value) <= limits[key])
                )
                for key, value in gaps.items()
            }
            comparisons.append(
                dict(
                    process=process,
                    variant=after["variant"],
                    gaps=gaps,
                    limits=limits,
                    passed=passed,
                )
            )
            for arm, row in (("base", before), ("head", after)):
                count = row["reads"][0]
                assert row["reads"] == [count] * 16
                rows.append(
                    (
                        arm,
                        process,
                        row["variant"],
                        count,
                        row["steps"],
                        row["wall_ms"],
                        row["wall_ms"] / count,
                    )
                )
                for lane, (residual, converged) in enumerate(
                    zip(row["residual"], row["converged"], strict=True)
                ):
                    certification.append(
                        (arm, process, row["variant"], lane, residual, converged)
                    )
    counts = {
        arm: [[r["reads"][0] for r in arms[arm, p]["rows"]] for p in (0, 1, 2)]
        for arm in ("base", "head")
    }
    measured_costs = {}
    for arm in ("base", "head"):
        selected = [row for row in rows if row[0] == arm and row[1] in (1, 2)]
        measured_costs[arm] = {
            "reads": [min(row[3] for row in selected), max(row[3] for row in selected)],
            "wall_ms": [
                min(row[5] for row in selected),
                max(row[5] for row in selected),
            ],
            "wall_per_read_ms": [
                min(row[6] for row in selected),
                max(row[6] for row in selected),
            ],
        }
    passed_precision = all(all(row["passed"].values()) for row in comparisons)
    certified = all(
        row[-1] and np.isfinite(row[-2]) and row[-2] <= 1e-10 for row in certification
    )
    stable = len(set(sum(counts["head"], []))) == 1
    # The precision predicate must refuse a known perturbation of a measured lane.
    control = json.loads(json.dumps(arms["head", 1]["rows"][0]))
    control["flux"][0][0] += 1e-6 * control["span"]
    gaps, limits = pair_gaps(arms["base", 1]["rows"][0], control)
    assert max(gaps["flux"]) > limits["flux"]
    summary = dict(
        reads=counts,
        fixed_head_reads=stable,
        all_certified=certified,
        precision_passed=passed_precision,
        measured_costs=measured_costs,
        maximum_fresh_residual=max(
            row[-2] for row in certification if row[0] == "head"
        ),
        precision=comparisons,
        precision_control_flux=max(gaps["flux"]),
        seed_relative_perturbations=perturbations,
        common_seed_sha256=common_digest,
        added_failures=added_failures,
    )
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    for arm, color, dash, width, layer in (
        ("base", "#929292", "--", 2.6, 2),
        ("head", InkStyle().flux_color, "-", 3.0, 3),
    ):
        ax.plot(
            range(6),
            counts[arm][1],
            color=color,
            linestyle=dash,
            linewidth=width,
            marker="o",
            label=arm,
            zorder=layer,
        )
        if counts[arm][1] != counts[arm][2]:
            ax.plot(
                range(6),
                counts[arm][2],
                color=color,
                linestyle=":",
                linewidth=width,
                marker="s",
                label=arm + " repeat",
                zorder=layer,
            )
    ax.set_xticks(
        range(6),
        [
            "common",
            "perturbation 1",
            "perturbation 2",
            "perturbation 3",
            "perturbation 4",
            "perturbation 5",
        ],
    )
    ax.set_xlabel("Initial moment seed set")
    ax.set_ylabel("Fresh batch reads")
    maximum_reads = max(sum(counts["base"] + counts["head"], []))
    ax.set_ylim(0, maximum_reads + 2)
    ax.set_yticks(np.arange(0, maximum_reads + 1, 5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.legend(loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.86)
    save_checked(fig, args.figure)
    args.figure.write_text(
        "\n".join(line.rstrip() for line in args.figure.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.run / "batch-reads.png", dpi=100)
    plt.close(fig)

    def table(headers, values):
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{escape(str(v))}</th>" for v in headers)
            + "</tr></thead><tbody>\n"
            + "\n".join(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                for row in values
            )
            + "\n</tbody></table>"
        )

    maximum = {
        key: max(max(row["gaps"][key]) for row in comparisons)
        for key in comparisons[0]["gaps"]
    }
    precision_rows = [
        (key, f"{value:.9e}", comparisons[0]["limits"][key])
        for key, value in maximum.items()
    ]
    wall_rows = [
        (
            arm,
            process,
            variant,
            count,
            ", ".join(map(str, steps)),
            f"{wall:.3f}",
            f"{per_read:.3f}",
        )
        for arm, process, variant, count, steps, wall, per_read in rows
    ]
    cert_rows = [
        (arm, process, variant, lane, f"{residual:.12e}", valid)
        for arm, process, variant, lane, residual, valid in certification
    ]
    wall_table = table(
        [
            "Arm",
            "Process",
            "Seed",
            "Reads per lane (all 16)",
            "Updates in lane order",
            "Whole solve [ms]",
            "Wall/read [ms]",
        ],
        wall_rows,
    )
    figure_src = (
        "/nova/figures/sub-millisecond-forward-path/"
        "bank-near-root-polish/batch-reads.svg"
    )
    base_wall = measured_costs["base"]["wall_ms"]
    head_wall = measured_costs["head"]["wall_ms"]
    base_reads = measured_costs["base"]["reads"]
    head_reads = measured_costs["head"]["reads"]
    text = f"""<section id="bank-near-root-polish-landing">
<h2>Fixed near-root polishing and fresh certification</h2>
<p><strong>Fixed read count: {stable}; every lane freshly certified: {certified};
paired precision budgets: {passed_precision}.</strong>
Head uses {head_reads[0]}–{head_reads[1]} reads per lane across all six seed sets,
against {base_reads[0]}–{base_reads[1]} at base. The fixed schedule costs more:
whole-solve wall is {head_wall[0]:.3f}–{head_wall[1]:.3f} ms at head against
{base_wall[0]:.3f}–{base_wall[1]:.3f} ms at base in the two fresh measured processes.
This meets the count-stability measure with a latency regression.</p>
<p>
The batch enters polishing when every active lane has a valid finite residual at most
1e-8.
Sixteen full Newton updates have no backtracking. Each lane retains its lowest-residual
valid state,
then a separate fresh read certifies the retained state against the unchanged 1e-10
limit.
A failed certificate resumes adaptive search once; the update cap remains binding.</p>
<p>The band is above the measured 1.6e-9 to 2.9e-9 residuals at update 2 and below
the preceding
3.7e-4 to 4.4e-4 residuals. The recorded adaptive near-root tail needed up to four
further updates.
An eight-step measurement still missed seed 5, lane 10: its best residual was
1.3227463568910025e-10. Sixteen steps double that measured insufficient budget.
The certificate, validity check and best-state selection decide success; this margin
alone does not.</p>
<p>At base 73864287c5db07bac083e61dca1f160859452351, the controlled seed-noise
regression
reproduces [2, 2] versus [3, 3] reads. Head 42039a936366f4b71a04c3573c46d90213fb5ec6
gives
[18, 18] for both. The scratch mutation certifies against the tracked reading instead
of the
fresh one, falsely accepting state 1.0 rather than the certified 1.5; the regression
fails
with EXIT=1. Invalid full trials are separately shown unable to replace a valid
state.</p>
<p>H200 allocation 1307321 uses one card, two CPU cores and 48 GB under the reservation.
The seed archive was built once at base in allocation 1307314 and reused byte for
byte in this final allocation; both revisions load the same arrays.
Five perturbations add a deterministic bounded random change of 1e-14 relative to
each moment
channel's maximum absolute value. Seeds and lane root positions are identical. Every
process
prints the common archive digest, revision, imported module, cwd, device and x64
flag.</p>
<p>Each revision has its own compilation cache. Process 0 warms it; processes 1 and 2
are fresh
interpreters with XLA_FLAGS unset. Each table wall is the median of three
synchronized solves
after an in-process warm-up. Wall per read is total solve wall divided by batch reads,
including Newton work; it is not isolated read latency. No equal-work speedup is
inferred
from different read counts.</p>
<figure id="bank-near-root-polish-reads"><img src="{figure_src}"
alt="Fresh batch reads for common initial moments and five tiny perturbations, base
against fixed polishing">
<figcaption>Sixteen lanes share every batch read. Common seeds and five perturbations at
1e-14 relative; both measured processes are shown where counts differ. Grey dashed is
base;
blue solid is the fixed polish. The shared saver checks text bounds and overlaps.
The producer inspected the 1400 by 720 PNG rendering; only the SVG is
committed.</figcaption></figure>
<h3 id="bank-near-root-polish-processes">Every process (0 warms the cache)</h3>
{wall_table}
<h3 id="bank-near-root-polish-precision">Paired state and analytic-oracle budgets</h3>
<p>The existing pair_gaps instrument checks flux and geometry against base, and axis,
X-point and boundary flux against the independent analytic oracle. Boundary oracle
errors
are conservative all-lane bounds. Every comparison retains all sixteen lane values in
summary.json. A controlled flux perturbation of 1e-6 of the span is rejected by the
1e-8 predicate.</p>
{table(["Quantity", "Largest normalized gap", "Limit"], precision_rows)}
<details id="bank-near-root-polish-certificates"><summary>Every lane's fresh
certificate</summary>
{table(["Arm", "Process", "Seed", "Lane", "Fresh residual", "Converged"], cert_rows)}
</details>
<p id="bank-near-root-polish-receipts">Raw arrays, per-process receipts,
negative-control.log and all paired suite logs are under
<code>{escape(str(args.run))}</code>. The original reproduction and eight-step
receipts remain in <code>{escape(str(args.run.parent))}</code>.
The paired sirius allocation is 1307322. Each file has its own fresh interpreter
and basetemp. There are {len(added_failures)} added failure ids.</p>
<h3 id="bank-near-root-polish-suites">Complete paired CPU suites</h3>
{table(["Arm", "File", "Pytest summary", "Exit"], suite_rows)}
</section>
"""
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(text)
    print(
        "SUMMARY",
        json.dumps(
            {key: value for key, value in summary.items() if key != "precision"}
        ),
        flush=True,
    )
    assert stable and certified and passed_precision and not added_failures


if __name__ == "__main__":
    main()
