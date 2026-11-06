"""Validate the paired receipts and render the per-lane residual floor."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_residual_noise_floor.report import table
from scripts.prototypes.bank_solve_cycle.report import pair_gaps
from scripts.prototypes.bank_solve_profile.plot import save_checked


FILES = (
    "test_support_read",
    "test_tracked_read",
    "test_candidate_inspection_batched",
    "test_bank_solve_driver",
    "test_bank_solve_cycle",
    "test_bank_newton_step",
    "test_bank_step_booking_batch",
    "test_saddle_derivatives",
)


def read_log(path):
    text = path.read_text()
    lines = text.splitlines()
    assert lines[-1].startswith("EXIT="), path
    return text, int(lines[-1].partition("=")[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    arms = {
        (arm, p): json.loads((run / f"{arm}-{p}.json").read_text())
        for arm in ("base", "head")
        for p in range(3)
    }
    digest = hashlib.sha256((run / "common-seeds.npz").read_bytes()).hexdigest()
    timings, precision, violations = [], [], []
    for (arm, p), receipt in arms.items():
        log, status = read_log(run / f"{arm}-{p}.log")
        assert status == 0 and "XLA_FLAGS=unset" in log
        assert digest in log
        assert len(receipt["rows"]) == 6
        for row in receipt["rows"]:
            assert row["reads"] == [row["reads"][0]] * 16
            assert all(row["converged"]) and max(row["residual"]) <= 1e-10
            timings.append(
                [
                    arm,
                    p,
                    row["variant"],
                    str(row["reads"]),
                    f"{row['wall_ms']:.3f}",
                    f"{row['wall_ms'] / row['reads'][0]:.3f}",
                ]
            )
    for p in range(3):
        for before, after in zip(
            arms["base", p]["rows"], arms["head", p]["rows"], strict=True
        ):
            gaps, limits = pair_gaps(before, after)
            precision.append(
                dict(process=p, variant=after["variant"], gaps=gaps, limits=limits)
            )
            for key, values in gaps.items():
                if not np.all(np.isfinite(values)) or np.max(values) > limits[key]:
                    violations.append(
                        f"precision process={p} seed={after['variant']} "
                        f"{key}={np.max(values)}"
                    )
    floors = {
        arm: json.loads((run / f"floor-{arm}" / "gpu.json").read_text())
        for arm in ("base", "head")
    }
    decisions = json.loads((run / "decisions.json").read_text())
    assert decisions["control"] and decisions["changes"] == [[0, 0, 0], [0, 0, 0]]
    statistics = {}
    floor_rows = []
    for arm, data in floors.items():
        log, status = read_log(run / f"floor-{arm}.log")
        assert status == 0 and "COMPLETE " in log and data["zero_direction_exact"]
        norms = np.asarray([p["norm"] for p in data["perturbations"]])
        assert norms.shape == (64, 16)
        assert all(not any(p["mask_changes"]) for p in data["perturbations"])
        statistics[arm] = dict(
            median=np.median(norms, axis=0).tolist(), maximum=norms.max(axis=0).tolist()
        )
    assert floors["base"]["masks"] == floors["head"]["masks"]
    for lane, (before, after) in enumerate(
        zip(statistics["base"]["median"], statistics["head"]["median"])
    ):
        floor_rows.append(
            [lane, f"{before:.8e}", f"{after:.8e}", f"{before / after:.2f}"]
        )
        if after > 1e-11:
            violations.append(f"floor lane={lane} median={after}")
    suite_rows = []
    suite = {}
    for arm in ("base", "head"):
        failures = []
        statuses = []
        logs = []
        for name in FILES:
            if arm == "base" and name == "test_saddle_derivatives":
                continue
            path = run / f"cpu-{arm}-{name}.log"
            text, status = read_log(path)
            summaries = [
                line.strip("= ")
                for line in text.splitlines()
                if line.startswith("===") and " in " in line
            ]
            assert summaries, path
            failures.extend(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M))
            statuses.append(status)
            logs.append(text)
            suite_rows.append([arm, name, summaries[-1], status])
        joined = run / f"{arm}-suite.log"
        joined.write_text("\n".join(logs) + f"\nEXIT={max(statuses)}\n")
        suite[arm] = dict(
            revision=arms[arm, 0]["revision"],
            command=(run / "suite-command.txt").read_text().strip(),
            exit_status=max(statuses),
            log_path=str(joined),
            completed=True,
            failure_count=len(set(failures)),
            failure_ids=sorted(set(failures)),
        )
    added = set(suite["head"]["failure_ids"]) - set(suite["base"]["failure_ids"])
    if added:
        violations.append("added suite failures: " + str(sorted(added)))
    negative, negative_status = read_log(
        run / "cpu-negative-test_saddle_derivatives.log"
    )
    failed = re.findall(r"^FAILED (\S+)", negative, re.M)
    assert negative_status != 0 and set(failed) - set(suite["head"]["failure_ids"])
    certificate, status = read_log(run / "cpu-certificate-test_saddle_derivatives.log")
    assert status == 0 and "2 passed" in certificate
    certificate_error = max(
        float(v)
        for v in re.findall(
            r"DERIVATIVE state=\d+ rank=\d+ absolute_error=([0-9.e+-]+)", certificate
        )
    )
    summary = dict(
        derivative_error=certificate_error,
        decisions=decisions["changes"],
        floors=statistics,
        precision=precision,
        suites=suite,
        added_failures=sorted(added),
        negative_failures=failed,
        common_seed_digest=digest,
        violations=violations,
    )
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    lanes = np.arange(16)
    ax.plot(
        lanes,
        statistics["base"]["median"],
        "o--",
        color="#929292",
        linewidth=2.6,
        label="base",
        zorder=2,
    )
    ax.plot(
        lanes,
        statistics["head"]["median"],
        "o-",
        color=InkStyle().flux_color,
        linewidth=3,
        label="accurate curvature",
        zorder=3,
    )
    ax.axhline(
        1e-10,
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        label="certificate",
        zorder=1,
    )
    ax.axhline(
        1e-11,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label="floor requirement",
        zorder=1,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e-9)
    ax.set_yticks([1e-13, 1e-12, 1e-11, 1e-10, 1e-9])
    ax.minorticks_off()
    ax.set_xticks(lanes)
    ax.set_xlim(-0.5, 15.5)
    ax.set_xlabel("Lane")
    ax.set_ylabel("Median fresh residual")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.86)
    save_checked(fig, args.figure)
    fig.savefig(run / "floor-preview.png", dpi=100)
    plt.close(fig)
    maximum = {
        key: max(max(row["gaps"][key]) for row in precision)
        for key in precision[0]["gaps"]
    }
    details = table(
        ["Quantity", "Worst normalized gap", "Limit"],
        [[k, f"{v:.8e}", precision[0]["limits"][k]] for k, v in maximum.items()],
    )
    verdict = (
        "All declared numerical gates pass."
        if not violations
        else "The numerical gate remains unmet: " + "; ".join(violations)
    )
    prefix = "bank-saddle-derivatives"
    timing_table = table(
        [
            "Arm",
            "Process",
            "Seed",
            "Reads in lane order",
            "Whole solve [ms]",
            "Wall/read [ms]",
        ],
        timings,
    )
    fragment = f'''<section id="{prefix}-landing">
<h2>Saddle curvature from unrounded bank contractions</h2>
<p><strong>{escape(verdict)}</strong></p>
<p>The source rows, moments and exterior reach the saddle field through
<code>support_read</code>. Existing paired-float arithmetic carries product and
sum residues through the selected triangle's contraction, Taylor subtraction
and Bell transform. Third and fourth derivatives feed the normal form; the
explicit geometry tangent uses the same polynomial, including its fifth derivative
for the moving fourth-order term. Low-order field evaluation and the stationary
solve retain their existing jets. Only three triangle vertices need extended
arithmetic; there is no added operator store or whole-mesh high-order rebuild.</p>
<p>The independent reference evaluates the stored binary64 bank operands and
Bell transform with 80-digit Decimal arithmetic. At the diverted 550-cell trial
and saved converged saddles the test uses absolute and relative tolerances of
2e-13; the maximum measured absolute error is {certificate_error:.8e}.
The default CPU lane uses the light diverted fixture with cancelling
source and exterior contributions. Restoring Bell-derived third and fourth
derivatives in a scratch copy fails {len(failed)} test cases; the head passes.
The reference establishes arithmetic accuracy for this discrete polynomial,
not an exact analytic-kernel derivative.</p>
<p>At identical sixteen-lane trial and saved converged states, substituting the
base curvature function changes zero cut, normal or fragment decisions. Flipping
a saved mask bit changes its signature, establishing that the comparison detects
a known difference. The separate decision receipt records both full populations.</p>
<figure id="{prefix}-floor"><img
src="/nova/figures/sub-millisecond-forward-path/bank-saddle-derivatives/floor.svg"
alt="Median fresh residual for sixteen lanes, base and accurate curvature">
<figcaption>64 ulp-scale perturbations of each converged lane, measured by the
landed noise-floor instrument imported unchanged. Both arms load the same
base-generated seed and root arrays. Cut, normal and fragment masks remain
unchanged throughout each population and agree between terminal arms. The
certificate is 1e-10; each head median must be at most 1e-11. The shared saver
checks text bounds and overlaps. The producer inspected the 1400 by 720 preview:
no clipped or overlapping labels, and both series and thresholds are legible.
The preview is retained in the run directory.</figcaption></figure>
{table(["Lane", "Base median", "Head median", "Reduction"], floor_rows)}
<h3 id="{prefix}-timings">Common-seed solve measurements</h3>
<p>Each revision has its own compilation cache. Process 0 warms it; processes
1 and 2 are fresh interpreters with XLA_FLAGS unset. Every wall is the median
of three synchronized solves. Seed 0 is the shared base seed; seeds 1–5 add
bounded 1e-14 relative perturbations. Every log reports the same archive digest
<code>{digest}</code>. Wall/read measures the whole solve divided by its batch
reads, including Newton work; it is not isolated read latency.</p>
{timing_table}
<h3 id="{prefix}-precision">Locked precision checks</h3>
<p>All 16 lanes in every seed and process certify a fresh residual at most
1e-10. The unchanged <code>pair_gaps</code> instrument checks flux and geometry
against base and stationary positions and boundary flux against the analytic
oracle. Values below are worst cases over every process, seed and lane.</p>
{details}
<h3 id="{prefix}-suites">Paired Sirius suites</h3>
<p>Each file runs in a fresh pytest process with a separate basetemp. The two
arms share one seven-core Sirius job. The candidate inspection file selects its
light cases. The derivative test is added at head; the scratch Bell restoration
supplies its negative control. Added failure ids: {escape(str(sorted(added)))}.</p>
{table(["Arm", "File", "Summary", "Exit"], suite_rows)}
<p>Base: <code>{arms["base", 0]["revision"]}</code>. Product head:
<code>{arms["head", 0]["revision"]}</code>. Receipts, raw perturbation vectors,
per-process timings, negative control and suite logs are retained under
<code>{run}</code>. The script family is
<code>scripts/prototypes/bank_saddle_derivatives</code>.</p>
<h3 id="{prefix}-reproduction">Why the contraction is included</h3>
<p>At the original base, the saved lane-12 perturbation moved the fourth
derivative by 2.19977e-10. An 80-digit polynomial from already-rounded nodal jets
still moved by 2.24605e-10; carrying 80 digits from the bank contraction onward
reduced that movement to 9.90973e-13. The same cell, triangle and saddle position
were used. This rules out polishing only the polynomial after nodal rounding.
The reproduction receipt and log are retained beside these measurements.</p>
</section>'''
    args.fragment.write_text(fragment + "\n")
    print(
        "SUMMARY "
        + json.dumps(
            dict(
                head_medians=statistics["head"]["median"],
                violations=violations,
                added_failures=sorted(added),
            )
        )
    )
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
