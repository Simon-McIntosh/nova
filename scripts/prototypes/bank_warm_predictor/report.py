"""Judge predictor counts and precision; retain device walls as information."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.bank_solve_profile.plot import save_checked

ARMS = ("plain", "secant", "tangent")
STEPS = (1e-5, 1e-3)
BOUNDS = dict(flux=1e-8, axis=5e-4, saddle=5e-4, boundary=1e-4)
WALL_LABEL = "titan P100, width 8, not a speed verdict"


def precision(reference, candidate, key, radius):
    span = np.abs(reference[key + "-span"])
    assert np.all(np.isfinite(span)) and np.all(span > 0)
    position = (
        np.linalg.norm(
            candidate[key + "-position"] - reference[key + "-position"], axis=-1
        )
        / radius
    )
    return dict(
        flux=np.max(
            np.abs(candidate[key + "-flux"] - reference[key + "-flux"]), axis=-1
        )
        / span,
        axis=position[:, 0],
        saddle=position[:, 1],
        boundary=np.abs(candidate[key + "-boundary"] - reference[key + "-boundary"])
        / span,
    )


def require_budget(values):
    for name, bound in BOUNDS.items():
        assert np.all(np.isfinite(values[name])) and np.max(values[name]) <= bound, (
            name,
            values[name],
        )


def total_reads(row):
    return max(row["reads"]) + int(row["arm"] == "tangent")


def duration(milliseconds):
    """Use a readable SI prefix for an informational duration."""
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.3f} s"
    if milliseconds >= 1:
        return f"{milliseconds:.3f} ms"
    return f"{milliseconds * 1000:.3f} µs"


def score(rows):
    return (
        sum(total_reads(row) for row in rows),
        sum(max(row["updates"]) + int(row["arm"] == "tangent") for row in rows),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    allocation = (args.run / "counts-gpu-allocation.log").read_text()
    assert allocation.splitlines()[-1] == "EXIT=0"
    job = re.search(r"JOB=(\d+)", allocation).group(1)
    scheduler = subprocess.check_output(
        ["sacct", "-n", "-X", "-j", job, "-o", "State,ExitCode", "--parsable2"],
        text=True,
    ).strip()
    assert scheduler.split("|")[:2] == ["COMPLETED", "0:0"], scheduler
    suites = json.loads((args.run / "suite-receipts.json").read_text())
    cpu_rows = []
    for arm in ("base", "head"):
        for name in ("test_bank_drive_continuation", "test_bank_solve_driver"):
            receipt = suites[arm + "-" + name]
            assert receipt["completed"] and receipt["exit_status"] == 0
            passed = re.findall(r"(\d+) passed", Path(receipt["log_path"]).read_text())[
                -1
            ]
            cpu_rows.append(
                f"<tr><td>{arm}</td><td>{name}</td><td>{passed}</td><td>0</td></tr>"
            )
    negative = (args.run / "unchanged-seed-mutant.log").read_text()
    assert (
        "FAILED tests/test_bank_drive_continuation.py::"
        "test_linear_stream_predicts_next_state_to_rounding"
    ) in negative
    assert negative.splitlines()[-1] == "EXIT=1"
    data, arrays, hashes = {}, {}, set()
    for arm in ARMS:
        stem = args.run / f"counts-{arm}-first"
        metadata = json.loads(stem.with_suffix(".inputs.json").read_text())
        assert metadata["completed"] and metadata["rows"] == 14
        encoded = json.dumps(
            metadata["inputs"], sort_keys=True, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(encoded).hexdigest() == metadata["inputs_hash"]
        hashes.add(metadata["inputs_hash"])
        for suffix in (".json", ".npz"):
            assert (
                hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
                == metadata["receipt_sha256"][suffix]
            )
        data[arm] = json.loads(stem.with_suffix(".json").read_text())
        arrays[arm] = dict(np.load(stem.with_suffix(".npz"), allow_pickle=False))
        assert len(data[arm]) == 14
        assert {(r["step"], r["slice"]) for r in data[arm]} == {
            (step, index) for step in STEPS for index in range(1, 8)
        }
        for row in data[arm]:
            assert row["arm"] == arm and row["width"] == 8
            assert (
                len(row["residual"])
                == len(row["seed_residual"])
                == len(row["reads"])
                == len(row["updates"])
                == 8
            )
            assert len(row["converged"]) == 8 and all(row["converged"])
            assert (
                np.all(np.isfinite(row["residual"])) and max(row["residual"]) <= 1e-10
            )
            assert np.all(np.isfinite(row["seed_residual"]))
    assert len(hashes) == 1
    maxima = {name: 0.0 for name in BOUNDS}
    for arm in ARMS:
        for row in data[arm]:
            key = f"{row['step']:g}-{row['slice']}"
            values = precision(arrays["plain"], arrays[arm], key, row["minor_radius"])
            require_budget(values)
            for name in maxima:
                maxima[name] = max(maxima[name], float(np.max(values[name])))
    reference, key = arrays["plain"], "1e-05-1"
    radius = data["plain"][0]["minor_radius"]
    refused = []
    for quantity in BOUNDS:
        mutant = {name: value.copy() for name, value in reference.items()}
        if quantity == "flux":
            mutant[key + "-flux"][0, 0] += 2e-8 * abs(reference[key + "-span"][0])
        elif quantity == "boundary":
            mutant[key + "-boundary"][0] += 2e-4 * abs(reference[key + "-span"][0])
        else:
            mutant[key + "-position"][0, 0 if quantity == "axis" else 1, 0] += (
                1e-3 * radius
            )
        try:
            require_budget(precision(reference, mutant, key, radius))
        except AssertionError:
            refused.append(quantity)
    assert set(refused) == set(BOUNDS)
    chosen = min(
        ("secant", "tangent"), key=lambda arm: (*score(data[arm]), arm != "secant")
    )
    scores = {
        arm: dict(
            total_fresh_reads=score(data[arm])[0],
            total_linear_solves=score(data[arm])[1],
        )
        for arm in ARMS
    }
    reduction = (
        1 - scores[chosen]["total_fresh_reads"] / scores["plain"]["total_fresh_reads"]
    )
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100, sharey=True)
    styles = dict(
        plain=("#929292", "--", "s"),
        secant=("#3366cc", "-", "o"),
        tangent=("#222222", ":", "^"),
    )
    for ax, step in zip(axes, STEPS, strict=True):
        for arm in ARMS:
            rows = [row for row in data[arm] if row["step"] == step]
            colour, line, marker = styles[arm]
            ax.plot(
                [
                    row["slice"] + {"plain": -0.08, "secant": 0, "tangent": 0.08}[arm]
                    for row in rows
                ],
                [total_reads(row) for row in rows],
                color=colour,
                linestyle=line,
                marker=marker,
                fillstyle="none",
                linewidth=3 if arm == chosen else 2.6,
                markersize=9,
                label=arm,
                zorder=4 if arm == chosen else 2,
            )
        ax.set_xlabel("slice index")
        ax.set_xticks(range(1, 8))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.text(
            0.5,
            1.04,
            f"drive step {step:g}",
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
        )
    top = max(total_reads(row) for rows in data.values() for row in rows)
    axes[0].set_yticks(range(1, top + 1))
    axes[0].set_ylim(0.7, top + 0.8)
    axes[0].set_ylabel("fresh reads per slice [count]")
    fig.legend(
        *axes[0].get_legend_handles_labels(), ncol=3, loc="upper center", frameon=False
    )
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.78, wspace=0.16)
    save_checked(fig, args.figure)
    fig.savefig(args.run / "reads-per-slice-preview.png", dpi=100)
    table, summary = [], []
    for step in STEPS:
        for arm in ARMS:
            rows = [row for row in data[arm] if row["step"] == step]
            base_rows = [row for row in data["plain"] if row["step"] == step]
            summary.append(
                dict(
                    arm=arm,
                    step=step,
                    total_reads=sum(total_reads(r) for r in rows),
                    reads=sorted({total_reads(r) for r in rows}),
                    updates=sorted({value for row in rows for value in row["updates"]}),
                    read_reduction=1
                    - sum(total_reads(r) for r in rows)
                    / sum(total_reads(r) for r in base_rows),
                    seed_residual=max(max(row["seed_residual"]) for row in rows),
                    terminal_residual=max(max(row["residual"]) for row in rows),
                    per_slice_ms=float(np.mean([row["per_slice_ms"] for row in rows])),
                    wall_label=WALL_LABEL,
                )
            )
            for row in rows:
                table.append(
                    f"<tr><td>{arm}</td><td>{step:g}</td><td>{row['slice']}</td>"
                    f"<td>{max(row['seed_residual']):.3e}</td><td>{total_reads(row)}</td>"
                    f"<td>{min(row['reads'])}–{max(row['reads'])}</td>"
                    f"<td>{min(row['updates'])}–{max(row['updates'])}</td>"
                    f"<td>{max(row['residual']):.3e}</td>"
                    f"<td>{row['per_slice_ms'] / 1000:.3f}</td>"
                    f"<td>{duration(row['prediction_ms'] / 8)}</td></tr>"
                )
    score_rows = "".join(
        f"<tr><td>{arm}</td><td>{scores[arm]['total_fresh_reads']}</td>"
        f"<td>{scores[arm]['total_linear_solves']}</td></tr>"
        for arm in ARMS
    )
    step_rows = "".join(
        f"<tr><td>{row['step']:g}</td><td>{row['arm']}</td>"
        f"<td>{row['total_reads']}</td>"
        f"<td>{row['read_reduction']:.1%}</td>"
        f"<td>{', '.join(map(str, row['updates']))}</td></tr>"
        for row in summary
    )
    precision_rows = "".join(
        f"<tr><td>{name}</td><td>{maxima[name]:.3e}</td><td>{BOUNDS[name]:g}</td></tr>"
        for name in BOUNDS
    )
    slow_rows = [
        row for row in data[chosen] if row["step"] == 1e-5 and row["slice"] > 1
    ]
    fast_rows = [row for row in data[chosen] if row["step"] == 1e-3]
    fast_seed_min = min(min(row["seed_residual"]) for row in fast_rows[1:])
    fast_seed_max = max(max(row["seed_residual"]) for row in fast_rows[1:])
    slow_seed_min = min(min(row["seed_residual"]) for row in slow_rows)
    slow_seed_max = max(max(row["seed_residual"]) for row in slow_rows)
    terminal_max = max(max(row["residual"]) for rows in data.values() for row in rows)
    report = dict(
        chosen_predictor=chosen,
        scores=scores,
        read_reduction=reduction,
        summary=summary,
        precision_maxima=maxima,
        precision_mutants_refused=refused,
        terminal_residual_max=terminal_max,
        certified_lane_slices=sum(
            len(row["converged"]) for rows in data.values() for row in rows
        ),
        measured_processes_per_arm=1,
        inputs_hash=next(iter(hashes)),
        wall_label=WALL_LABEL,
    )
    (args.run / "predictor-report.json").write_text(json.dumps(report, indent=2))
    fragment = f"""<section data-reckon="fragment" id="bank-warm-predictor-fragment"
data-plan="sub-millisecond-forward-path" data-node="bank-warm-predictor">
<h2>Drive predictor counts: {chosen}</h2>
<p>The selected predictor is <strong>{chosen}</strong>, selected by total fresh
reads across the fourteen slices, with total linear solves breaking a tie and
secant preferred if those counts tie. Counts include the tangent predictor’s own
fresh read and linear solve. The totals below sum the largest lane count of each
of the fourteen batched slices. The chosen predictor reduces total fresh reads
by {reduction:.1%} against plain, with {scores["plain"]["total_linear_solves"]}
to {scores[chosen]["total_linear_solves"]} linear solves.</p>
<table id="bank-warm-predictor-totals"><thead><tr><th>arm</th>
<th>total fresh reads</th><th>total linear solves</th></tr></thead>
<tbody>{score_rows}</tbody></table>
<table id="bank-warm-predictor-step-summary"><thead><tr><th>drive step</th>
<th>arm</th><th>total fresh reads</th><th>read reduction against plain</th>
<th>updates per slice, observed counts</th></tr></thead>
<tbody>{step_rows}</tbody></table>
<p>The slow-ramp objective is reached once two converged history states exist:
slices 2–7 at step 1e-5 each take one update and two fresh reads. Their seed fresh
residual lies between {slow_seed_min:.3e} and {slow_seed_max:.3e}, against
1.796e-3 for plain. Slice 1 retains the documented plain-start fallback.</p>
<p>The step 1e-3 result has a clear limit: secant takes three updates and four
fresh reads on every slice. Neither measured predictor reaches one update at
that step. Secant seed residuals range from {fast_seed_min:.3f} to {fast_seed_max:.3f}
after the fallback,
so seed residual alone does not rank the nonlinear solve cost. The count
reduction is established; a one-update fast-ramp claim is not.</p>
<p>Each arm uses the first eight saved lanes at drive steps 1e-5 and 1e-3 over
slices 1–7, with one warmup and one measured process. Matching inputs hash:
<code>{next(iter(hashes))}</code>. Slice 1 has only one history state, so secant
falls back to the plain warm seed. Every lane is certified by its terminal fresh
read at residual at most 1e-10; neither predictor changes that certificate.</p>
<figure id="bank-warm-predictor-reads"><img
src="/nova/figures/sub-millisecond-forward-path/bank-warm-predictor/reads-per-slice.svg"
alt="Fresh reads per slice for plain, secant and tangent at two drive steps"
width="1400"><figcaption>Read counts include the tangent predictor’s one fresh
read. Plain is grey, secant blue and tangent dotted near-black. Small horizontal
offsets separate coincident counts. Walls are informational only: {WALL_LABEL}.
The H200 width-16 wall remains pending a free card.</figcaption></figure>
<table id="bank-warm-predictor-counts"><thead><tr><th>arm</th><th>drive step</th>
<th>slice</th><th>seed fresh residual, max</th><th>total fresh reads, max</th>
<th>solve reads, lane range</th><th>updates, lane range</th>
<th>terminal fresh residual, max</th><th>total wall / slice [s]; {WALL_LABEL}</th>
<th>predictor wall / slice; {WALL_LABEL}</th></tr></thead>
<tbody>{"".join(table)}</tbody></table>
<p>All eight lanes in every slice meet the precision budget against plain:
flux ≤1e-8 of span, axis and X-point ≤5e-4 of minor radius, boundary flux ≤1e-4
of span. All 336 lane/slice results certify; their maximum terminal fresh
residual is {terminal_max:.3e}. The same check refused synthetic violations
of each of the four precision bounds.</p>
<table id="bank-warm-predictor-precision"><thead><tr><th>ratio to local scale</th>
<th>observed maximum</th><th>budget</th></tr></thead>
<tbody>{precision_rows}</tbody></table>
<table id="bank-warm-predictor-cpu"><thead><tr><th>arm</th><th>file</th>
<th>passed</th><th>exit</th></tr></thead><tbody>{"".join(cpu_rows)}</tbody></table>
<p>The paired sirius files have zero added failures. The unchanged-seed mutant
fails the linear-stream check with EXIT=1; the head passes it. Imported module,
revision and working-directory provenance are retained in the logs.</p>
<p>Titan job {job} completed with scheduler record <code>{escape(scheduler)}</code>
and allocation EXIT=0. Logs, per-slice arrays, input hashes and source digests
are under <code>{escape(str(args.run))}</code>, using
<code>counts-{{plain,secant,tangent}}-first</code> prefixes. Width 16 previously
failed to allocate 15,120,805,120 bytes on P100; width 8 retains the original
saved streams’ first eight lanes. Titan wall time does not decide speed.
The shared saver passed its SVG text-bounds and overlap check. The producer
rendered the SVG at 1400 pixels wide and inspected it: axis labels and legend
are legible, with no clipped or colliding text.</p>
</section>"""
    args.fragment.write_text(fragment + "\n")
    print("SUMMARY " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
