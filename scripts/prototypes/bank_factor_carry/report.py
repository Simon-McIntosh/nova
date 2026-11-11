"""Validate paired factor counts and draw their per-slice assembly costs."""

import argparse
from html import escape
import json
from pathlib import Path
import hashlib
import re
import subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.prototypes.bank_warm_predictor.report import (
    precision,
    require_budget,
    BOUNDS,
)
from scripts.prototypes.bank_solve_profile.plot import save_checked

MAIN = "#929292"
CARRIED = "#1f5d92"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    allocation = (args.run / "counts-gpu-allocation.log").read_text()
    assert allocation.splitlines()[-1] == "EXIT=0"
    job = re.search(r"JOB=(\d+)", allocation).group(1)
    scheduler = subprocess.check_output(
        ["sacct", "-n", "-X", "-j", job, "-o", "State,Elapsed,ExitCode", "--parsable2"],
        text=True,
    ).strip()
    state, elapsed, exit_code = scheduler.split("|")[:3]
    assert state == "COMPLETED" and exit_code == "0:0", scheduler
    suites = json.loads((args.run / "suite-receipts.json").read_text())
    armed = json.loads((args.run / "armed-suite-receipts.json").read_text())
    cpu_rows = []
    for arm in ("base", "head"):
        assert armed[arm]["completed"] and armed[arm]["exit_status"] == 0
        for name in ("test_bank_drive_continuation", "test_bank_solve_driver"):
            receipt = suites[arm + "-" + name]
            assert receipt["completed"] and receipt["exit_status"] == 0
            log = Path(receipt["log_path"]).read_text()
            assert log.splitlines()[-1] == "EXIT=0"
            passed = re.findall(r"(\d+) passed", log)[-1]
            cpu_rows.append(f"<tr><td>{arm}</td><td>{name}</td><td>{passed}</td></tr>")
    negative = (args.run / "unguarded-reuse-mutant.log").read_text()
    assert negative.splitlines()[-1] == "EXIT=1"
    assert (
        "FAILED tests/test_bank_solve_driver.py::"
        "test_carried_factor_first_update_is_guarded[4.0-1]"
    ) in negative
    data, arrays, inputs = {}, {}, {}
    for arm in ("main", "head"):
        stem = args.run / f"counts-{arm}-measured"
        metadata = json.loads(stem.with_suffix(".inputs.json").read_text())
        assert metadata["completed"] and metadata["rows"] == 14
        for suffix in (".json", ".npz"):
            assert (
                hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
                == metadata["receipt_sha256"][suffix]
            )
        data[arm] = json.loads(stem.with_suffix(".json").read_text())
        arrays[arm] = dict(np.load(stem.with_suffix(".npz"), allow_pickle=False))
        inputs[arm] = metadata["inputs"]
        assert {(row["step"], row["slice"]) for row in data[arm]} == {
            (step, index) for step in (1e-5, 1e-3) for index in range(1, 8)
        }
        assert all(
            len(row["residual"]) == 8
            and all(row["converged"])
            and max(row["residual"]) <= 1e-10
            for row in data[arm]
        )
    for path in (
        "nova/equilibrium/support_read.py",
        "nova/equilibrium/drive_continuation.py",
    ):
        assert (
            hashlib.sha256((args.root / path).read_bytes()).hexdigest()
            == (inputs["head"]["sources"][path])
        ), path
    for key in (
        "width",
        "steps",
        "slices",
        "seed_sha256",
        "bank_sha256",
        "bank_structure",
        "jax",
        "numpy",
        "x64",
        "backend",
        "device_kind",
    ):
        assert inputs["main"][key] == inputs["head"][key], key
    maxima = dict.fromkeys(BOUNDS, 0.0)
    for row in data["head"]:
        key = f"{row['step']:g}-{row['slice']}"
        values = precision(arrays["main"], arrays["head"], key, row["minor_radius"])
        require_budget(values)
        for name in maxima:
            maxima[name] = max(maxima[name], float(np.max(values[name])))
    refused = []
    key = "1e-05-1"
    reference = arrays["main"]
    for name in BOUNDS:
        mutant = {k: v.copy() for k, v in reference.items()}
        if name == "flux":
            mutant[key + "-flux"][0, 0] += 2e-8 * abs(reference[key + "-span"][0])
        elif name == "boundary":
            mutant[key + "-boundary"][0] += 2e-4 * abs(reference[key + "-span"][0])
        else:
            mutant[key + "-position"][0, 0 if name == "axis" else 1, 0] += (
                1e-3 * data["main"][0]["minor_radius"]
            )
        try:
            require_budget(
                precision(reference, mutant, key, data["main"][0]["minor_radius"])
            )
        except AssertionError:
            refused.append(name)
    assert set(refused) == set(BOUNDS)
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    handles = []
    for ax, step in zip(axes, (1e-5, 1e-3), strict=True):
        for arm, color, shift in (("main", MAIN, -0.035), ("head", CARRIED, 0.035)):
            rows = [r for r in data[arm] if r["step"] == step]
            (line,) = ax.plot(
                [r["slice"] + shift for r in rows],
                [max(r["refactors"]) for r in rows],
                color=color,
                marker="o",
                linewidth=3 if arm == "head" else 2.6,
                zorder=3 if arm == "head" else 2,
            )
            if step == 1e-5:
                handles.append(line)
        ax.set_xlabel("slice")
        ax.set_ylabel("factor assemblies per batch")
        ax.set_xticks(range(1, 8))
        ax.set_ylim(
            -0.15,
            max(max(r["refactors"]) for rows in data.values() for r in rows) + 0.5,
        )
        highest = int(ax.get_ylim()[1])
        stride = max(1, (highest + 5) // 6)
        ax.set_yticks(range(0, highest + 1, stride))
        ax.text(0.05, 0.93, f"drive step {step:g}", transform=ax.transAxes, fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.legend(
        handles,
        ["secant", "secant + carried factor"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.97),
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.84, wspace=0.25)
    destination = (
        args.root
        / "docs/figures/sub-millisecond-forward-path/bank-factor-carry"
        / "factor-assemblies.svg"
    )
    save_checked(fig, destination)
    fig.savefig(args.run / "factor-assemblies.png", dpi=100)
    table = []
    totals = {}
    for arm, rows in data.items():
        totals[arm] = {
            key: sum(max(r[key]) for r in rows)
            for key in ("reads", "updates", "refactors")
        }
        for r in rows:
            reads, updates, factors = (
                max(r[k]) for k in ("reads", "updates", "refactors")
            )
            estimate = (reads * 35.7 + updates * (44.9 - 25.5) + factors * 25.5) / 16
            possible_retry = (
                arm == "head" and r["slice"] > 1 and factors > 0 and reads > updates + 1
            )
            estimate_text = f"{estimate:.3f}"
            if possible_retry:
                estimate_text += f"–{estimate + (44.9 - 25.5) / 16:.3f}"
            table.append(
                f"<tr><td>{arm}</td><td>{r['step']:g}</td><td>{r['slice']}</td>"
                f"<td>{reads}</td><td>{updates}</td><td>{factors}</td>"
                f"<td>{max(r['residual']):.3e}</td>"
                f"<td>{r['batch_ms'] / 8 / 1000:.3f}</td>"
                f"<td>{estimate_text}</td></tr>"
            )
    summary = dict(
        totals=totals,
        precision_maxima=maxima,
        precision_controls_refused=refused,
        terminal_residual_max=max(
            max(r["residual"]) for rows in data.values() for r in rows
        ),
        lane_slices=224,
        cpu_per_file={key: item["exit_status"] for key, item in suites.items()},
        cpu_armed={key: item["exit_status"] for key, item in armed.items()},
    )
    (args.run / "count-report.json").write_text(json.dumps(summary, indent=2))
    precision_rows = "".join(
        f"<tr><td>{k}</td><td>{maxima[k]:.3e}</td><td>{BOUNDS[k]:g}</td></tr>"
        for k in BOUNDS
    )
    total_rows = "".join(
        f"<tr><td>{arm}</td><td>{counts['reads']}</td>"
        f"<td>{counts['updates']}</td><td>{counts['refactors']}</td></tr>"
        for arm, counts in totals.items()
    )
    assembly_reduction = 1 - totals["head"]["refactors"] / totals["main"]["refactors"]
    fragment = f"""<section data-reckon="fragment" id="bank-factor-carry-result"
data-plan="sub-millisecond-forward-path" data-node="bank-factor-carry">
<h2>Newton factors carried across certified drive slices</h2>
<p>The secant driver carries the last accepted Newton factor. The first carried
correction is provisional: its fresh residual must contract by a finite ratio at
most the contraction limit, otherwise the factor is rebuilt at the incumbent.
Only the terminal fresh read at residual at most 1e-10 certifies a slice.
Rejected continuation attempts do not replace the accepted factor.</p>
<p>Both arms use the first eight saved lanes, drive steps 1e-5 and 1e-3 and seven
slices at each step, with one warmup and one measured process per arm. The bank
and seed hashes match across arms. Main uses secant seeds with fresh factors;
head enables guarded factor reuse within solves and carries factors across
slices. Assemblies fall by {assembly_reduction:.1%}, accompanied by more fresh
reads and accepted updates. All {summary["lane_slices"]} lane/slice results
certify, at maximum residual {summary["terminal_residual_max"]:.3e}.</p>
<table id="bank-factor-carry-totals"><thead><tr><th>arm</th>
<th>total fresh reads</th><th>total accepted updates</th>
<th>total factor assemblies</th></tr></thead><tbody>{total_rows}</tbody></table>
<p>These totals sum the largest lane count of each of the fourteen batched
slices. Primed slow-ramp slices skip assembly throughout their solves, while
their accepted update count rises from one in main to two or three in head.
At the fast step, the head reaches nineteen updates in slice five; its factor
is rebuilt in slice six. The assembly count alone does not describe this
additional correction and fresh-read work.</p>
<figure id="bank-factor-carry-counts"><img
src="/nova/figures/sub-millisecond-forward-path/bank-factor-carry/factor-assemblies.svg"
width="1400" alt="Factor assemblies at each slice for secant and carried factors
at two drive steps"><figcaption>Batch-maximum factor assemblies; secant is grey
and carried factors blue. Small horizontal offsets separate coincident points.
These are titan P100 width-eight counts. The producer rendered the committed SVG
at 1400 pixels and inspected its labels and marks.</figcaption></figure>
<table id="bank-factor-carry-table"><thead><tr><th>arm</th><th>drive step</th>
<th>slice</th><th>fresh reads</th><th>accepted updates</th><th>factor assemblies</th>
<th>fresh residual, maximum</th><th>titan P100 wall per slice [s]</th>
<th>H200 width-16 estimate per slice [ms]</th></tr></thead>
<tbody>{"".join(table)}</tbody></table>
<p>The H200 column is an estimate from batch counts: (35.7 ms × reads +
19.4 ms × updates + 25.5 ms × factor assemblies) / 16, using the measured H200
update cost 44.9 ms and its 25.5 ms assembly/factor portion. It is not an H200
timing measurement. Titan walls carry no speed verdict. The retained run.sh
accepts the repository root and GPFS run directory as arguments, defaulting to
width sixteen for a later H200 submission. The lead deferred H200 timing to a
later node. Update counts are accepted Newton
iterations. A rejected carried proposal can add one linear solve before the
fresh correction, while every associated read and assembly is counted. Where
the saved counters cannot distinguish that retry from ordinary damping, the
estimate is a range charging zero to one additional 19.4 ms batch correction.</p>
<table id="bank-factor-carry-precision"><thead><tr>
<th>quantity, ratio to local scale</th><th>maximum against main</th><th>bound</th>
</tr></thead><tbody>{precision_rows}</tbody></table>
<p>Every lane meets the locked flux, axis, X-point and boundary-flux budgets
against main. Synthetic violations of all four precision bounds were refused.
Receipts and full logs are retained under <code>{escape(str(args.run))}</code>.
The original API reproduction reports <code>TypeError: solve_batch() got an
unexpected keyword argument 'factor_cache'</code>. Five focused cases pass;
the paired sirius files have zero added failures. The exact armed two-file
command also passes both arms. The unguarded-reuse scratch mutation fails the
non-contracting first-update test with EXIT=1; the same test passes at head.</p>
<table id="bank-factor-carry-cpu"><thead><tr><th>arm</th><th>fresh process file</th>
<th>passed</th></tr></thead><tbody>{"".join(cpu_rows)}</tbody></table>
<p>Titan job {job} completed in {elapsed} against its authorized eight-hour
allocation, with scheduler state {state}, exit {exit_code}, and allocation-log
EXIT=0. One warmup and one measured process ran for each arm. Source digests
for the measured head match the delivered product files; JSON and array receipt
digests validate. The figure passed the shared text-bounds and overlap check
and was rendered from SVG and visually inspected before commit.</p>
</section>"""
    (
        args.root
        / "docs/evidence/fragments/sub-millisecond-forward-path/bank-factor-carry.html"
    ).write_text(fragment + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
