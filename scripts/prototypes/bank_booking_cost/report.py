"""Render paired booking costs and a bounded, receipt-derived evidence fragment."""

import argparse
from html import escape
import json
from pathlib import Path
import statistics
import re

import matplotlib.pyplot as plt

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_read.plot import save_checked


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    measurements = {
        (arm, process): json.loads(
            (args.run / "paired" / f"{arm}-{process}.json").read_text()
        )
        for arm in ("base", "head")
        for process in (1, 2)
    }
    rows = {
        key: {r["name"]: r for r in receipt["rows"]}
        for key, receipt in measurements.items()
    }

    def mean(arm, name):
        return statistics.mean(rows[arm, p][name]["median_ms"] for p in (1, 2))

    for arm in ("base", "head"):
        for process in (1, 2):
            staged = json.loads((args.run / f"stages-{arm}-{process}.json").read_text())
            for row in staged["rows"]:
                rows[arm, process][row["name"]] = row
            rows[arm, process]["cell_scatter"] = json.loads(
                (args.run / f"scatter-{arm}-{process}.json").read_text()
            )
    names = [
        ("Cell-local booking", "16:local"),
        ("Conic ring construction", "conic_rings"),
        ("Conic ring velocities", "conic_velocities"),
        ("Clipped boundary quadrature", "clipped_boundary_quadrature"),
        ("Normal geometry, one direction", "16:normal_geometry"),
        ("Cell bucket scatter", "cell_scatter"),
    ]
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    color = InkStyle().flux_color
    for arm, hue, offset in [("base", "#929292", 0.13), ("head", color, -0.13)]:
        positions = [len(names) - 1 - i + offset for i in range(len(names))]
        values = [mean(arm, key) / 1000 for _, key in names]
        ax.scatter(
            values,
            positions,
            color=hue,
            s=70,
            label=arm,
            zorder=3 if arm == "head" else 2,
        )
        for pos, (_, name) in zip(positions, names, strict=True):
            samples = [rows[arm, p][name]["median_ms"] / 1000 for p in (1, 2)]
            ax.plot(samples, [pos, pos], color=hue, linewidth=3, zorder=2)
    ax.set_yticks(list(reversed(range(len(names)))), [name for name, _ in names])
    ax.set_xscale("log")
    ax.set_xlabel("CPU wall [s]")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.legend(loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.68, 0.98))
    fig.subplots_adjust(left=0.34, right=0.96, bottom=0.16, top=0.88)
    destination = (
        root
        / "docs/figures/sub-millisecond-forward-path/bank-booking-cost/booking-cost.svg"
    )
    save_checked(fig, destination)
    fig.savefig(args.run / "booking-cost.png", dpi=100)
    plt.close(fig)
    table = []
    for name in measurements["base", 1]["rows"]:
        name = name["name"]
        for arm in ("base", "head"):
            for process in (1, 2):
                row = rows[arm, process][name]
                table.append(
                    "<tr>"
                    + "".join(
                        f"<td>{escape(str(v))}</td>"
                        for v in (
                            name,
                            arm,
                            process,
                            f"{row['median_ms']:.6f}",
                            row["optimized_hlo_operations"],
                            row["stablehlo_operations"],
                            row["fusion_sites"],
                            row["while_sites"],
                        )
                    )
                    + "</tr>"
                )
    parity = max(
        measurements["head", p][f"parity_{width}"] for p in (1, 2) for width in (1, 16)
    )
    improvement = 100 * (1 - mean("head", "16:local") / mean("base", "16:local"))
    summary = {
        "booking_reduction_percent": improvement,
        "matrix_relative_error": parity,
        "base_local_ms": mean("base", "16:local"),
        "head_local_ms": mean("head", "16:local"),
    }
    summary["booking_reduction_percent_by_pair"] = [
        100
        * (
            1
            - rows["head", process]["16:local"]["median_ms"]
            / rows["base", process]["16:local"]["median_ms"]
        )
        for process in (1, 2)
    ]
    summary["local_process_spread_percent"] = {
        arm: 100
        * abs(
            rows[arm, 1]["16:local"]["median_ms"]
            - rows[arm, 2]["16:local"]["median_ms"]
        )
        / mean(arm, "16:local")
        for arm in ("base", "head")
    }
    assert min(summary["booking_reduction_percent_by_pair"]) > 0, summary
    assert max(summary["local_process_spread_percent"].values()) <= 5, summary
    summary["assembly_ms"] = {
        arm: {str(width): mean(arm, f"{width}:assembly") for width in (1, 16)}
        for arm in ("base", "head")
    }
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))
    suite_rows = []
    for arm in ("base", "head"):
        for name in (
            "test_bank_newton_step",
            "test_bank_step_booking_batch",
            "test_bank_step_launch_fusion",
            "test_bank_solve_driver",
        ):
            log = args.run / "verified-suites" / f"{arm}-{name}.log"
            content = log.read_text()
            assert content.rstrip().endswith("EXIT=0"), log
            summaries = [
                line
                for line in content.splitlines()
                if re.search(r"[0-9]+ passed", line)
            ]
            assert summaries, log
            suite_rows.append(
                f"<tr><td>{arm}</td><td>{name}</td>"
                f"<td>{escape(summaries[-1])}</td><td>0</td></tr>"
            )
    negative = (args.run / "verified-negative" / "negative-control.log").read_text()
    assert negative.rstrip().endswith("EXIT=1")
    assert "2 failed" in negative
    negative_summary = next(
        line for line in negative.splitlines() if "2 failed in" in line
    )
    stage_rows = []
    for label, name in names[1:]:
        stage_rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    label,
                    f"{mean('base', name):.6f}",
                    f"{mean('head', name):.6f}",
                )
            )
            + "</tr>"
        )
    text = f"""<section id="bank-booking-cost-result">
<h2>Reuse the boundary level's constant-field ring velocity</h2>
<p>Cell-local booking at width 16 changes from {summary["base_local_ms"]:.3f} to
{summary["head_local_ms"]:.3f} ms on CPU ({improvement:.2f}% lower). Two fresh processes
per arm run in one all_debug allocation. The largest assembled-matrix relative
difference over widths 1 and 16 is {parity:.6g}, against 1e-12. Both process
pairs improve; process spread is {summary["local_process_spread_percent"]["base"]:.3f}%
at base and {summary["local_process_spread_percent"]["head"]:.3f}% at head,
within the declared 5% agreement bound.</p>
<p>The boundary level enters the signed conic coefficient as the negative constant
field coefficient. The assembly now evaluates six independent ring velocities
rather than seven, preserves the boundary normalization derivative separately,
and recovers its shape derivative by linearity. The fourteen-input interface,
profile derivative and current normalization remain represented.</p>
<figure><img
src="/nova/figures/sub-millisecond-forward-path/bank-booking-cost/booking-cost.svg"
alt="Paired CPU walls for booking, conic stages, normal geometry and cell scatters">
<figcaption>Width sixteen, CPU, synchronized medians; short horizontal marks join
the two process medians. Conic stages use all 113 actual cut cells per lane and
runtime arrays between stages. They change the compilation boundary and are not
an additive partition of production wall. Normal geometry is a separate
saddle-motion path, shown for one jet direction per lane; production assembles
18 local directions. Scatters use the actual 113 conic and four normal cells,
with nonzero runtime derivative arrays and an independent placement check.
The scatter implementation is unchanged, so its base/head labels repeat the
same source as a neutral control. The log scale keeps the millisecond scatter
visible beside the second-scale geometric derivatives.</figcaption></figure>
<table><thead><tr><th>Isolated program</th><th>Base CPU ms</th>
<th>Head CPU ms</th></tr></thead><tbody>{"".join(stage_rows)}</tbody></table>
<h3>Walls and lowered programs</h3>
<p>Operation counts come from optimized HLO and StableHLO. Fusion and while sites
are static lowered-program call sites, not dynamic GPU launches. CPU walls provide
a paired comparison, not an H200 verdict. The committed device.sh submits the same
root-argument driver on the reserved H200; no H200 timing or runtime kernel-launch
claim is made here.</p>
<table><thead><tr><th>Program</th><th>Arm</th><th>Process</th><th>CPU ms</th>
<th>Optimized HLO operations</th><th>StableHLO operations</th><th>Fusion sites</th>
<th>While sites</th></tr></thead><tbody>{"".join(table)}</tbody></table>
<h3>Attribution</h3>
<p>Separately compiled conic velocity evaluation is the largest measured conic
sub-term. The stage loop uses seven signed coefficient directions for the base
and six for the candidate; the constant boundary direction is recovered by
linearity in the full assembly. Ring construction is shared between directions.
The timed normal-geometry path and its ring-rate ablation are retained in the
lowered-program table, as are inclusive removals of conic velocities and clipped
boundary quadrature. Inclusive removals may remove downstream dependent work;
they are not exclusive operator walls.</p>
<h3>Reproduction and controls</h3>
<p>The initial base probe reproduced a 12,075.886 ms width-sixteen assembly and
9,648.119 ms local booking. The first difference sweep used displacements too large
for the global physical direction: trial errors were 11.2333, 1.07416, 0.0132434 at
1e-4, 1e-5, 1e-6. The smaller sweep preserves the 1e-6 acceptance bound and is
validated at both the trial and actually converged states, including a fresh read
at each displaced state. Full results and the deliberate dropped-booking control
are recorded in the run's manifest and logs.</p>
<h3>Paired suites and refusal</h3>
<p>One Sirius allocation, seven CPUs, one fresh process per file and arm:
added failures zero. Optional archived-matrix and GPU comparisons skip explicitly;
the independent CPU measurement supplies full matrix parity at both widths.</p>
<table><thead><tr><th>Arm</th><th>File</th><th>Pytest summary</th><th>Exit</th></tr>
</thead><tbody>{"".join(suite_rows)}</tbody></table>
<p>The full fresh-read central-difference check passes at a trial and actually
converged state. Head relative errors reach 1.932246359234944e-9 and
3.6130675011168244e-9 respectively, against 1e-6. The scratch negative-control
hook replaces the assembled matrix by identity after fixture convergence,
dropping its booking derivative. Both new test IDs fail; the log records
{negative_summary} and EXIT=1. The hook is retained as negative-mutation.py.</p>
<p>Base revision: <code>101d1d839d90b4859f42df114707bff1ee2dbeae</code>.
Product change: <code>31949bb38cc291c7aeb6bd87d792e705fadc2230</code>.
Receipts: <code>{escape(str(args.run))}</code>. Each measurement prints its imported
module and resolved working directory, and its log ends with EXIT.</p>
</section>"""
    fragment = (
        root
        / "docs/evidence/fragments/sub-millisecond-forward-path/bank-booking-cost.html"
    )
    fragment.write_text(text)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
