"""Publish paired scalar-booking receipts with a checked timing comparison."""

import argparse
from html import escape
import json
from pathlib import Path
import re
import statistics

import matplotlib.pyplot as plt

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_read.plot import save_checked


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    run = args.run.resolve()
    measurement = run / "final-measurement"
    validation = run / "final-validation"
    sweep = json.loads((run / "sweep-fixed.json").read_text())
    receipts = {
        (arm, process): json.loads((measurement / f"{arm}-{process}.json").read_text())
        for arm in ("base", "head")
        for process in (1, 2)
    }
    assert all(receipt["complete"] for receipt in receipts.values())
    assert len({r["job"] for r in receipts.values()}) == 1
    assert len({r["instrument_sha256"] for r in receipts.values()}) == 1
    rows = {
        key: {(r["width"], r["kind"]): r for r in value["rows"]}
        for key, value in receipts.items()
    }

    def mean(arm, width, kind):
        return statistics.mean(rows[arm, p][width, kind]["median_ms"] for p in (1, 2))

    summary = {}
    table = []
    for width in (1, 16):
        for kind in ("booking", "carried", "full"):
            base, head = (mean(arm, width, kind) for arm in ("base", "head"))
            pairs = [
                100
                * (
                    1
                    - rows["head", p][width, kind]["median_ms"]
                    / rows["base", p][width, kind]["median_ms"]
                )
                for p in (1, 2)
            ]
            spreads = {
                arm: 100
                * abs(
                    rows[arm, 1][width, kind]["median_ms"]
                    - rows[arm, 2][width, kind]["median_ms"]
                )
                / mean(arm, width, kind)
                for arm in ("base", "head")
            }
            summary[f"{width}:{kind}"] = dict(
                base_ms=base,
                head_ms=head,
                reduction_percent=100 * (1 - head / base),
                reduction_by_pair=pairs,
                process_spread_percent=spreads,
            )
            table.append(
                f"<tr><td>{width}</td><td>{kind}</td><td>{base:.3f}</td>"
                f"<td>{head:.3f}</td><td>{100 * (1 - head / base):.2f}%</td>"
                f"<td>{pairs[0]:.2f}%, {pairs[1]:.2f}%</td>"
                f"<td>{spreads['base']:.2f}%, {spreads['head']:.2f}%</td></tr>"
            )
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    colour = InkStyle().flux_color
    names = ["Scalar booking", "Carried read + booking", "Full read + booking"]
    for ax, width in zip(axes, (1, 16), strict=True):
        for arm, hue, offset in (("base", "#929292", 0.1), ("head", colour, -0.1)):
            for index, kind in enumerate(("booking", "carried", "full")):
                points = [rows[arm, p][width, kind]["median_ms"] / 1000 for p in (1, 2)]
                y = 2 - index + offset
                ax.plot(points, [y, y], color=hue, linewidth=3, zorder=2)
                ax.scatter(
                    [statistics.mean(points)],
                    [y],
                    color=hue,
                    s=70,
                    label=arm if index == 0 and width == 1 else None,
                    zorder=4 if arm == "head" else 3,
                )
        ax.set_yticks([2, 1, 0], names if width == 1 else [])
        ax.set_xlabel(f"CPU batch wall [s], width {width}")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.45, 2.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.legend(loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.64, 0.98))
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.16, top=0.88, wspace=0.14)
    destination = (
        root
        / "docs/figures/sub-millisecond-forward-path/booking-scalar-tile/cpu-walls.svg"
    )
    save_checked(fig, destination)
    fig.savefig(run / "cpu-walls.png", dpi=100)
    plt.close(fig)
    suite = []
    suite_receipts = json.loads((validation / "suite-receipts.json").read_text())
    for key, receipt in suite_receipts.items():
        text = Path(receipt["log_path"]).read_text()
        summaries = [
            line
            for line in text.splitlines()
            if re.search(r"\d+ (passed|failed)", line)
        ]
        assert summaries and text.rstrip().endswith(f"EXIT={receipt['exit_status']}")
        suite.append(
            f"<tr><td>{escape(key)}</td><td>{escape(summaries[-1])}</td>"
            f"<td>{receipt['exit_status']}</td></tr>"
        )
    negative = (validation / "negative-control.log").read_text()
    assert negative.rstrip().endswith("EXIT=1")
    assert (
        "FAILED tests/test_booking_scalar_tile.py::"
        "test_all_booked_moments_match_independent_population"
        in negative
    )
    negative_summary = [
        line for line in negative.splitlines() if re.search(r"\d+ failed", line)
    ][-1]
    positive = suite_receipts["head:test_booking_scalar_tile.py"]
    assert positive["exit_status"] == 0
    base = receipts["base", 1]["revision"]
    head = receipts["head", 1]["revision"]
    instrument = receipts["head", 1]["instrument_sha256"]
    job = receipts["head", 1]["job"]
    result = summary["16:booking"]
    fragment = f"""<section id="booking-scalar-tile-result">
<h2>Wider scalar conic tiles</h2>
<p>The scalar tile grows from 32 to 64 cells, reducing the diverted bank's
113 conic cut cells from four buckets to two. The normal-form population
retains its independent eight-cell tile. The existing ceiling division already
handled both small and partial populations correctly and is retained.
The same-input CPU sweep measured 2815.838 ms at 32 cells, 3158.677 ms at 64,
3208.195 ms at 128 and 6184.424 ms at 256. Width 64 is the fastest wider
candidate; none of the wider candidates beats the narrow CPU baseline.
The initial 128-cell paired run independently confirmed the CPU regression.
This is a measured shortfall against the expected scalar speedup.</p>
<p>At width sixteen, scalar booking changes from {result["base_ms"]:.3f} to
{result["head_ms"]:.3f} ms ({abs(result["reduction_percent"]):.2f}%
{"lower" if result["reduction_percent"] >= 0 else "higher"}).
The two fresh process pairs and their spreads are preserved below. The booking
program times current.book directly on precomputed physical flux samples;
carried and full programs include the complete native read and booking.
All operands enter the timed programs at runtime. Each row uses one synchronized
warm-up and three synchronized measured calls, excluding compilation.</p>
<figure><img
src="/nova/figures/sub-millisecond-forward-path/booking-scalar-tile/cpu-walls.svg"
alt="Paired scalar booking and complete read walls at widths one and sixteen">
<figcaption>CPU batch walls, widths one and sixteen in separate panels.
Points show means of two fresh-process medians; segments join those medians.
Base is grey and the wider scalar tile uses the established booking colour.
The panels have separate time scales. These CPU measurements make no H200
latency claim. The checked 1400 by 720 render was inspected before delivery.
</figcaption></figure>
<table><thead><tr><th>Width</th><th>Program</th><th>Base ms</th><th>Head ms</th>
<th>Reduction</th><th>Reductions by pair</th><th>Base/head process spread</th>
</tr></thead>
<tbody>{"".join(table)}</tbody></table>
<h3>Parity and deliberate refusal</h3>
<p>The unchanged 32-cell booking and an independent unbucketed placement oracle
supply comparisons for repeated physical cut-cell inventories from both limited
and diverted reads, with 17, 47, 67 and 137 conic cells. The tests compare cell current,
radial and vertical moments, raw current and normalization at 1e-12 relative
(with a 1e-14 absolute floor), including prescribed total-current normalization.
The repeated inventory isolates scheduling from mesh resolution; it is not a
new physical mesh or a convergence study. Separate checks exercise 0, 17, 67,
128, 137 and 257 active cells in a 270-cell array. A test pins the selected
64-cell width and the independent normal width. Before changing the source
the original 128-cell sentinel failed with
<code>assert 32 == 128</code>, reproducing the recorded narrow-tile condition.</p>
<p>Trial and converged fresh residuals on both analytic fixtures agree with the
32-cell path at the existing 1e-12 absolute comparison floor; the converged
fixed-exterior, unit-density fixture has residual at most 1e-10. This is a
booking-regression check, not a new nonlinear bank-solve certificate.</p>
<p>The scratch mutation replaces only the scalar population ceiling division by
floor division, dropping its last partial tile. Its parity run reports
<code>{escape(negative_summary)}</code> and <code>EXIT=1</code>.
The unchanged head parity tests pass. The mutated source and complete refusal log
remain in the run directory.</p>
<h3>Paired validation and provenance</h3>
<p>The paired validation uses one Sirius job with seven CPUs, one fresh process per
file and arm. The new scalar-tile file runs only at head. The archived-moment
pin fails identically at base and head (absolute 2.83388764e-15, relative
4.59452653e-12); support-read tests pass at both arms. Added failures are zero.
The archived comparison and its tolerance remain unchanged.</p>
<table><thead><tr><th>Arm and file</th><th>Pytest summary</th><th>Exit</th></tr></thead>
<tbody>{"".join(suite)}</tbody></table>
<p>Base revision <code>{base}</code>; executable head <code>{head}</code>.
CPU measurement job <code>{job}</code>, all_debug, seven CPUs, two fresh processes
per arm on the same node. Committed measurement instrument SHA-256
<code>{instrument}</code>. Every receipt records the imported current module,
working directory, source revision, instrument hash, backend and job identity.
The width-sweep receipt carries instrument SHA-256
<code>{sweep["instrument_sha256"]}</code>.
Logs and JSON receipts: <code>{escape(str(run))}</code>.</p>
</section>"""
    path = (
        root
        / "docs/evidence/fragments/sub-millisecond-forward-path"
        / "booking-scalar-tile.html"
    )
    path.write_text(fragment)
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
