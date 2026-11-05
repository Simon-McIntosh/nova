"""Validate paired receipts and emit one per-lane evidence table."""

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.bank_solve_cycle.report import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    base, head = (
        json.loads((args.run / f"{args.label}-{arm}.json").read_text())
        for arm in ("base", "head")
    )
    precision = compare(base, head)
    (args.run / "precision.json").write_text(json.dumps(precision, indent=2))
    before, after = base["rows"][-1], head["rows"][-1]
    assert before["lanes"] == after["lanes"] == 16
    for receipt in (base, head):
        for key in ("steps", "reads", "residual"):
            assert receipt["rows"][0][key] == receipt["rows"][1][key]
    gaps = {}
    for key, unit in (
        ("flux", before["span"]),
        ("axis", before["minor_radius"]),
        ("saddle", before["minor_radius"]),
        ("boundary_flux", before["span"]),
    ):
        values = np.asarray(after[key]) - np.asarray(before[key])
        if key in ("axis", "saddle"):
            values = np.linalg.norm(values, axis=-1)
        if key == "flux":
            values = np.max(np.abs(values), axis=-1)
        gaps[key] = np.abs(values) / unit
    rows = []
    for lane in range(16):
        cells = [str(lane + 1)]
        for key in ("steps", "reads"):
            cells.append(f"{before[key][lane]} → {after[key][lane]}")
        cells.append(f"{after['residual'][lane]:.3e}")
        cells.extend(f"{gaps[key][lane]:.3e}" for key in gaps)
        rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
    prefix = args.fragment.stem
    incumbent = json.loads((args.run / "incumbent-head.json").read_text())["rows"][0]
    stalled = [
        f"lane {lane + 1}: {incumbent['steps'][lane]} updates, "
        f"residual {incumbent['residual'][lane]:.6e}"
        for lane, converged in enumerate(incumbent["converged"])
        if not converged
    ]
    assert stalled
    summary = {
        "base_per_slice_ms": before["per_slice_ms"],
        "head_per_slice_ms": after["per_slice_ms"],
        "base_steps": before["steps"],
        "head_steps": after["steps"],
        "head_reads": after["reads"],
        "precision": precision,
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="plan-slug" content="{prefix}">
<meta name="plan-title" content="Shared dyadic bank backtracking">
<meta name="plan-summary"
content="One search, complete read counters and paired precision">
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
<meta name="plan-evidence-for" content="sub-millisecond-forward-path">
<meta name="plan-verifies" content="sub-millisecond-forward-path#s3">
<title>Shared bank backtracking | nova</title></head><body><main class="plan-doc">
<h1 id="{prefix}-landing">Shared dyadic backtracking and complete read counts</h1>
<p>The cold Newton correction and the bank driver use one bounded dyadic search.
Every batch read counts for every lane, including lanes already converged or no
longer searching. The two-halving fixture observes four reads and requires
[4, 4, 4]; the pending-lane counter reports [4, 2, 1] and fails.</p>
<p>Keeping the incumbent when all trials are worse stalled the physical seed:
{escape("; ".join(stalled))}. This arm failed its convergence assertion with
EXIT=1 in <code>incumbent-head.log</code>. The bank therefore uses the authorized
fallback: retain the smallest-residual valid trial among the eight dyadic
candidates when none improves. The docstring states this policy. Cold Newton
retains its incumbent, and tests cover both policies through the same search.</p>
<p>One H200, two cores and 48 GB; the existing cycle measurement is imported.
Base <code>{base["revision"]}</code>, head <code>{head["revision"]}</code>, each
from a git archive. Both executable calls after compilation are checked for
precision; the table gives the second call. Repeated converged wall per slice:
<strong>{before["per_slice_ms"]:.3f} → {after["per_slice_ms"]:.3f} ms</strong>.
Step counts vary between allocations, so this pair is not a general speed claim.
First-call walls: {base["rows"][0]["per_slice_ms"]:.3f} →
{head["rows"][0]["per_slice_ms"]:.3f} ms/slice. Base read counters exclude companion
reads and are shown as reported; head counts every actual batch evaluation.</p>
<table id="{prefix}-lanes"><thead><tr><th>Lane</th><th>Steps, base → head</th>
<th>Reads, reported base → head</th><th>Fresh residual</th><th>Flux / span</th>
<th>Axis / radius</th><th>X-point / radius</th><th>Boundary / span</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>Every lane must have fresh residual ≤ 1e-10, flux difference from base ≤ 1e-8
of span, axis and X-point difference ≤ 5e-4 of minor radius, and boundary flux
difference ≤ 1e-4 of span. The imported comparison also checks axis, X-point
and boundary flux against the analytic oracle. Across both calls the largest
flux difference is {max(r['flux'] for r in precision):.4e} of span; oracle axis,
X-point and boundary ratios are {max(r['oracle_axis'] for r in precision):.4e},
{max(r['oracle_saddle'] for r in precision):.4e} and
{max(r['oracle_boundary'] for r in precision):.4e}, respectively.</p>
<p id="{prefix}-cpu">Nine focused driver and cycle tests pass. The required
four-file baseline and head CPU arms and the scratch counter mutation are scheduled
on sirius; their results remain pending.</p>
<p>Raw receipts and logs: <code>{escape(str(args.run))}</code>;
<code>{args.label}-base.log</code>, <code>{args.label}-head.log</code>, and
<code>precision.json</code>. Each measurement log prints its revision, imported
module and working directory and ends with its exit status.</p>
</main></body></html>'''
    args.fragment.write_text(text)


if __name__ == "__main__":
    main()
