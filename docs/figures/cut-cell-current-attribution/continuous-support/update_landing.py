"""Publish persisted row receipts with explicit numerical acceptance."""
# ruff: noqa: E501 -- Captions and persisted HTML retain their literal text.

import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
WORKTREE = ROOT.parents[3]
PREFIX = "/nova/figures/cut-cell-current-attribution/continuous-support/"
BEFORE = {
    110: (0.05726747947821424, 0.1887859512, 0.001074741678),
    300: (0.05393078396762727, 0.1965519077, 0.0007424374321),
}


def link(path, label):
    return (
        '<a href="'
        + PREFIX
        + str(path.relative_to(ROOT))
        + '">'
        + html.escape(label)
        + "</a>"
    )


rows = []
figures = []
for requested in (110, 300):
    path = ROOT / "rows-titan-adjoint/record" / f"cells-{requested}/receipt.json"
    if not path.exists() or "terminal" not in json.loads(path.read_text()):
        candidate = ROOT / "rows-titan/record" / f"cells-{requested}/receipt.json"
        if (
            requested == 300
            and candidate.exists()
            and "terminal" in json.loads(candidate.read_text())
        ):
            path = candidate
    if not path.exists():
        rows.append({"requested_cells": requested, "completed": False})
        continue
    data = json.loads(path.read_text())
    if "terminal" not in data:
        rows.append(
            {"requested_cells": requested, "completed": False, "receipt": str(path)}
        )
        continue
    terminal = data["terminal"]
    last = data["trips"][-1]
    share = data["outside_centroid_fraction"]
    pitch = data["characteristic_pitch_m"]
    entry = {
        "requested_cells": requested,
        "realised_cells": data["realised_cells"],
        "completed": True,
        "receipt": str(path.relative_to(ROOT)),
        "terminal_residual": terminal["residual"],
        "converged": terminal["converged"],
        "trips": terminal["trips"],
        "termination": terminal["termination"],
        "axis_error_m": last["axis_error_m"],
        "characteristic_pitch_m": pitch,
        "outside_centroid_fraction": share,
        "analytic_one_map_residual": data["map_checks"]["analytic_pinned"][
            "relative_sup"
        ],
        "quadrature_sha256": data.get("quadrature_sha256"),
        "timing": data.get("timing"),
        "raw_receipt": data["raw_receipt"],
        "raw_sha256": data["raw_sha256"],
        "residual_pass": terminal["residual"] <= 1e-12,
        "position_pass": last["axis_error_m"] <= pitch,
        "outside_fraction_pass": share <= 0.049,
    }
    rows.append(entry)
    panels = path.parents[2] / "panels"
    for trip in data["trips"]:
        stem = panels / f"cells-{requested}-trip-{trip['trip']}"
        png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
        if png.exists() and svg.exists():
            caption = f"{data['realised_cells']} cells, trip {trip['trip']}: residual {trip['residual']:.12g}, converged {str(trip['converged']).lower()}; analytic blue and solved ochre on shared levels, both axis markers, no admitted saddles, and the wall as its unit collection."
            figures.append(
                '<figure><img src="'
                + PREFIX
                + str(png.relative_to(ROOT))
                + '" alt="'
                + html.escape(caption)
                + '"><figcaption>'
                + html.escape(caption)
                + " "
                + link(svg, "SVG")
                + "</figcaption></figure>"
            )

paired = (
    json.loads((ROOT / "paired-suite.json").read_text())
    if (ROOT / "paired-suite.json").exists()
    else []
)
for result in paired:
    log = Path(result["log_path"]).read_text()
    result["failure_ids"] = re.findall(r"^FAILED\s+(\S+)", log, re.M)
    if result["exit_status"] and not result["failure_ids"]:
        result["failure_ids"] = [
            f"process:{result['target']}:exit-{result['exit_status']}"
        ]
    result["passed_count"] = (
        int(re.findall(r"(\d+) passed", log)[-1])
        if re.findall(r"(\d+) passed", log)
        else 0
    )
complete_modules = len(paired) == 12
baseline_failures = {
    failure
    for result in paired
    if result["variant"] == "baseline"
    for failure in result["failure_ids"]
}
after_failures = {
    failure
    for result in paired
    if result["variant"] == "after"
    for failure in result["failure_ids"]
}
added_failures = sorted(after_failures - baseline_failures)
acceptance = {
    "rows": rows,
    "tests": paired,
    "complete_module_delta": complete_modules,
    "added_failures": added_failures,
    "tangent_rung_accepted": True,
    "delivery_gate_complete": complete_modules and not added_failures,
    "coordinator_acceptance": "Retain the improved tangent rung; cold-seed convergence is not accepted. The outside-centroid share is accepted as no material rise.",
    "accepted": complete_modules
    and not added_failures
    and all(
        row.get("completed")
        and row.get("residual_pass")
        and row.get("position_pass")
        and row.get("outside_fraction_pass")
        for row in rows
    ),
}
(ROOT / "acceptance.json").write_text(json.dumps(acceptance, indent=2))

table = "<table><thead><tr><th>Cells</th><th>Before / after residual</th><th>Trips and termination</th><th>Before / after axis error (m)</th><th>Before / after analytic one-map residual</th></tr></thead><tbody>"
for row in rows:
    requested = row["requested_cells"]
    before = BEFORE[requested]
    if not row.get("completed"):
        table += f'<tr><td>{requested} requested</td><td colspan="4">Terminal receipt pending; no acceptance inferred.</td></tr>'
        continue
    table += f"<tr><td>{row['realised_cells']}</td><td>{before[0]:.12g} / {row['terminal_residual']:.12g}</td><td>{row['trips']}; {row['termination']}; converged {str(row['converged']).lower()}</td><td>{before[1]:.9g} / {row['axis_error_m']:.9g}; pitch {row['characteristic_pitch_m']:.9g}</td><td>{before[2]:.12g} / {row['analytic_one_map_residual']:.12g}</td></tr>"
table += "</tbody></table>"
body = (
    """<h3 id="continuous-confined-moments">Moving confined moments: tangent repaired, convergence not accepted</h3>
<p><strong>Coordinator ruling: retain the tangent repair as a completed component; cold-seed convergence is not accepted.</strong>
Production commit <code>6a2a7fb90</code> and test commit <code>e6f8bddd8</code> integrate and verify each closure
over its side of the cell-local quadratic separatrix, using the existing clip
and boundary-moment reductions. Six local coefficient tangents exclude undefined
inactive-root cotangents before transposition. The current pin, sufficient-decrease
rule and sixteen-trip budget are unchanged. No driving followup or section is closed.</p>
<p>The defect reproduces at base <code>814c854d79</code>: JVP 0.0 Wb, identity
control error 0.0, and a 1e-8 step removes 847.431878816 A from cell 66 and jumps
the map by 0.0241941949241 Wb. The repaired JVP is 6.38333066894 Wb, with central
finite-difference relative errors 5.64147e-9 at 1e-6 and 3.70040e-7 at 1e-8.
The full-map transpose is finite; duality error is 4.94519e-16. The synthetic
cell pins current, first moments, their derivatives and zero wholly outside
current. The negative control restores pointwise fixed quadrature and returns
two continuity failures and one outside-cell pass, exit 1.</p>
"""
    + table
    + """
<p>The outside-centroid current share is 4.913399823 percent at the 135-cell
cold seed, against the recorded 4.914472474 percent. The literal 4.9 percent
ceiling is exceeded by 0.013399823 percentage points. The coordinator accepts
this as no material rise: it is 0.001072651 percentage points below the recorded
value. This is an explicit acceptance ruling, not a changed measurement or a
rounded literal pass. Whole-cell outside
exclusion remains exactly zero in the synthetic test. Both analytic one-map
errors improve, but remain nonzero.</p>
<p>The production merit/trust instrument at the recorded coarse stalled state
accepts the half-defect candidate: merit 0.0626216340 to 0.0621299455, predicted
0.0624428164, residual 0.0535951963. The full analytic-direction candidate lowers
actual merit to 9.62674026e-5 but is refused because the local model predicts
0.348071531. <strong>Named remainder: the local merit model refuses the analytic-direction
candidate despite its measured merit reduction.</strong> This defect requires a
separately scoped repair and was not attempted here. These are the unmodified
production ladder and trust decisions,
not a replacement acceptance rule. Local tangent correctness has not established
global cold-seed convergence.</p>
"""
)
body += (
    f"<p>Imported-module gate: {len(paired)} of 12 baseline/after module processes completed; complete delta {str(complete_modules).lower()}. "
    + link(ROOT / "acceptance.json", "current numerical and test acceptance receipt")
    + ". A canceled initial baseline run is excluded because its child interpreters bypassed the import pin; the paired wrapper pins those child imports too.</p>"
)
body += f"<p>The imported-module inventory is six modules, each run against baseline and candidate: twelve fresh processes. The two interrupted processes are the baseline and candidate arms of <code>tests/test_exact_clip_memory.py</code>. Their completion allocation is 1275237 on all_debug, 59 minutes, eight CPUs and 64 GiB, with TMPDIR=/tmp in both submit and payload, JAX_PLATFORMS=cpu and the root interpreter invoked directly. Complete delta: {str(complete_modules).lower()}; newly added failures: {len(added_failures)}. The earlier 25-minute timeout is retained as interrupted evidence, not a pass.</p>"
body += "<p>Compute: H200 solve job 1275180 is coordinator-owned and was left untouched; titan is the GPU fallback and every quoted GPU wall belongs to its P100. Each certificate row runs in its own process, within one allocation per measurement. The first CPU fallback was submitted while titan lacked the required four cores. Compiler-event wall and remaining execution-plus-host wall are separate receipt fields; the latter is not asserted to be pure device time.</p>"
for row in rows:
    if row.get("completed"):
        timing = row["timing"]
        measured = (
            f" Compiler events {timing['compiler_event_wall_seconds']:.3f} s; execution plus host {timing['execute_and_host_wall_seconds']:.3f} s; solve call {timing['solve_wall_seconds']:.3f} s; cache hit {str(timing['compilation_cache_hit']).lower()}."
            if timing
            else " Compile and execution wall were not separated for this preliminary row."
        )
        body += (
            "<p>"
            + link(
                ROOT / row["receipt"],
                f"{row['realised_cells']}-cell compact terminal receipt",
            )
            + "."
            + measured
            + " Full receipt (including fields, per-trip states and compiler events): <code>"
            + html.escape(row["raw_receipt"])
            + "</code>; SHA-256 <code>"
            + row["raw_sha256"]
            + "</code>.</p>"
        )
body += (
    "<p>All three full record trees are preserved under <code>/home/ITER/mcintos/.config/reckon/crew/runs/r-20260921T151458383790-cca-continuous-confined-moments-and-tangent/records/</code>, in <code>rows-all_debug/record</code>, <code>rows-titan/record</code> and <code>rows-titan-adjoint/record</code>. All 172 files in the three source trees were hash-verified before compaction. Earlier CPU and titan rows are superseded diagnostic evidence; the two rows under rows-titan-adjoint are authoritative. "
    + link(ROOT / "measurement-index.json", "archive and compact-receipt index")
    + ".</p>"
)
body += (
    "<p>"
    + link(ROOT / "adjoint-cpu.json", "directional and transpose checks")
    + "; "
    + link(
        ROOT / "negative-test_continuous_confined_moments.log",
        "declared negative-control log",
    )
    + "; "
    + link(ROOT / "merit-scores.json", "production merit and trust scores")
    + ".</p>"
)

plan = WORKTREE / "docs/plans/cut-cell-current-attribution.html"
evidence = WORKTREE / "docs/evidence/archive/cut-cell-current-attribution-landed.html"
for path, content, pattern in (
    (
        plan,
        body
        + '<p><a href="/nova/evidence/archive/cut-cell-current-attribution-landed.html#continuous-confined-moments">All per-trip panels and the cumulative evidence</a>. The coordinator owns closure and any follow-on scope.</p>\n\n',
        r'<h3 id="continuous-confined-moments">.*?(?=<h2 id="s-contract">)',
    ),
    (
        evidence,
        '<section id="continuous-confined-support-record">\n'
        + body
        + "<details><summary>Every captured production trip</summary>"
        + "".join(figures)
        + "</details></section>",
        r'<section id="continuous-confined-support-record">.*?</section>',
    ),
):
    original = path.read_text()
    revised, count = re.subn(pattern, lambda match: content, original, flags=re.S)
    assert count == 1
    assert re.findall(r"<meta\b[^>]*>", original) == re.findall(
        r"<meta\b[^>]*>", revised
    )
    path.write_text(revised)
print(
    json.dumps(
        {
            "rows": [
                {key: value for key, value in row.items() if key != "timing"}
                for row in rows
            ],
            "test_processes": len(paired),
            "panels": len(figures),
        },
        indent=2,
    )
)
