"""Render the vessel-only comparison from the polarity-aware read receipt."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from analyse import ADAPTER_A


REBUILD_GRADIENT_TOL = 1e-9


def require_rebuild(row):
    """Reject a recorded axis that is not stationary on the reconstructed field."""
    rb = row["rebuild"]
    if (
        rb.get("status") != "pass"
        or rb["axis_psi_abs_diff"] > 1e-10
        or rb["grad_norm_at_axis"] > REBUILD_GRADIENT_TOL
    ):
        raise AssertionError(
            f"{row['case']}: spline rebuild failed: "
            f"|delta psi|={rb['axis_psi_abs_diff']:.3e}, "
            f"|grad psi|={rb['grad_norm_at_axis']:.3e}"
        )


def comparison_allowed(row):
    try:
        require_rebuild(row)
    except AssertionError:
        return False
    return row["machine"] != "JT-60SA"


def fmt(value, digits=5):
    return "&ndash;" if value is None else f"{value:.{digits}g}"


def rebuild_table(rows):
    cells = []
    for row in rows:
        rb = row["rebuild"]
        try:
            require_rebuild(row)
            verdict = "pass"
        except AssertionError:
            verdict = "<strong>rebuild failed</strong>"
        cells.append(
            f"<tr><td>{html.escape(row['case'])}</td>"
            f"<td>{fmt(rb['axis_psi_reported'], 8)}</td>"
            f"<td>{fmt(rb['axis_psi_spline'], 8)}</td>"
            f"<td>{fmt(rb['grad_norm_at_axis'], 4)}</td><td>{verdict}</td></tr>"
        )
    return (
        "<table><thead><tr><th>record</th><th>axis ψ (trace)</th>"
        "<th>axis ψ (spline)</th><th>|∇ψ| at axis</th><th>check</th>"
        "</tr></thead><tbody>" + "".join(cells) + "</tbody></table>"
    )


def summary_table(rows):
    cells = []
    for row in rows:
        efit, nova = row["efit"], row["adapter_a"]
        axis = row["code_points"]["nova_a_axis"]
        nova_position = f"{fmt(axis['r'])}, {fmt(axis['z'])}" if axis else "&ndash;"
        cells.append(
            f"<tr><td>{html.escape(row['case'])}</td>"
            f"<td>{html.escape(efit['label'])}</td>"
            f"<td>{html.escape(nova['nova_label'])}</td>"
            f"<td>{fmt(efit['axis_r'])}, {fmt(efit['axis_z'])}</td>"
            f"<td>{nova_position}"
        )
        cells[-1] += (
            f"</td><td>{fmt(efit['boundary_level'], 7)}</td>"
            f"<td>{fmt(nova['nova_boundary_psi'], 7)}</td>"
            f"<td>{row['n_in_wall']}/{row['n_census']}</td></tr>"
        )
    return (
        "<table><thead><tr><th>record</th><th>imas-efit label</th>"
        "<th>nova label</th><th>imas-efit axis [m]</th>"
        "<th>nova axis [m]</th><th>imas-efit boundary ψ</th>"
        "<th>nova boundary ψ</th><th>census in vessel / total</th>"
        "</tr></thead><tbody>" + "".join(cells) + "</tbody></table>"
    )


def saddle_table(rows):
    cells = []
    for row in rows:
        efit, nova = row["efit"], row["adapter_a"]
        cells.append(
            f"<tr><td>{html.escape(row['case'])}</td>"
            f"<td>{fmt(efit['saddle_refined'], 7)}</td>"
            f"<td>not stored</td>"
            f"<td>{fmt(nova['nova_saddle_psi'], 7)}</td>"
            f"<td>{fmt(nova['nova_saddle_r'])}, "
            f"{fmt(nova['nova_saddle_z'])}</td></tr>"
        )
    return (
        "<table><thead><tr><th>record</th><th>imas-efit saddle ψ</th>"
        "<th>imas-efit refined position</th><th>nova saddle ψ</th>"
        "<th>nova saddle [m]</th></tr></thead><tbody>"
        + "".join(cells)
        + "</tbody></table>"
    )


def figure_chunks(machine, members):
    """Keep each canvas legible and below the evidence file-size ceiling."""
    count = len(members) if machine == "synthetic" else math.ceil(len(members) / 3)
    width, extra = divmod(len(members), count)
    start = 0
    for index in range(count):
        end = start + width + (index < extra)
        yield index, start, members[start:end]
        start = end


def figure_links(rows):
    figures = []
    for machine in sorted({row["machine"] for row in rows}):
        members = [row for row in rows if row["machine"] == machine]
        slug = machine.lower().replace("-", "")
        for index, start, subset in figure_chunks(machine, members):
            failures = sum(not comparison_allowed(row) for row in subset)
            case_key = "".join(
                f"<li>{number}: {html.escape(row['case'])}</li>"
                for number, row in enumerate(subset, start=start + 1)
            )
            suffix = "" if index == 0 else f"-{index + 1}"
            figures.append(
                f"<figure><img src='/nova/figures/converged-forward-solve/"
                f"efit-axis-adjudication/efit-axis-{slug}{suffix}.svg' "
                f"alt='{html.escape(machine)} vessel-masked flux panels' width='100%'>"
                f"<figcaption>{html.escape(machine)}: {len(subset)} trace records, "
                f"{failures} with a failed spline rebuild. Each panel uses "
                f"line contours on the trace's stated ψ levels inside the vessel, "
                f"in-vessel stationary "
                f"points, the wall, the imas-efit axis, and the polarity-aware nova "
                f"axis and admitted saddle where present. The trace stores a refined "
                f"saddle level but no refined coordinate, so no imas-efit saddle "
                f"position is marked. Failed rebuilds carry no code comparison."
                f"</figcaption><ol>{case_key}</ol></figure>"
            )
    return "\n".join(figures)


def build_page(rows, commits=""):
    stress = [row for row in rows if row["machine"] == "JT-60SA"]
    compared = [row for row in rows if row["machine"] != "JT-60SA"]
    qualified = [row for row in compared if comparison_allowed(row)]
    rebuilds = len(qualified)
    excluded = [row for row in compared if not comparison_allowed(row)]
    outside = sum(row["n_census"] - row["n_in_wall"] for row in compared)
    refused = sum(row["adapter_a"]["nova_axis_r"] is None for row in qualified)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Effect axis adjudication | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
<style>table{{border-collapse:collapse;font-size:.8rem;width:100%}}
th,td{{border-bottom:1px solid #ddd;padding:3px 6px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} figure{{margin:1.5rem 0}}</style>
</head><body><main class="plan-doc">
<h1 id="efit-axis-adjudication-title">Effect axis adjudication</h1>
<p>The independent spline rebuild passes on {rebuilds} of {len(compared)}
non-stress records; {len(excluded)} recorded axes fail its gradient check.
The polarity-aware nova read supplies every nova mark and decision in the
comparison. {refused} records have no admitted nova axis. The census found
{outside} points outside the vessel in these records; they remain in the
receipt's total counts but appear in no panel or in-vessel census table.</p>
<p>The nova decision fields come from the fixed adapter receipt at
<code>{html.escape(str(ADAPTER_A))}</code>. The spline census is retained from
the independent trace rebuild; vessel membership is recomputed from the
indexed wall loops before painting.</p>
<p>JT-60SA's {len(stress)} records are wrong-basin iterates, not equilibria;
the third wall loop is a synthesised extended wall. They contribute to no
agreement figure or comparison table and have no poloidal panel here.</p>
<h2 id="efit-axis-adjudication-method">Method and limits</h2>
<p>Each trace's 65×65 field is rebuilt as a not-a-knot bicubic spline.
The rebuild check requires |Δψ| ≤ 1e−10 and |∇ψ| ≤ 1e−9 at the
recorded axis. Census points are polished on that spline. The panels mask
the field to the vessel loop and remove obstacle interiors. The trace stores
the imas-efit saddle's refined level, but only a grid-node source for its
position; the source is not plotted or used for a distance comparison.
The nova axis and admitted saddle are read with the trace polarity.
The plotted flux levels are shared across the code boundary contours
within each panel. This compares decision rules on a supplied field,
not two independently reconstructed equilibria.</p>
<h2 id="efit-axis-adjudication-rebuild">Spline rebuild</h2>
{rebuild_table(compared)}
<h2 id="efit-axis-adjudication-figures">Poloidal panels</h2>
{figure_links(compared)}
<h2 id="efit-axis-adjudication-summary">Code decisions</h2>
{summary_table(qualified)}
<h2 id="efit-axis-adjudication-saddles">Admitted saddle evidence</h2>
{saddle_table(qualified)}
<p>The imas-efit coordinate is not stored. A difference in the listed
levels is a flux-level difference, not a measured saddle-position gap.</p>
</main></body></html>"""


def build_fragment(rows, commits, verdict):
    compared = [row for row in rows if row["machine"] != "JT-60SA"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="efit-axis-adjudication">
<meta name="plan-evidence-for" content="converged-forward-solve">
<meta name="plan-verifies" content="converged-forward-solve#s2">
<meta name="plan-commits" content="{html.escape(commits)}">
<meta name="plan-verdict" content="{html.escape(verdict)}">
<title>Effect axis adjudication | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css"></head>
<body><main class="plan-doc"><h1 id="efit-axis-adjudication-rec-title">
Effect axis adjudication</h1><p id="efit-axis-adjudication-rec-summary">
{len(compared)} non-stress traces are shown with vessel-masked line contours
and the polarity-aware nova read. The imas-efit saddle position is absent
from its trace, so no source grid-node proxy is marked. JT-60SA's
{len(rows) - len(compared)} wrong-basin iterates are excluded from comparisons.
See <a href="/nova/research/efit-axis-adjudication.html">the full record</a>.</p>
{figure_links(compared)}</main></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--commits", default="")
    parser.add_argument("--verdict", default="partial")
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    args.page.write_text(build_page(rows, args.commits))
    args.fragment.write_text(build_fragment(rows, args.commits, args.verdict))
    print("wrote", args.page, args.fragment)


if __name__ == "__main__":
    main()
