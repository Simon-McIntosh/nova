"""Compose the research page and the evidence fragment from the census results."""

from __future__ import annotations

import argparse
import html
import json
import pathlib


def fmt(value, nd=4):
    if value is None:
        return "&ndash;"
    return f"{value:.{nd}g}"


def rebuild_table(rows):
    out = [
        "<table><thead><tr><th>record</th><th>polarity</th>"
        "<th>axis &psi; (record)</th><th>axis &psi; (spline)</th>"
        "<th>|&Delta;&psi;|</th><th>|&nabla;&psi;| at axis</th></tr></thead><tbody>"
    ]
    for row in rows:
        rb = row["rebuild"]
        out.append(
            f"<tr><td>{html.escape(row['case'])}</td><td>{row['polarity']}</td>"
            f"<td>{fmt(rb['axis_psi_reported'], 8)}</td>"
            f"<td>{fmt(rb['axis_psi_spline'], 8)}</td>"
            f"<td>{fmt(rb['axis_psi_abs_diff'], 3)}</td>"
            f"<td>{fmt(rb['grad_norm_at_axis'], 3)}</td></tr>"
        )
    out.append("</tbody></table>"
    )
    return "\n".join(out)


def code_points_table(row):
    order = ["efit_axis", "nova_a_axis", "nova_b_axis"]
    names = {"efit_axis": "imas-efit axis", "nova_a_axis": "adapter A axis",
             "nova_b_axis": "adapter B axis"}
    out = []
    for key in order:
        cp = row["code_points"].get(key)
        if cp is None:
            continue
        ok = "yes" if cp["agrees_with_polarity"] else "NO"
        out.append(
            f"<tr><td>{names[key]}</td><td>{cp['kind']}</td><td>{ok}</td>"
            f"<td>{fmt(cp['r'],5)}, {fmt(cp['z'],5)}</td>"
            f"<td>{fmt(cp['psi'],6)}</td>"
            f"<td>{fmt(cp['eig'][0],4)}, {fmt(cp['eig'][1],4)}</td>"
            f"<td>{fmt(cp['grad_norm'],3)}</td></tr>"
        )
    return "\n".join(out)


def census_table(row):
    out = ["<table><thead><tr><th>r</th><th>z</th><th>&psi;</th><th>type</th>"
           "<th>in wall</th><th>&lambda;&#8321;</th><th>&lambda;&#8322;</th>"
           "<th>efit side</th><th>nova A</th><th>nova B</th></tr></thead><tbody>"]
    points = row["census"]
    in_wall = [p for p in points if p["inside_wall"]]
    outside = [p for p in points if not p["inside_wall"]]
    # every in-wall point is listed; the out-of-wall set is listed whole up to a
    # cap, so a record whose spline carries the ripple of an oscillatory field
    # does not bury the reader.  The complete census is the results receipt.
    capped = outside[:200]
    for point in in_wall + capped:
        out.append(
            f"<tr><td>{fmt(point['r'],5)}</td><td>{fmt(point['z'],5)}</td>"
            f"<td>{fmt(point['psi'],6)}</td><td>{point['kind']}</td>"
            f"<td>{'yes' if point['inside_wall'] else 'no'}</td>"
            f"<td>{fmt(point['eig'][0],4)}</td><td>{fmt(point['eig'][1],4)}</td>"
            f"<td>{point['side_efit_boundary']}</td>"
            f"<td>{point['side_nova_a_boundary'] or '&ndash;'}</td>"
            f"<td>{point['side_nova_b_boundary'] or '&ndash;'}</td></tr>"
        )
    if len(outside) > len(capped):
        out.append(
            f"<tr><td colspan='10'>and {len(outside) - len(capped)} further out-of-wall "
            f"census points not listed; the complete census is the results receipt</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def record_tables(rows):
    blocks = []
    for row in rows:
        efit = row["efit"]
        a = row["adapter_a"]
        b = row["adapter_b"]
        ext = row["axis_extremum"]
        a_note = f" &mdash; {html.escape(a['nova_note'])}" if a and a.get("nova_note") else ""
        a_line = (
            f"<li>adapter A saddle &psi; = {fmt(a['nova_saddle_psi'],6)} at "
            f"({fmt(a['nova_saddle_r'],5)}, {fmt(a['nova_saddle_z'],5)}); "
            f"nova boundary &psi; = {fmt(a['nova_boundary_psi'],6)}; "
            f"label {a['nova_label']}</li>"
            if a else "<li>adapter A: no row for this record</li>"
        )
        if a and a["nova_axis_r"] is None:
            a_line = (
                f"<li>adapter A: nova admits no axis or saddle for this record; "
                f"label {a['nova_label']}; boundary &psi; = "
                f"{fmt(a['nova_boundary_psi'],6)}{a_note}</li>"
            )
        b_line = (
            f"<li>adapter B saddle &psi; = {fmt(b['nova_saddle_level'],6)} at "
            f"({fmt(b['nova_saddle_position'][0],5)}, {fmt(b['nova_saddle_position'][1],5)}); "
            f"nova boundary &psi; = {fmt(b['nova_boundary_level'],6)}; label {b['nova_label']}</li>"
            if b and b["nova_saddle_position"]
            else "<li>adapter B: no saddle for this record</li>"
        )
        ext_line = (
            f"{ext['kind']} at ({fmt(ext['r'],5)}, {fmt(ext['z'],5)}), &psi; = {fmt(ext['psi'],6)}"
            if ext else "none"
        )
        blocks.append(
            f"<section class='record' id='efit-axis-adjudication-{row['case']}'>"
            f"<h3>{html.escape(row['case'])} "
            f"<span class='muted'>{row['machine']}, slice {row['slice']}, "
            f"iteration {row['iteration']}, verdict {row['verdict']} "
            f"({row['efit_label']})</span></h3>"
            f"<p class='muted'>source: {html.escape(row['source_note'])} "
            f"&mdash; <code>{html.escape(row['path'])}</code></p>"
            f"<div class='grid2'>"
            f"<div><h4>code axes</h4><table><thead><tr><th>source</th><th>Hessian type</th>"
            f"<th>polarity ok</th><th>r, z</th><th>&psi;</th>"
            f"<th>&lambda;&#8321;, &lambda;&#8322;</th>"
            f"<th>|&nabla;&psi;|</th></tr></thead><tbody>{code_points_table(row)}</tbody></table></div>"
            f"<div><h4>code saddles / boundaries</h4><ul>"
            f"<li>imas-efit saddle_refined &psi; = {fmt(efit['saddle_refined'],6)} "
            f"(source node {efit['saddle_source_node']}); boundary level &psi; = "
            f"{fmt(efit['boundary_level'],6)}; label {efit['label']}</li>"
            f"{a_line}{b_line}</ul>"
            f"<p>in-wall extremum of &psi; in the polarity direction: {ext_line}</p>"
            f"<p>{row['n_census']} census points, {row['n_in_wall']} inside the wall.</p>"
            f"</div></div>"
            f"<h4>census</h4>{census_table(row)}"
            f"</section>"
        )
    return "\n".join(blocks)


def summary_table(rows):
    out = ["<table><thead><tr><th>record</th><th>disagreement</th>"
           "<th>distance [m]</th><th>&psi; gap</th></tr></thead><tbody>"]
    for row in rows:
        efit = row["efit"]
        a = row["adapter_a"]
        if a is None:
            continue
        if a["nova_axis_r"] is None:
            continue
        dr = a["nova_axis_r"] - efit["axis_r"]
        dz = a["nova_axis_z"] - efit["axis_z"]
        dist = (dr * dr + dz * dz) ** 0.5
        gap = a["nova_axis_psi"] - efit["axis_psi"]
        if dist > 1e-3:
            out.append(
                f"<tr><td>{html.escape(row['case'])}</td>"
                f"<td>axis: imas-efit vs adapter A</td>"
                f"<td>{fmt(dist,4)}</td><td>{fmt(gap,4)}</td></tr>"
            )
        src = efit["saddle_source_position"]
        if src:
            ds = ((a["nova_saddle_r"] - src[0]) ** 2
                  + (a["nova_saddle_z"] - src[1]) ** 2) ** 0.5
            if ds > 1e-3:
                out.append(
                    f"<tr><td>{html.escape(row['case'])}</td>"
                    f"<td>saddle: imas-efit source node vs adapter A</td>"
                    f"<td>{fmt(ds,4)}</td>"
                    f"<td>{fmt(a['nova_saddle_psi'] - efit['saddle_refined'],4)}</td></tr>"
                )
    if len(out) == 1:
        out.append("<tr><td colspan='4'>no axis or saddle disagreement over 1 mm</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def figure_links(rows):
    machines = sorted({row["machine"] for row in rows})
    out = []
    for machine in machines:
        slug = machine.lower().replace("-", "")
        out.append(
            f"<figure><img src='/nova/figures/converged-forward-solve/"
            f"efit-axis-adjudication/efit-axis-{slug}.svg' "
            f"alt='{html.escape(machine)} panels' width='100%'>"
            f"<figcaption>{html.escape(machine)} records: &psi; line contours (grey), the wall "
            f"(dark), imas-efit's boundary contour (dashed blue), adapter A's boundary (solid "
            f"orange) and adapter B's (dotted green); imas-efit's axis (solid triangle) and "
            f"X-point (filled cross); adapter A's axis (hollow circle) and saddle (filled "
            f"circle); adapter B's axis (hollow square) and saddle (filled square); every census "
            f"stationary point as a hollow marker by type (triangle up maximum, triangle down "
            f"minimum, cross saddle).</figcaption></figure>"
        )
    return "\n".join(out)


def build_page(rows, commits):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Effect axis adjudication | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
<style>
table{{border-collapse:collapse;font-size:0.78rem;margin:0.5rem 0;width:100%}}
th,td{{border-bottom:1px solid #ddd;padding:2px 5px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
.record{{border-top:1px solid #eee;padding-top:0.6rem;margin-top:0.8rem}}
.muted{{color:#666;font-weight:normal;font-size:0.85em}}
figure{{margin:1rem 0}}
</style></head>
<body><main class="plan-doc">
<h1 id="efit-axis-adjudication-title">Effect axis adjudication</h1>
<p id="efit-axis-adjudication-summary">Evidence for the lead to adjudicate the imas-efit and nova
fixed-point disagreements on imas-efit's boundary-level trace records. Every record is rebuilt from
its own 65&times;65 grid with a scipy not-a-knot bicubic spline, checked at imas-efit's recorded axis,
and censused exhaustively for stationary points. Both codes' axes and saddles, both boundary
contours and every census point are drawn. No verdict on which code is right is stated here.</p>

<h2 id="efit-axis-adjudication-method">Method</h2>
<ul>
<li><strong>Field.</strong> The record's own spline: <code>scipy.interpolate.RectBivariateSpline</code>
of its 65&times;65 grid at <code>k=3, s=0</code>, evaluated through a coefficient-matrix form for the
census. Nothing here uses either nova field-swap adapter.</li>
<li><strong>Rebuild check.</strong> At imas-efit's recorded axis the spline value must equal the
record's axis &psi; and |&nabla;&psi;| must sit at rounding level; both are tabulated per record.</li>
<li><strong>Census.</strong> Newton on &nabla;&psi; = 0 from a 120&times;120 seed lattice over the grid
box, deduplicated at 10<sup>&minus;4</sup> m; points outside the grid box are dropped, where the
spline extrapolates. Each point's Hessian eigenvalues give its type. A point lying on one side of a
code's boundary contour is nearer that code's axis in &psi; than the boundary level.</li>
<li><strong>Adapter A axis rule.</strong> The nova axis comes from nova's own census: adapter A calls
<code>nova.equilibrium.topology.read</code> on the spline field (<code>crosscheck.py</code>), so its
axis is nova's topology read, unanchored to the record.</li>
<li><strong>Adapter B axis rule.</strong> The record axis is a seed only (<code>adapter.py</code>):
adapter B seeds a bracket census with <code>(record.axis_r, record.axis_z)</code>, polishes it, and
picks the stationary point nearest the record axis, reporting <code>reading.axis_flux</code>. Its
exact match with imas-efit's axis level is therefore by construction. Adapter B records no axis
<em>position</em>, so its axis is identified among the census points as the one whose &psi; equals its
recorded <code>axis_level</code>; that match is stated in each panel's table.</li>
<li><strong>Absent fixed points.</strong> Adapter A reports <code>NaN</code> geometry where nova admits no
stationary point; those records carry no adapter A mark, and their geometry columns read
&ndash;. Every one of them carries adapter A's own note in the per-record table.</li>
</ul>

<h2 id="efit-axis-adjudication-rebuild">(1) Rebuild check</h2>
{rebuild_table(rows)}

<h2 id="efit-axis-adjudication-figures">(3) Poloidal panels</h2>
{figure_links(rows)}

<h2 id="efit-axis-adjudication-summary">(4) Disagreement summary</h2>
{summary_table(rows)}

<h2 id="efit-axis-adjudication-records">(4) Per-record tables and (2) census</h2>
{record_tables(rows)}
</main></body></html>"""


def build_fragment(rows, commits, verdict):
    n = len(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="docs-project" content="nova">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="efit-axis-adjudication">
<meta name="plan-title" content="Effect axis adjudication">
<meta name="plan-summary" content="Census of every imas-efit trace record's own spline, both codes' axes and saddles, for the lead to adjudicate the fixed-point disagreements">
<meta name="plan-evidence-for" content="converged-forward-solve">
<meta name="plan-verifies" content="converged-forward-solve#s2">
<meta name="plan-commits" content="{commits}">
<meta name="plan-recorded-at" content="2026-10-18">
<meta name="plan-verdict" content="{verdict}">
<title>Effect axis adjudication | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body><main class="plan-doc">
<h1 id="efit-axis-adjudication-rec-title">Effect axis adjudication</h1>
<p id="efit-axis-adjudication-rec-summary">{n} imas-efit boundary-level trace records each rebuilt
from its own 65&times;65 grid with a scipy not-a-knot bicubic spline and censused exhaustively for
stationary points; both codes' axes, saddles and boundary contours are drawn beside every census
point. The full record is in
<a href="/nova/research/efit-axis-adjudication.html">Effect axis adjudication</a>.</p>
<div id="efit-axis-adjudication-rec-figures">
{figure_links(rows)}
</div>
<p id="efit-axis-adjudication-rec-rules">Adapter A's nova axis comes from nova's own topology read on
the spline field; adapter B seeds its bracket census with the record axis and picks the nearest
polished stationary point, so its axis level matches imas-efit's by construction.</p>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--page", required=True)
    parser.add_argument("--fragment", required=True)
    parser.add_argument("--commits", default="")
    parser.add_argument("--verdict", default="partial")
    args = parser.parse_args()

    rows = json.loads(pathlib.Path(args.results).read_text())
    pathlib.Path(args.page).write_text(build_page(rows, args.commits))
    pathlib.Path(args.fragment).write_text(build_fragment(rows, args.commits, args.verdict))
    print("wrote", args.page)
    print("wrote", args.fragment)


if __name__ == "__main__":
    main()