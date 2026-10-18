"""Compose the research page and the evidence fragment from the census results."""

from __future__ import annotations

import argparse
import html
import json
import pathlib
from itertools import combinations


REBUILD_GRADIENT_TOL = 1e-9


def require_rebuild(row):
    """Reject a recorded axis whose independent spline rebuild is not stationary."""
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
    """Allow code comparisons only for a spline-consistent recorded axis."""
    try:
        require_rebuild(row)
    except AssertionError:
        return False
    return True


def fmt(value, nd=4):
    if value is None:
        return "&ndash;"
    return f"{value:.{nd}g}"


def rebuild_table(rows):
    out = [
        "<table><thead><tr><th>record</th><th>polarity</th>"
        "<th>axis &psi; (record)</th><th>axis &psi; (spline)</th>"
        "<th>|&Delta;&psi;|</th><th>|&nabla;&psi;| at axis</th><th>check</th></tr></thead><tbody>"
    ]
    for row in rows:
        rb = row["rebuild"]
        try:
            require_rebuild(row)
            passed = True
        except AssertionError:
            passed = False
        out.append(
            f"<tr><td>{html.escape(row['case'])}</td><td>{row['polarity']}</td>"
            f"<td>{fmt(rb['axis_psi_reported'], 8)}</td>"
            f"<td>{fmt(rb['axis_psi_spline'], 8)}</td>"
            f"<td>{fmt(rb['axis_psi_abs_diff'], 3)}</td>"
            f"<td>{fmt(rb['grad_norm_at_axis'], 3)}</td>"
            f"<td>{'pass' if passed else '<strong>rebuild failed</strong>'}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def code_points_table(row):
    order = ["efit_axis", "nova_a_axis", "nova_b_axis"]
    names = {
        "efit_axis": "imas-efit axis",
        "nova_a_axis": "adapter A axis",
        "nova_b_axis": "adapter B axis",
    }
    out = []
    for key in order:
        cp = row["code_points"].get(key)
        if cp is None:
            continue
        ok = "yes" if cp["agrees_with_polarity"] else "NO"
        out.append(
            f"<tr><td>{names[key]}</td><td>{cp['kind']}</td><td>{ok}</td>"
            f"<td>{fmt(cp['r'], 5)}, {fmt(cp['z'], 5)}</td>"
            f"<td>{fmt(cp['psi'], 6)}</td>"
            f"<td>{fmt(cp['eig'][0], 4)}, {fmt(cp['eig'][1], 4)}</td>"
            f"<td>{fmt(cp['grad_norm'], 3)}</td></tr>"
        )
    return "\n".join(out)


def census_table(row):
    out = [
        "<table><thead><tr><th>r</th><th>z</th><th>&psi;</th><th>type</th>"
        "<th>in wall</th><th>&lambda;&#8321;</th><th>&lambda;&#8322;</th>"
        "<th>efit side</th><th>nova A</th><th>nova B</th></tr></thead><tbody>"
    ]
    points = row["census"]
    in_wall = [p for p in points if p["inside_wall"]]
    outside = [p for p in points if not p["inside_wall"]]
    allowed = comparison_allowed(row)
    for point in in_wall + outside:
        out.append(
            f"<tr><td>{fmt(point['r'], 5)}</td><td>{fmt(point['z'], 5)}</td>"
            f"<td>{fmt(point['psi'], 6)}</td><td>{point['kind']}</td>"
            f"<td>{'yes' if point['inside_wall'] else 'no'}</td>"
            f"<td>{fmt(point['eig'][0], 4)}</td><td>{fmt(point['eig'][1], 4)}</td>"
            f"<td>{point['side_efit_boundary']}</td>"
            f"<td>{point['side_nova_a_boundary'] or '&ndash;' if allowed else '&ndash;'}</td>"
            f"<td>{point['side_nova_b_boundary'] or '&ndash;' if allowed else '&ndash;'}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def record_tables(rows):
    blocks = []
    for row in rows:
        if not comparison_allowed(row):
            blocks.append(
                f"<section class='record' id='efit-axis-adjudication-{row['case']}'>"
                f"<h3>{html.escape(row['case'])}</h3>"
                f"<p><strong>rebuild failed</strong>: recorded-axis |&nabla;&psi;| = "
                f"{fmt(row['rebuild']['grad_norm_at_axis'], 5)} exceeds 1e-9. "
                f"Axis and saddle comparisons are excluded for this record.</p>"
                f"<p>{row['n_census']} spline census points, "
                f"{row['n_in_wall']} inside the wall.</p>"
                f"<h4>census of the stated spline</h4>{census_table(row)}</section>"
            )
            continue
        efit = row["efit"]
        a = row["adapter_a"]
        b = row["adapter_b"]
        ext = row["axis_extremum"]
        source = efit["saddle_source_position"]
        source_text = (
            f" at ({fmt(source[0], 5)}, {fmt(source[1], 5)})" if source else ""
        )
        a_note = (
            f" &mdash; {html.escape(a['nova_note'])}"
            if a and a.get("nova_note")
            else ""
        )
        a_line = (
            f"<li>adapter A saddle &psi; = {fmt(a['nova_saddle_psi'], 6)} at "
            f"({fmt(a['nova_saddle_r'], 5)}, {fmt(a['nova_saddle_z'], 5)}); "
            f"nova boundary &psi; = {fmt(a['nova_boundary_psi'], 6)}; "
            f"label {a['nova_label']}</li>"
            if a
            else "<li>adapter A: no row for this record</li>"
        )
        if a and a["nova_axis_r"] is None:
            a_line = (
                f"<li>adapter A: nova admits no axis or saddle for this record; "
                f"label {a['nova_label']}; boundary &psi; = "
                f"{fmt(a['nova_boundary_psi'], 6)}{a_note}</li>"
            )
        b_line = (
            f"<li>adapter B saddle &psi; = {fmt(b['nova_saddle_level'], 6)} at "
            f"({fmt(b['nova_saddle_position'][0], 5)}, {fmt(b['nova_saddle_position'][1], 5)}); "
            f"nova boundary &psi; = {fmt(b['nova_boundary_level'], 6)}; label {b['nova_label']}</li>"
            if b and b["nova_saddle_position"]
            else "<li>adapter B: no saddle for this record</li>"
        )
        ext_line = (
            f"{ext['kind']} at ({fmt(ext['r'], 5)}, {fmt(ext['z'], 5)}), &psi; = {fmt(ext['psi'], 6)}"
            if ext
            else "none"
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
            f"<li>imas-efit saddle_refined &psi; = {fmt(efit['saddle_refined'], 6)} "
            f"(source grid-node proxy {efit['saddle_source_node']}{source_text}; "
            f"the record has no refined coordinate); boundary level &psi; = "
            f"{fmt(efit['boundary_level'], 6)}; label {efit['label']}</li>"
            f"{a_line}{b_line}</ul>"
            f"<p>most extreme censused in-wall stationary point in the polarity direction: "
            f"{ext_line}</p>"
            f"<p>{row['n_census']} census points, {row['n_in_wall']} inside the wall.</p>"
            f"</div></div>"
            f"<h4>census</h4>{census_table(row)}"
            f"</section>"
        )
    return "\n".join(blocks)


def summary_table(rows):
    excluded = [row for row in rows if not comparison_allowed(row)]
    exclusions = "".join(
        f"{html.escape(row['case'])} (|&nabla;&psi;| = "
        f"{fmt(row['rebuild']['grad_norm_at_axis'], 5)}); "
        for row in excluded
    )
    out = [
        f"<p>Rebuild failed; axis and saddle comparisons excluded: {exclusions}</p>"
        "<p>imas-efit saddle distances use its source grid-node proxy, not a refined "
        "coordinate. Its &psi; gap uses the recorded refined saddle level.</p>",
        "<table><thead><tr><th>record</th><th>fixed point</th><th>pair</th>"
        "<th>distance [m]</th><th>&psi; gap</th></tr></thead><tbody>",
    ]
    for row in rows:
        if not comparison_allowed(row):
            continue
        efit = row["efit"]
        a = row["adapter_a"]
        b = row["adapter_b"]
        point = row["code_points"]["nova_b_axis"]
        axes = {
            "imas-efit": (efit["axis_r"], efit["axis_z"], efit["axis_psi"]),
            "adapter A": (a["nova_axis_r"], a["nova_axis_z"], a["nova_axis_psi"])
            if a and a["nova_axis_r"] is not None
            else None,
            "adapter B": (point["r"], point["z"], b["nova_axis_level"])
            if point and b
            else None,
        }
        source = efit["saddle_source_position"]
        saddles = {
            "imas-efit source grid-node proxy": (
                source[0],
                source[1],
                efit["saddle_refined"],
            )
            if source
            else None,
            "adapter A": (a["nova_saddle_r"], a["nova_saddle_z"], a["nova_saddle_psi"])
            if a and a["nova_saddle_r"] is not None
            else None,
            "adapter B": (*b["nova_saddle_position"], b["nova_saddle_level"])
            if b and b["nova_saddle_position"]
            else None,
        }
        for kind, points in (("axis", axes), ("saddle", saddles)):
            for (left_name, left), (right_name, right) in combinations(
                points.items(), 2
            ):
                if left is None or right is None:
                    distance = gap = None
                else:
                    distance = (
                        (right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2
                    ) ** 0.5
                    gap = right[2] - left[2]
                    if distance <= 1e-3 and abs(gap) <= 1e-6:
                        continue
                out.append(
                    f"<tr><td>{html.escape(row['case'])}</td>"
                    f"<td>{kind}</td><td>{left_name} vs {right_name}"
                    f"{' (one absent)' if distance is None else ''}</td>"
                    f"<td>{fmt(distance, 4)}</td><td>{fmt(gap, 4)}</td></tr>"
                )
    if len(out) == 2:
        out.append(
            "<tr><td colspan='5'>no axis or saddle disagreement over 1 mm or 1e-6 psi</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def figure_links(rows):
    machines = sorted({row["machine"] for row in rows})
    out = []
    for machine in machines:
        slug = machine.lower().replace("-", "")
        captions = []
        for row in rows:
            if row["machine"] != machine:
                continue
            levels = row["contour_levels"]
            contour_text = (
                f"grey &psi; levels start {fmt(levels[0], 8)}, "
                f"step {fmt(levels[1] - levels[0], 8)}, count {len(levels)}"
            )
            if not comparison_allowed(row):
                captions.append(
                    f"<li>{html.escape(row['case'])}: {contour_text}; "
                    f"<strong>rebuild failed</strong> at the recorded axis "
                    f"(|&nabla;&psi;| = {fmt(row['rebuild']['grad_norm_at_axis'], 5)}); "
                    f"axis and saddle comparisons excluded.</li>"
                )
                continue
            efit, a, b = row["efit"], row["adapter_a"], row["adapter_b"]
            captions.append(
                f"<li>{html.escape(row['case'])}: {contour_text}; imas-efit axis &psi; "
                f"{fmt(efit['axis_psi'], 6)}, boundary &psi; {fmt(efit['boundary_level'], 6)}, "
                f"label {html.escape(efit['label'])}; adapter A axis &psi; "
                f"{fmt(a['nova_axis_psi'], 6) if a else '&ndash;'}, boundary &psi; "
                f"{fmt(a['nova_boundary_psi'], 6) if a else '&ndash;'}, "
                f"label {html.escape(a['nova_label']) if a else '&ndash;'}; "
                f"adapter B axis &psi; {fmt(b['nova_axis_level'], 6) if b else '&ndash;'}, "
                f"boundary &psi; {fmt(b['nova_boundary_level'], 6) if b else '&ndash;'}, "
                f"label {html.escape(b['nova_label']) if b else '&ndash;'}. "
                f"The imas-efit X marker is its saddle source grid-node proxy.</li>"
            )
        has_comparisons = any(
            comparison_allowed(row) for row in rows if row["machine"] == machine
        )
        style_caption = (
            "Code boundary and fixed-point styles are identified in the figure legend. "
            "The imas-efit saddle cross is a source grid-node proxy."
            if has_comparisons
            else "Only the stated spline contours, wall and hollow census markers are drawn; "
            "code comparisons are excluded because the rebuild failed."
        )
        out.append(
            f"<figure><img src='/nova/figures/converged-forward-solve/"
            f"efit-axis-adjudication/efit-axis-{slug}.svg' "
            f"alt='{html.escape(machine)} panels' width='100%'>"
            f"<figcaption>{html.escape(machine)} records: {style_caption} Each panel's grey "
            f"&psi; levels are stated below it. Every census stationary point is marked "
            f"hollow by Hessian type (up triangle maximum, down triangle minimum, cross saddle)."
            f"</figcaption><ul>{''.join(captions)}</ul></figure>"
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
and censused for stationary points. The recorded-axis rebuild holds for the 24 measured records
and fails for both synthetic crossing inputs. The synthetic contours and census remain shown,
with axis and saddle comparisons excluded. No verdict on which code is right is stated here.</p>

<h2 id="efit-axis-adjudication-method">Method</h2>
<ul>
<li><strong>Field.</strong> The record's own spline: <code>scipy.interpolate.RectBivariateSpline</code>
of its 65&times;65 grid at <code>k=3, s=0</code>, evaluated through a coefficient-matrix form for the
census. Nothing here uses either nova field-swap adapter.</li>
<li><strong>Rebuild check.</strong> At imas-efit's recorded axis the spline value must equal the
record's axis &psi; within 10<sup>&minus;10</sup> and |&nabla;&psi;| must be below
10<sup>&minus;9</sup>. Both are tabulated per record with an explicit pass or rebuild failed.
This check holds for all 24 measured records. The two synthetic
records fail the gradient check at 2.433&times;10<sup>&minus;4</sup> and
2.638&times;10<sup>&minus;4</sup>; their contours and census therefore describe the stated scipy
spline, but do not validate that this spline is the field from which their recorded axes came.
Their axis and saddle comparisons are excluded.</li>
<li><strong>Census.</strong> Newton on &nabla;&psi; = 0 from a 120&times;120 seed lattice over the grid
box, deduplicated at 10<sup>&minus;4</sup> m; points outside the grid box are dropped, where the
spline extrapolates. Each point's Hessian eigenvalues give its type. The side columns state whether
its &psi; is above, below or on each code's boundary level. This flux-side classification does not
claim geometric containment in a particular contour component.</li>
<li><strong>Adapter A axis rule.</strong> The nova axis comes from nova's own census: adapter A calls
<code>nova.equilibrium.topology.read</code> on the spline field (<code>crosscheck.py</code>), so its
axis is nova's topology read, unanchored to the record.</li>
<li><strong>Adapter B axis rule.</strong> Adapter B seeds a bracket census at the record axis,
polishes the candidates, and keeps the nearest polished point to that recorded axis
(<code>scripts/prototypes/efit_field_swap/adapter.py:163&ndash;208</code>). Its level match is
record-seeded, not an independent axis selection or an identity forced by the code.
Adapter B records no axis
<em>position</em>, so its axis is identified among the census points as the one whose &psi; equals its
recorded <code>axis_level</code>; the level matching rule is implemented in
<code>scripts/prototypes/efit_axis_adjudication/analyse.py</code> and reported by this page generator
<code>scripts/prototypes/efit_axis_adjudication/page.py:376&ndash;384</code>. The match is stated in each
measured panel's table.</li>
<li><strong>imas-efit saddle position.</strong> The trace stores a refined saddle level and a source
grid node, not a refined saddle coordinate. Every plotted cross and tabulated saddle distance
uses the source grid node as a position proxy. The &psi; gaps use the stored refined level.</li>
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
from its own 65&times;65 grid with a scipy not-a-knot bicubic spline and censused for
stationary points. The rebuild check holds for 24 measured records and fails for the two synthetic
inputs: their recorded-axis gradient norms are 2.433e-4 and 2.638e-4 against a 1e-9 limit.
The synthetic panels show the stated spline but exclude axis and saddle comparisons.
The full record is in
<a href="/nova/research/efit-axis-adjudication.html">Effect axis adjudication</a>.</p>
<div id="efit-axis-adjudication-rec-figures">
{figure_links(rows)}
</div>
<p id="efit-axis-adjudication-rec-rules">Adapter A's nova axis comes from nova's own topology read on
the spline field. Adapter B seeds a bracket census at the record axis, polishes the candidates,
and keeps the nearest polished point; its match is record-seeded rather than independent.
imas-efit's plotted saddle source node is a grid-node position proxy, not a refined coordinate.</p>
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
    pathlib.Path(args.fragment).write_text(
        build_fragment(rows, args.commits, args.verdict)
    )
    print("wrote", args.page)
    print("wrote", args.fragment)


if __name__ == "__main__":
    main()
