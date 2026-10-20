"""Render a geometric LIUQE comparison from one corrected sweep receipt."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from scripts.prototypes.tcv_liuqe_known_answer.summarise import measure_gaps


def _number(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _point(value):
    if value is None or len(value) != 2 or not all(math.isfinite(x) for x in value):
        return "—"
    return f"({_number(value[0])}, {_number(value[1])})"


def _coverage(row):
    return (
        f"{html.escape(str(row.get('coverage_certified')))}; winding "
        f"{_number(row.get('coverage_total'), 0)} / index "
        f"{_number(row.get('coverage_index_sum'), 0)}; "
        f"{row.get('coverage_cells_uncertified', '—')} uncleared cells"
    )


def _row_html(row):
    gap = measure_gaps(row)
    xpoints = "; ".join(_point(point) for point in row["liuqe_xpoints"]) or "—"
    nova_saddle = row.get("nova_boundary") if row["nova_label"] == "diverted" else None
    return (
        f"<tr><th scope='row'>{row['index']}<br>{_number(row['liuqe_time'], 3)} s</th>"
        f"<td>{html.escape(row['liuqe_label'])}<br>{html.escape(row['nova_label'])}</td>"
        f"<td>{_number(row['liuqe_wall_distance'])}<br>"
        f"{_number(row['liuqe_xpoint_distance'])}</td>"
        f"<td>LIUQE {_point(row['liuqe_axis'])}<br>nova {_point(row.get('nova_axis'))}"
        f"<br>Δ {_number(gap['axis_gap'])} m</td>"
        f"<td>LIUQE {xpoints}<br>nova "
        f"{_point(nova_saddle)}</td>"
        f"<td>LIUQE {_number(row['liuqe_boundary_psi_grid'])}<br>"
        f"nova {_number(row.get('nova_boundary_psi'))}<br>"
        f"|Δ| {_number(gap['boundary_gap'])}; /span "
        f"{_number(gap['boundary_relative'], 3)}</td>"
        f"<td>{_coverage(row)}</td><td>{row.get('resolution') or '—'}</td></tr>"
    )


def _diagnosis(row):
    """Locate each different boundary decision on the common LIUQE grid."""
    gap = measure_gaps(row)
    label_differs = row["liuqe_label"] != row["nova_label"]
    flux_differs = math.isfinite(gap["boundary_gap"]) and gap["boundary_gap"] > 1e-9
    if not label_differs and not flux_differs:
        return ""
    reference_level = row["liuqe_boundary_psi_grid"]
    measured_level = row.get("nova_boundary_psi")
    evidence = (
        f"LIUQE ψaxis {_number(row['liuqe_axis_psi_grid'])}, "
        f"nova ψaxis {_number(row.get('nova_axis_psi'))}; "
        f"LIUQE ψboundary {_number(reference_level)}, nova {_number(measured_level)}; "
        f"LIUQE axis {_point(row['liuqe_axis'])}, nova {_point(row.get('nova_axis'))}; "
        f"nova boundary {_point(row.get('nova_boundary'))}; "
        f"{row.get('nova_saddles', '—')} admitted saddles; "
        f"wall contact {_number(row['liuqe_wall_distance'])} m, "
        f"nearest stored X-point {_number(row['liuqe_xpoint_distance'])} m "
        "from outline."
    )
    if row["nova_label"] == "coverage uncertified":
        departure = (
            "Coverage refusal: " + _coverage(row) + ". "
            "The census certificate in crosscheck.py:914 and the refusal at "
            "crosscheck.py:930–939 stop the read before it publishes a boundary. "
            "Inspect the uncleared cells and missing stationary roots on this "
            "LIUQE grid; retain the refusal until the exclusion is proven."
        )
    elif row["nova_label"] == "refused invalid axis":
        departure = (
            "Axis refusal at crosscheck.py:957–963 publishes no boundary. "
            "Check the admitted root type and position on this LIUQE grid."
        )
    elif row["nova_label"] == "unread":
        departure = (
            "Read failed before a qualified comparison; the receipt's attempts "
            f"say {html.escape(str(row.get('attempts')))}."
        )
    else:
        parts = []
        if math.isfinite(gap["axis_gap"]) and gap["axis_gap"] > 0.02:
            parts.append(
                f"axis selection differs by {_number(gap['axis_gap'])} m: "
                "kernel.py:302–305 ranks eligible extrema by signed ψ; inspect "
                "the selected basin against LIUQE's stored axis"
            )
        if label_differs or flux_differs:
            prescription = (
                "recompute the effective wall path ceiling and admitted saddle "
                "score on LIUQE's ψ, then compare the chosen contact with its outline"
                if label_differs
                else "compare the discrete wall or saddle candidate level with "
                "LIUQE's sampled outline level and refine the candidate where "
                "that comparison identifies an error"
            )
            parts.append(
                f"boundary choice differs by {_number(gap['boundary_gap'])} ψ "
                f"({_number(gap['boundary_relative'], 3)} reference spans): "
                "kernel.py:403–409 compares the wall path level with the admitted "
                f"saddle; {prescription}"
            )
        departure = "; ".join(parts) + "."
    return (
        f"<li><strong>Slice {row['index']} ({_number(row['liuqe_time'], 3)} s): "
        f"LIUQE {html.escape(row['liuqe_label'])}, nova "
        f"{html.escape(row['nova_label'])}.</strong> {evidence} {departure}</li>"
    )


def build(rows, receipt_path):
    if len(rows) != 17 or len({row["index"] for row in rows}) != 17:
        raise ValueError("report requires every distinct corrected grid block")
    if any(row.get("nova_label") == "unread" for row in rows):
        raise ValueError("unread rows require review before reporting")
    if any(row.get("liuqe_label") not in ("limited", "diverted") for row in rows):
        raise ValueError("reference geometry must classify every slice")
    agreements = sum(row["liuqe_label"] == row["nova_label"] for row in rows)
    refusals = sum(row["nova_label"] not in ("limited", "diverted") for row in rows)
    diagnoses = "\n".join(filter(None, (_diagnosis(row) for row in rows)))
    count_diagnosed = diagnoses.count("<li>")
    summary = (
        f"{agreements} of 17 geometric labels agree; {refusals} nova reads refuse "
        f"a label; {count_diagnosed} slices have a label or measurable boundary "
        "flux difference."
    )
    table = "\n".join(_row_html(row) for row in rows)
    figure_root = "/nova/figures/converged-forward-solve/tcv-liuqe-known-answer"
    figures = (("early", rows[:6]), ("middle", rows[6:12]), ("late", rows[12:]))
    figure_html = "\n".join(
        f"<figure><img src='{figure_root}/tcv-77549-liuqe-nova-boundary-{name}.svg' "
        f"alt='LIUQE and nova boundary comparison for slices "
        f"{group[0]['index']} through {group[-1]['index']}'>"
        "<figcaption>Corrected COCOS-11 LIUQE ψ: common line-contour levels; "
        "LIUQE stored outline dashed red, grid contour at the outline's mean "
        "sampled ψ dotted grey, nova boundary blue; "
        "both null sets and the wall are marked. "
        + "; ".join(
            f"{row['index']}: LIUQE {row['liuqe_label']}, nova {row['nova_label']}"
            for row in group
        )
        + ".</figcaption></figure>"
        for name, group in figures
    )
    geometry_rule = (
        "The LIUQE label comes from its own stored boundary outline: diverted "
        "when an outline segment passes within 0.020 m of a stored X-point; "
        "otherwise limited when the outline touches the wall within 0.002 m. "
        "An outline that meets neither test is refused. Both minimum distances "
        "are listed for every slice. The limits separate the observed sampled "
        "contacts: limited wall distances at most 0.00024 m and nearest distant "
        "X-point at least 0.044 m; diverted X-point distances at most 0.011 m "
        "and wall clearance at least 0.009 m. A stored X-point beyond a limited "
        "outline does not change its label."
    )
    report = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<title>TCV 77549 LIUQE known-answer read</title></head><body>
<h1>TCV 77549 LIUQE known-answer read</h1>
<p>The reference is imas-efit's corrected full pulse at
<code>/home/ITER/mcintos/public/efit/validation/TCV/77549/liuqe-corrected</code>,
read with imas-python at its written DD 3.41.0 (COCOS 11). It has 17 distinct
ψ blocks; every block start is scored once. The wall IDS is DD 4.1.0.
No sign compensation is applied in nova.</p>
<h2>Reference rule</h2><p>{geometry_rule}</p>
<p><strong>{summary}</strong> All rows come from one sirius job receipt:
<code>{html.escape(str(receipt_path))}</code>. A dash means absent or refused,
never zero. Boundary flux and axes from both readers are evaluated on the same
LIUQE grid spline. The stored outline defines LIUQE's geometric label; the
grey diagnostic contour at the outline's mean sampled grid ψ may depart from
that outline and does not change its label.</p>
<table><thead><tr><th>Slice / time</th><th>LIUQE / nova</th>
<th>Wall / X distance [m]</th><th>Axis R,Z [m]</th><th>Stored X / nova saddle [m]</th>
<th>Boundary ψ on LIUQE grid</th><th>Coverage</th><th>Lattice</th></tr></thead>
<tbody>{table}</tbody></table>
<h2>Departures on LIUQE ψ</h2><p>Every label disagreement, refusal, or measurable
boundary flux gap is listed. The cited
rules are the read decisions to inspect; each prescription stays conditional
on the measured field and preserves a coverage refusal until it is certified.</p>
<ol>{diagnoses}</ol>
{figure_html}
</body></html>\n"""
    reference_fragment = f"""<a name='tcv-liuqe-known-answer'></a>
<section id='tcv-liuqe-refresh-reference'>
<h2>Corrected TCV LIUQE boundary read</h2>
<p id='tcv-liuqe-refresh-summary'>{summary}</p>
<p id='tcv-liuqe-refresh-rule'>{geometry_rule}</p>
<p>The 17 rows, measured distances, departures and three poloidal panels are in
<a href='/nova/research/tcv-liuqe-known-answer.html'>the research record</a>.
The one-job receipt is <code>{html.escape(str(receipt_path))}</code>.</p>
</section>\n"""
    landing_fragment = f"""<section id='tcv-liuqe-refresh-landing'>
<h2>TCV LIUQE corrected reference comparison</h2><p>{summary}</p>
<p>Slice 0's corrected outline touches the wall while its stored X-points stay
outside it; the base label-rule test fails and the corrected rule passes.
<a href='/nova/research/tcv-liuqe-known-answer.html'>The table and three panels</a>
show the complete one-job rescore and per-slice read diagnosis.</p>
</section>\n"""
    return report, reference_fragment, landing_fragment, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--fragment", required=True)
    parser.add_argument("--landing-fragment", required=True)
    args = parser.parse_args()
    source = Path(args.rows)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    rows.sort(key=lambda row: row["index"])
    report, fragment, landing, summary = build(rows, source)
    Path(args.report).write_text(report)
    Path(args.fragment).write_text(fragment)
    Path(args.landing_fragment).write_text(landing)
    print(summary)


if __name__ == "__main__":
    main()
