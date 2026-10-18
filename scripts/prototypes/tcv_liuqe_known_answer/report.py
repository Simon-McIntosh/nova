"""Build the known-answer report and landing fragment from one sweep receipt."""

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
    if (
        value is None
        or len(value) != 2
        or not all(math.isfinite(float(x)) for x in value)
    ):
        return "—"
    return f"({_number(value[0])}, {_number(value[1])})"


def _pair(reference, reading, gap, relative, *, digits=4):
    return (
        f"LIUQE {_number(reference, digits)}<br>nova {_number(reading, digits)}"
        f"<br>|Δ| {_number(gap, digits)}; /span {_number(relative, 3)}"
    )


def _point_pair(reference, reading, gap, relative):
    return (
        f"LIUQE {_point(reference)}<br>nova {_point(reading)}"
        f"<br>|Δ| {_number(gap, 4)} m; /span {_number(relative, 3)}"
    )


def _x_pair(row, gaps):
    points = row.get("liuqe_xpoints") or []
    nova = row.get("nova_boundary") if row.get("nova_label") == "diverted" else None
    reference = "; ".join(_point(point) for point in points) if points else "—"
    return (
        f"LIUQE {reference}<br>nova {_point(nova)}"
        f"<br>nearest |Δ| {_number(gaps['x_gap'])} m; "
        f"/span {_number(gaps['x_relative'], 3)}"
    )


def _coverage(row):
    return (
        f"outer {_number(row.get('coverage_outer_winding'), 0)}, "
        f"obstacles {_number(row.get('coverage_obstacle_winding'), 0)}, "
        f"domain {_number(row.get('coverage_total'), 0)}; "
        f"admitted index {_number(row.get('coverage_index_sum'), 0)}"
        f"<br>empty cells clear {row.get('coverage_cells_cleared', '—')}, "
        f"uncertified {row.get('coverage_cells_uncertified', '—')}, "
        f"occupied {row.get('coverage_cells_occupied', '—')}"
    )


def _panel_labels(group):
    return "; ".join(
        f"{row['index']}: LIUQE {row['liuqe_label']}, nova {row['nova_label']}"
        for row in group
    )


def _row_html(row):
    gap = measure_gaps(row)
    label = html.escape(str(row.get("nova_label", "unread")))
    axis = _point_pair(
        row["liuqe_axis"], row.get("nova_axis"), gap["axis_gap"], gap["axis_relative"]
    )
    axis_flux = _pair(
        row["liuqe_axis_psi_grid"],
        row.get("nova_axis_psi"),
        gap["axis_psi_gap"],
        gap["axis_psi_relative"],
    )
    boundary_flux = _pair(
        row["liuqe_boundary_psi_grid"],
        row.get("nova_boundary_psi"),
        gap["boundary_gap"],
        gap["boundary_relative"],
    )
    return (
        f"<tr><th scope='row'>{row['index']}<br>{_number(row['liuqe_time'], 3)} s</th>"
        f"<td>{html.escape(row['liuqe_label'])}<br>{label}</td>"
        f"<td>{axis}</td>"
        f"<td>{_number(gap['span_length'])}</td>"
        f"<td>{_number(gap['axis_x_distance'])}</td>"
        f"<td>{axis_flux}</td>"
        f"<td>{_x_pair(row, gap)}</td>"
        f"<td>{boundary_flux}</td>"
        f"<td>{_coverage(row)}</td><td>{row.get('resolution') or '—'}</td></tr>"
    )


def build(
    rows,
    field_swap_rows,
    receipt_path,
    *,
    previous_rows=(),
    previous_field_swap_rows=(),
):
    if len(rows) != 12 or len({row["index"] for row in rows}) != 12:
        raise ValueError("report requires twelve distinct, completed sweep rows")
    if any(row.get("nova_label") == "unread" for row in rows):
        raise ValueError("unread rows require explicit review before reporting")
    if len(field_swap_rows) != 5 or len({row["case"] for row in field_swap_rows}) != 5:
        raise ValueError("report requires five distinct field-swap refusal rows")
    if any(row.get("nova_label") == "unread" for row in field_swap_rows):
        raise ValueError("unread field-swap rows require review before reporting")
    if previous_rows and len(previous_rows) != len(rows):
        raise ValueError("comparison requires the same twelve slice identifiers")
    if previous_field_swap_rows and len(previous_field_swap_rows) != len(
        field_swap_rows
    ):
        raise ValueError("comparison requires the same five field-swap identifiers")

    def verdict_changes(current, previous, identifier):
        old_by_id = {item[identifier]: item for item in previous}
        if previous and set(old_by_id) != {item[identifier] for item in current}:
            raise ValueError("comparison identifiers differ")
        return [
            f"{item[identifier]}: {old_by_id[item[identifier]]['nova_label']} → "
            f"{item['nova_label']}"
            for item in current
            if previous
            and (
                old_by_id[item[identifier]]["nova_label"] != item["nova_label"]
                or old_by_id[item[identifier]].get("coverage_certified")
                != item.get("coverage_certified")
            )
        ]

    slice_changes = verdict_changes(rows, previous_rows, "index")
    swap_changes = verdict_changes(field_swap_rows, previous_field_swap_rows, "case")
    change_summary = (
        "Coverage verdict changes from the preceding receipt: slices "
        + ("; ".join(slice_changes) if slice_changes else "none")
        + "; field-swap records "
        + ("; ".join(swap_changes) if swap_changes else "none")
        + "."
        if previous_rows and previous_field_swap_rows
        else ""
    )
    agreements = sum(row["liuqe_label"] == row["nova_label"] for row in rows)
    refusals = sum(row["nova_label"] == "coverage uncertified" for row in rows)
    prior_limited = {0, 14, 456, 796, 1257}
    misses = sum(
        row["index"] in prior_limited and row["nova_label"] == "coverage uncertified"
        for row in rows
    )
    gaps = [measure_gaps(row) for row in rows]
    if any(not math.isfinite(gap["span_length"]) for gap in gaps):
        raise ValueError("every slice requires LIUQE's boundary outline span")
    if any(
        math.isfinite(gap["axis_gap"]) and not math.isfinite(gap["axis_relative"])
        for gap in gaps
    ):
        raise ValueError("a measured axis gap lacks its relative value")

    def largest(name):
        return max(
            (
                (row["index"], gap[name])
                for row, gap in zip(rows, gaps)
                if math.isfinite(gap[name])
            ),
            key=lambda pair: pair[1],
            default=(None, float("nan")),
        )

    axis_slice, max_axis = largest("axis_gap")
    axis_relative_slice, max_axis_relative = largest("axis_relative")
    x_slice, max_x = largest("x_gap")
    x_relative_slice, max_x_relative = largest("x_relative")
    axis_psi_slice, max_axis_psi = largest("axis_psi_gap")
    axis_psi_relative_slice, max_axis_psi_relative = largest("axis_psi_relative")
    boundary_slice, max_boundary = largest("boundary_gap")
    boundary_relative_slice, max_boundary_relative = largest("boundary_relative")
    x_phrase = (
        f"X-point absolute {_number(max_x)} m at slice {x_slice}; relative "
        f"{_number(max_x_relative, 3)} spans at slice {x_relative_slice}"
        if math.isfinite(max_x)
        else "no paired X-point distance is measurable"
    )
    figure_root = "/nova/figures/converged-forward-solve/tcv-liuqe-known-answer"
    figures = (
        ("early", rows[:4]),
        ("middle", rows[4:8]),
        ("late", rows[8:]),
    )
    caption = (
        "LIUQE psi line contours at 12 equally spaced levels between the grid "
        "axis and boundary psi values tabulated above. The TCV wall is per loop; "
        "LIUQE's grid boundary is red dashed, its stored outline grey dotted, "
        "and nova's available boundary contour blue. Both axis and X-point "
        "marker sets are shown. A refused read has no nova boundary to draw."
    )
    summary = (
        f"{agreements} of 12 labels agree; {refusals} reads are coverage uncertified. "
        f"Of the five diverted slices previously read as limited, {misses} are now "
        "coverage misses. "
        f"Largest measured gaps: axis position absolute {_number(max_axis)} m at "
        f"slice {axis_slice}; relative {_number(max_axis_relative, 3)} spans at "
        f"slice {axis_relative_slice}; {x_phrase}; axis psi absolute "
        f"{_number(max_axis_psi)} at slice {axis_psi_slice}; relative "
        f"{_number(max_axis_psi_relative, 3)} flux spans at slice "
        f"{axis_psi_relative_slice}; boundary psi absolute {_number(max_boundary)} "
        f"at slice {boundary_slice}; relative {_number(max_boundary_relative, 3)} "
        f"flux spans at slice {boundary_relative_slice}."
    )
    table = "\n".join(_row_html(row) for row in rows)
    field_swap_table = "\n".join(
        f"<tr><th scope='row'>{html.escape(row['case'])}</th>"
        f"<td>{html.escape(row['nova_label'])}</td>"
        f"<td>outer {_number(row.get('coverage_outer_winding'), 0)}, "
        f"obstacles {_number(row.get('coverage_obstacle_winding'), 0)}, "
        f"domain {_number(row.get('coverage_winding'), 0)}, "
        f"index {_number(row.get('coverage_index_sum'), 0)}</td>"
        f"<td>{row.get('coverage_cells_cleared', '—')} clear; "
        f"{row.get('coverage_cells_uncertified', '—')} uncertified</td></tr>"
        for row in field_swap_rows
    )
    figure_html = "\n".join(
        f"<figure><img src='{figure_root}/tcv-77549-liuqe-nova-boundary-{name}.svg' "
        f"alt='TCV LIUQE and nova poloidal flux, slices "
        f"{group[0]['index']} through {group[-1]['index']}'>"
        f"<figcaption>{caption} Left to right: "
        f"{_panel_labels(group)}."
        "</figcaption></figure>"
        for name, group in figures
    )
    report = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>TCV 77549 LIUQE known-answer read</title></head><body>
<h1>TCV 77549 LIUQE known-answer read</h1>
<p>LIUQE is the reference. Nova reads each LIUQE psi grid through the shared field-swap
spline and per-loop TCV wall adapter. The named validation extract contains one frozen
MAG_ONLY grid over 50 slices; this measurement uses the source reconstruction at
<code>/home/ITER/mcintos/public/imasdb/tcv/3/77549/1</code>, DD 3.41.0, whose 1,703
slices contain 17 distinct psi grids. The wall IDS uses DD 4.1.0 and is read with
<code>nova.equilibrium.wall_mask.wall_units_from_ids</code>. LIUQE boundary/type is
unset, so stored X-point presence supplies its limited or diverted label.</p>
<h2>Certificate and scoring</h2>
<p>A read qualifies only if the counterclockwise outer boundary winding minus obstacle
windings equals the admitted critical-point index sum and every sampled loop clears
its gradient floor. In every root-associated cell, a Krawczyk box must prove that
the admitted root is the only gradient zero nearby: its Hessian enclosure makes
the preconditioned gradient map a strict self-map with contraction norm below one.
Bernstein enclosures then exclude zeros from every part of every lattice cell outside
those boxes. An uncleared box is bisected to depth 6; work stops after 20,000 boxes
per lattice, including root-box attempts. Any remaining cell leaves the read
<em>coverage uncertified</em>. This is a refusal, not a limited answer.</p>
<p>Flux pairs use the LIUQE grid spline, not the stored scalar psi values, whose gauge
differs across blocks. Absolute flux gaps are divided by LIUQE's grid axis-to-boundary
flux span. Position gaps are divided by the shortest distance from LIUQE's axis
to its own piecewise-linear boundary outline, including limited slices. The
axis-to-nearest-X distance is an extra measure when LIUQE stores an X-point.
The X pair uses the LIUQE X-point nearest nova's admitted saddle where both exist.</p>
<p><strong>{summary}</strong> All twelve rows come from one rescore receipt:
<code>{html.escape(str(receipt_path))}</code>. A dash means the value was not returned
or no reference X-point exists; it is never scored as zero.</p>
<p>{html.escape(change_summary)}</p>
<table><thead><tr><th>Slice / time</th><th>LIUQE / nova label</th><th>Axis R,Z [m]</th>
<th>Axis-to-boundary [m]</th><th>Axis-to-X [m]</th><th>Axis ψ</th>
<th>X-point R,Z [m]</th><th>Boundary ψ from grid</th>
<th>Coverage certificate</th><th>Lattice</th></tr></thead><tbody>
{table}
</tbody></table>
<h2>Previously refused field-swap records</h2>
<p>The five records the earlier square-lattice census refused were rerun in the same
job with the same certificate. Their complete attempts are in the paired field-swap
receipt beside the LIUQE receipt.</p>
<table><thead><tr><th>Record</th><th>Nova outcome</th><th>Boundary / index</th>
<th>Cell enclosure</th></tr></thead><tbody>{field_swap_table}</tbody></table>
{figure_html}
<p>The certificate counts are lattice cells: cells proved zero-free away from the
root boxes, cells left uncertified, and cells associated with an admitted root box.
The receipt records
each tried resolution and its refusal reason. A matching boundary index alone is not
enough: an omitted extremum and saddle can cancel in that sum.</p>
</body></html>\n"""
    fragment = f"""<section id="tcv-liuqe-known-answer">
<h2 id="tcv-liuqe-known-answer-heading">TCV LIUQE known-answer boundary read</h2>
<p id="tcv-liuqe-known-answer-summary">{summary}</p>
<p id="tcv-liuqe-known-answer-verdict-changes">{html.escape(change_summary)}</p>
<p id="tcv-liuqe-known-answer-certificate">The boundary-index check, Krawczyk
uniqueness boxes around admitted roots, and Bernstein gradient exclusion on the
remaining cell regions agree on qualified reads. Any uncleared region returns
coverage uncertified. The twelve rows and all five prior field-swap refusals
were rescored in one CPU job. The per-slice comparison is in
<a href="/nova/research/tcv-liuqe-known-answer.html">the research report</a>.</p>
{figure_html}
</section>\n"""
    return report, fragment, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--field-swap-rows", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--fragment", required=True)
    parser.add_argument("--figure-rows", required=True)
    parser.add_argument("--previous-rows")
    parser.add_argument("--previous-field-swap-rows")
    args = parser.parse_args()
    source = Path(args.rows)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    field_swap_rows = [
        json.loads(line)
        for line in Path(args.field_swap_rows).read_text().splitlines()
        if line
    ]
    rows.sort(key=lambda row: row["index"])
    previous_rows = (
        [
            json.loads(line)
            for line in Path(args.previous_rows).read_text().splitlines()
            if line
        ]
        if args.previous_rows
        else []
    )
    previous_field_swap_rows = (
        [
            json.loads(line)
            for line in Path(args.previous_field_swap_rows).read_text().splitlines()
            if line
        ]
        if args.previous_field_swap_rows
        else []
    )
    report, fragment, summary = build(
        rows,
        field_swap_rows,
        source,
        previous_rows=previous_rows,
        previous_field_swap_rows=previous_field_swap_rows,
    )
    Path(args.report).write_text(report)
    Path(args.fragment).write_text(fragment)
    figure_rows = Path(args.figure_rows)
    figure_rows.write_text(json.dumps({"rows": rows}, indent=2))
    for name, group in (
        ("early", rows[:4]),
        ("middle", rows[4:8]),
        ("late", rows[8:]),
    ):
        (figure_rows.parent / f"figure-{name}.json").write_text(
            json.dumps({"rows": group}, indent=2)
        )
    print(summary)


if __name__ == "__main__":
    main()
