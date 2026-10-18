"""Compare imas-efit's escape-level rule with nova's boundary read, side by side.

Both rules are read from their own recorded output and joined on the trace
record side. Nothing here evaluates a field or re-implements a rule: the
imas-efit columns come from its escape-level replay log, the nova columns from
the field-swap per-loop sweep. The module writes one row per record side, a
machine-level agreement summary, one diagnostic line per disagreement, and the
research page and evidence fragment that carry them.

A record side is imas-efit's A or B trace of one paired input. JT-60SA sides
carry no trusted reconstruction, so their comparison is stated as rule
consistency only.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import pathlib
import re


ESCAPE_DIR = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-escape-level-domain"
)
NOVA_SWEEP = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261018T104543238298-efit-field-swap-repair/exports/disagreement-loops.json"
)

# the escape walk reports a tracked maximum and a recorded axis per pair; only
# this side is stated by the source to consult the tracked maximum instead
TRACKED_AXIS_SIDE = ("jt60sa-5slice-s5-301-302", "B")

MACHINES = (
    ("jt60sa", "JT-60SA"),
    ("aug", "AUG"),
    ("iter-dina3420", "ITER"),
    ("iter-fullPulse", "ITER"),
    ("iter", "ITER"),
    ("mastu", "MAST-U"),
    ("synthetic", "synthetic"),
)

# sides whose machine carries no trusted reconstruction
RULE_CONSISTENCY_ONLY = "JT-60SA"


def machine_of(pair: str) -> str:
    for key, label in MACHINES:
        if pair.startswith(key):
            return label
    return "other"


def nova_case(pair: str, side: str) -> str:
    if pair == "synthetic_crossing":
        return f"synthetic_crossing_{side.lower()}"
    return f"{pair}:{side}"


def parse_escape(lines):
    accuracy = {}
    endpoints = {}
    for line in lines:
        match = re.match(
            r"\s*ESCAPE_ACCURACY: pair=(\S+) side=(\S+) label=(\S+) "
            r"carrier=(\S+) refined=(\S+) carrier_r=(\S+) carrier_z=(\S+) "
            r"refined_r=(\S+) refined_z=(\S+)",
            line,
        )
        if match:
            accuracy[(match.group(1), match.group(2))] = {
                "label": int(match.group(3)),
                "carrier": float(match.group(4)),
                "level": float(match.group(5)),
                "bottleneck": [float(match.group(8)), float(match.group(9))],
                "carrier_bottleneck": [float(match.group(6)), float(match.group(7))],
            }
            continue
        match = re.match(
            r"ESCAPE_AXIS_ENDPOINT: pair=(\S+) tracked_r=(\S+) tracked_z=(\S+) "
            r"recorded_r=(\S+) recorded_z=(\S+) distance=(\S+)",
            line,
        )
        if match:
            endpoints[match.group(1)] = {
                "tracked": [float(match.group(2)), float(match.group(3))],
                "recorded": [float(match.group(4)), float(match.group(5))],
                "distance": float(match.group(6)),
            }
    return accuracy, endpoints


def escape_axis(pair, side, endpoints):
    """The axis imas-efit's escape walk starts from at this side.

    The walk starts from the record's stored axis, except on the one side the
    source states it consults the tracked maximum instead.
    """
    if (pair, side) == TRACKED_AXIS_SIDE:
        return list(endpoints[pair]["tracked"]), "tracked maximum"
    return list(endpoints[pair]["recorded"]), "recorded axis"


def separation(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_rows(accuracy, endpoints, nova):
    rows = []
    seen = set()
    for pair in dict.fromkeys(key[0] for key in accuracy):
        for side in ("A", "B"):
            if (pair, side) not in accuracy:
                continue
            seen.add(pair)
            esc = accuracy[(pair, side)]
            n = nova.get(nova_case(pair, side))
            rows.append(join_side(pair, side, esc, endpoints, n))
    return rows


def nova_contact(nova_label, nova_boundary, n, nova_wall):
    """The contact nova's read admitted, at the level that read decided.

    A limited decision admits a wall contact: the boundary level nova published
    is realised on the wall at the point its read admitted, so that point is
    reported at nova's own boundary level. The wall extremum over the whole
    vessel is a different quantity -- on a record whose strongest wall maximum
    lies on a lobe the axis-connected component never reaches it sits above the
    admitted level, as MAST-U's four limited sides do -- so it is never
    reported as the contact. A diverted decision admits an interior saddle at
    the level the read published for it.
    """
    if nova_label == "limited":
        contact = {
            "kind": "wall",
            "level": nova_boundary,
            "position": None,
            "source": "wall contact at nova's boundary level",
        }
        if n.get("nova_saddle_r") is not None and n.get("nova_saddle_z") is not None:
            contact["position"] = [n["nova_saddle_r"], n["nova_saddle_z"]]
        if nova_boundary is None or not math.isfinite(nova_boundary):
            return None
        return contact
    if nova_label == "diverted":
        if n.get("nova_saddle_psi") is None or not math.isfinite(n["nova_saddle_psi"]):
            return None
        return {
            "kind": "saddle",
            "level": n["nova_saddle_psi"],
            "position": [n["nova_saddle_r"], n["nova_saddle_z"]],
            "source": "interior saddle at nova's published saddle level",
        }
    return None


def join_side(pair, side, esc, endpoints, n):
    axis, axis_source = escape_axis(pair, side, endpoints)
    nova_axis = None
    if n is not None and n.get("nova_axis_r") is not None:
        nova_axis = [n["nova_axis_r"], n["nova_axis_z"]]
    if n is None:
        nova_label = "absent"
        nova_boundary = None
        contact = None
        loops = None
        nova_span = None
        efit_span = None
        nova_wall = None
    else:
        nova_label = n["nova_label"]
        nova_boundary = n["nova_boundary_psi"]
        loops = n["wall_loop_count"]
        nova_span = n["nova_span"]
        efit_span = n["efit_span"]
        nova_wall = n["nova_wall_level"]
        contact = nova_contact(nova_label, nova_boundary, n, nova_wall)

    gap = None
    rel = None
    if nova_boundary is not None and math.isfinite(nova_boundary):
        gap = abs(nova_boundary - esc["level"])
        if efit_span not in (None, 0) and math.isfinite(efit_span):
            rel = gap / abs(efit_span)

    label_map = {1: "limited", 2: "diverted"}
    return {
        "pair": pair,
        "side": side,
        "case": f"{pair}-{side}",
        "machine": machine_of(pair),
        "loops": loops,
        "efit_label": label_map[esc["label"]],
        "efit_label_value": esc["label"],
        "efit_level": esc["level"],
        "efit_bottleneck": esc["bottleneck"],
        "efit_carrier_bottleneck": esc["carrier_bottleneck"],
        "efit_axis": axis,
        "efit_axis_source": axis_source,
        "efit_axis_note": "tracked maximum, not the recorded axis"
        if axis_source != "recorded axis"
        else "",
        "nova_label": nova_label,
        "nova_boundary": nova_boundary,
        "nova_contact": contact,
        "nova_axis": nova_axis,
        "nova_wall_extremum": nova_wall,
        "nova_admitted_point": None
        if n is None or n.get("nova_saddle_r") is None
        else [n["nova_saddle_r"], n["nova_saddle_z"]],
        "gap_abs": gap,
        "gap_rel": rel,
        "efit_span": efit_span,
        "nova_span": nova_span,
        "axis_separation": separation(nova_axis, axis),
        "note": (n or {}).get("note", ""),
    }


def diagnose(row):
    if row["nova_label"] == "absent":
        return "no nova row: this side has no field-swap decision to compare"
    if row["nova_label"] == "unqualified":
        return (
            "nova refused a boundary decision: no lattice admitted a saddle, so "
            "only imas-efit's escape label is recorded"
        )
    if row["nova_label"] == row["efit_label"]:
        return "agree"
    axis_gap = row["axis_separation"]
    rows = []
    if axis_gap is not None and axis_gap > 0.05:
        rows.append(
            f"axis: nova reads ({fmt(row['nova_axis'][0])},"
            f"{fmt(row['nova_axis'][1])}) against the escape walk's "
            f"({fmt(row['efit_axis'][0])},{fmt(row['efit_axis'][1])}), "
            f"{fmt(axis_gap)} m apart"
        )
    if row["loops"] and row["loops"] > 1:
        rows.append(
            f"obstacle loops: {row['loops']} wall loops read as separate units "
            f"typed vessel"
        )
    rows.append(
        "saddle-versus-wall contest: escape levels "
        f"{row['efit_label']} at {fmt(row['efit_level'])} on the "
        f"{'wall' if row['efit_label'] == 'limited' else 'interior saddle'}, "
        f"nova levels {row['nova_label']} at {fmt(row['nova_boundary'])}"
    )
    return "; ".join(rows)


def fmt(value, nd=6):
    if value is None:
        return "&ndash;"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "&ndash;"
        if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e5):
            return f"{value:.{nd}g}"
        return f"{value:.{nd}g}"
    return str(value)


def summary(rows):
    counts = {}
    for row in rows:
        entry = counts.setdefault(
            row["machine"], {"sides": 0, "agree": 0, "disagree": 0, "unqualified": 0}
        )
        entry["sides"] += 1
        if row["nova_label"] == "unqualified":
            entry["unqualified"] += 1
        elif row["nova_label"] == row["efit_label"]:
            entry["agree"] += 1
        else:
            entry["disagree"] += 1
    return counts


def side_table(rows):
    out = [
        "<table><thead><tr>"
        "<th>side</th><th>machine</th><th>loops</th>"
        "<th>escape label</th><th>escape level</th><th>escape bottleneck (R, Z)</th>"
        "<th>escape axis used</th>"
        "<th>nova label</th><th>nova boundary</th><th>nova contact</th>"
        "<th>nova axis</th><th>|&Delta;&psi;|</th><th>|&Delta;| / efit span</th>"
        "</tr></thead><tbody>"
    ]
    for row in rows:
        contact = row["nova_contact"]
        if contact is None:
            contact_text = "&ndash;"
        elif contact["kind"] == "saddle":
            contact_text = (
                f"saddle &psi; {fmt(contact['level'])} at "
                f"({fmt(contact['position'][0])}, {fmt(contact['position'][1])})"
            )
        elif contact["position"] is not None:
            contact_text = (
                f"wall contact &psi; {fmt(contact['level'])} at "
                f"({fmt(contact['position'][0])}, {fmt(contact['position'][1])})"
            )
        else:
            contact_text = f"wall contact &psi; {fmt(contact['level'])}"
        axis_text = (
            f"({fmt(row['efit_axis'][0])}, {fmt(row['efit_axis'][1])})"
            f"<br><span class='muted'>{html.escape(row['efit_axis_source'])}</span>"
        )
        if row["efit_axis_note"]:
            axis_text += " <strong>&#9873;</strong>"
        nova_axis_text = (
            f"({fmt(row['nova_axis'][0])}, {fmt(row['nova_axis'][1])})"
            if row["nova_axis"]
            else "&ndash;"
        )
        out.append(
            f"<tr><td>{html.escape(row['case'])}</td><td>{row['machine']}</td>"
            f"<td>{row['loops'] if row['loops'] is not None else '&ndash;'}</td>"
            f"<td>{row['efit_label']}</td><td>{fmt(row['efit_level'])}</td>"
            f"<td>({fmt(row['efit_bottleneck'][0])}, "
            f"{fmt(row['efit_bottleneck'][1])})</td>"
            f"<td>{axis_text}</td>"
            f"<td>{row['nova_label']}</td><td>{fmt(row['nova_boundary'])}</td>"
            f"<td>{contact_text}</td><td>{nova_axis_text}</td>"
            f"<td>{fmt(row['gap_abs'])}</td><td>{fmt(row['gap_rel'])}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def summary_table(rows):
    counts = summary(rows)
    out = [
        "<table><thead><tr><th>machine</th><th>sides</th><th>agree</th>"
        "<th>disagree</th><th>nova unqualified</th></tr></thead><tbody>"
    ]
    for machine, entry in counts.items():
        out.append(
            f"<tr><td>{machine}</td><td>{entry['sides']}</td>"
            f"<td>{entry['agree']}</td><td>{entry['disagree']}</td>"
            f"<td>{entry['unqualified']}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def diagnosis_list(rows):
    out = ["<ul>"]
    for row in rows:
        text = diagnose(row)
        if text == "agree":
            continue
        out.append(
            f"<li><code>{html.escape(row['case'])}</code> "
            f"(escape {row['efit_label']}, nova {row['nova_label']}): "
            f"{html.escape(text)}</li>"
        )
    out.append("</ul>")
    return "\n".join(out)


def figure_links():
    out = []
    for machine in ("JT-60SA", "AUG", "ITER", "MAST-U", "synthetic"):
        slug = machine.lower().replace("-", "")
        out.append(
            f"<figure><img src='/nova/figures/converged-forward-solve/"
            f"efit-axis-adjudication/efit-axis-{slug}.svg' "
            f"alt='{html.escape(machine)} panels' width='100%'>"
            f"<figcaption>{html.escape(machine)} trace records. Each panel draws the "
            f"record's own spline as line contours on shared levels, the wall as "
            f"separate indexed loops, and the escape bottleneck as its own marker. "
            f"Fixed points and boundary levels are identified in the legend."
            f"</figcaption></figure>"
        )
    return "\n".join(out)


def build_page(rows, commits):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Escape-level rule comparison | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_data/dashboard.css">
</head>
<body><main class="plan-doc">
<h1 id="efit-escape-level-comparison-title">Escape-level rule comparison</h1>
<p id="efit-escape-level-comparison-summary">Nova's boundary read and imas-efit's
escape-level rule are set side by side on the 26 sides of imas-efit's 13 paired
trace records. Both rules are read from their own recorded output; neither is
changed here and no verdict is stated on which rule is right. The escape columns
come from imas-efit's escape-level replay (revision f009d75e), the nova columns
from the field-swap per-loop sweep. On JT-60SA sides the comparison is rule
consistency only, because no reconstruction there is trusted.</p>

<h2 id="efit-escape-level-comparison-method">Method</h2>
<p>The escape level is imas-efit's maximin, over axis-to-vessel paths inside the
obstacle-aware vessel domain, of polarity&middot;&psi;. Its label is LIMITED (1)
when the bottleneck is on the wall and DIVERTED (2) at an interior saddle. The
first three columns of each row are the escape label, the refined escape level
and the refined bottleneck (R, Z). Nova's columns are its label, its boundary
level (the same maximin read by its own topology rule) and the contact it admits
&mdash; an interior saddle with position, or a wall contact at nova's own
boundary level with the position its read admitted. The strongest wall maximum
over the whole vessel is not the contact: on the four MAST-U limited sides it
sits on a lobe the axis-connected component never reaches and rises above the
admitted level, so a limited row reports the admitted level, not that maximum.
The gap is nova's boundary minus the escape level, absolute and divided
by imas-efit's axis-to-boundary span. The escape axis column names the axis the
escape walk started from; a flag marks the one side that uses the tracked maximum
rather than the recorded axis.</p>

<h2 id="efit-escape-level-comparison-table">(1) Per-side comparison</h2>
{side_table(rows)}

<h2 id="efit-escape-level-comparison-summary-table">(1) Label agreement by machine</h2>
{summary_table(rows)}
<p>Agreement counts a side only where nova qualified a boundary decision; a side
nova left unqualified is counted separately and is not an agreement or a
disagreement. JT-60SA carries no trusted reconstruction, so its counts are rule
consistency only.</p>

<h2 id="efit-escape-level-comparison-diagnoses">(2) Disagreement diagnoses</h2>
{diagnosis_list(rows)}

<h2 id="efit-escape-level-comparison-figures">(3) Poloidal panels</h2>
<p>Each panel is redrawn with the record's wall as its separate indexed loops and
the escape bottleneck overlaid as its own marker, through the nova.media painters.
The panel field is the record's own spline; the boundary levels of both codes and
their fixed points are drawn as in the axis-adjudication evidence.</p>
{figure_links()}
</main></body></html>  """


def build_fragment(rows, commits, verdict):
    n_agree = sum(1 for r in rows if r["nova_label"] == r["efit_label"])
    n_compared = sum(
        1 for r in rows if r["nova_label"] not in ("unqualified", "absent")
    )
    n_unqualified = sum(1 for r in rows if r["nova_label"] == "unqualified")
    return f"""<section id="efit-escape-level-comparison">
<h2 id="efit-escape-level-comparison-heading">Escape-level rule comparison on
imas-efit's 26 trace sides</h2>
<p id="efit-escape-level-comparison-intro">Nova's boundary read and imas-efit's
escape-level rule are compared rule against rule on the 26 sides of imas-efit's
13 paired trace records, from each code's own recorded output. Neither rule is
changed and no verdict on which is right is stated. Escape labels agree with
nova's qualified decisions on {n_agree} of {n_compared} compared sides; nova
left {n_unqualified} JT-60SA sides unqualified and those are not compared. The
full per-side table, the machine-level agreement counts and the diagnosis for
every disagreement are in
<a href="/nova/research/efit-escape-level-comparison.html">the research page</a>.</p>
<p id="efit-escape-level-comparison-rules">The escape columns come from the
escape-level replay at revision f009d75e; the nova columns from the field-swap
per-loop sweep, whose wall loops reach nova as separate units typed vessel, so
the obstacle wall-contact type is an approximation of the source domain. JT-60SA
sides are rule consistency only, because no reconstruction there is trusted.</p>
<div id="efit-escape-level-comparison-figures">
{figure_links()}
</div>
<p id="efit-escape-level-comparison-commits">Measured at {html.escape(commits)}.</p>
</section>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--page", required=True)
    parser.add_argument("--fragment", required=True)
    parser.add_argument("--figure-map", required=True)
    parser.add_argument("--commits", default="")
    parser.add_argument("--verdict", default="partial")
    args = parser.parse_args()

    accuracy, endpoints = parse_escape(
        (ESCAPE_DIR / "evidence-accuracy.log").read_text().splitlines()
    )
    nova = {r["case"]: r for r in json.loads(NOVA_SWEEP.read_text())["rows"]}
    rows = build_rows(accuracy, endpoints, nova)
    if len(rows) != 26:
        raise SystemExit(f"expected 26 sides, built {len(rows)}")

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))

    figure_map = {
        row["case"]: {
            "bottleneck": row["efit_bottleneck"],
            "label": row["efit_label"],
            "level": row["efit_level"],
        }
        for row in rows
    }
    pathlib.Path(args.figure_map).write_text(json.dumps(figure_map, indent=1))

    pathlib.Path(args.page).write_text(build_page(rows, args.commits))
    pathlib.Path(args.fragment).write_text(
        build_fragment(rows, args.commits, args.verdict)
    )
    print("wrote", out / "rows.json")
    print("wrote", args.figure_map)
    print("wrote", args.page)
    print("wrote", args.fragment)
    for row in rows:
        print("ROW", row["case"], row["efit_label"], row["nova_label"], row["loops"])


if __name__ == "__main__":
    main()
