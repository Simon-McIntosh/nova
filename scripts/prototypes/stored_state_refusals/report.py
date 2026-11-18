"""Assemble the standing-case evidence fragment from the arm records."""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
ROOT = Path(__file__).resolve().parents[3]
FRAGMENT = ROOT / "docs/evidence/fragments/forward-solve-near-stored-states/stored-state-refusals.html"
FIGURES = ROOT / "docs/figures/forward-solve-near-stored-states/stored-state-refusals"
SRC = "/nova/figures/forward-solve-near-stored-states/stored-state-refusals"

LABELS = (
    "mastu-stored-default",
    "mastu-stored-continuation",
    "mastu-displaced-default",
    "mastu-displaced-continuation",
    "tcv-stored-default",
    "iter-stored-default",
)
CASES = {"mastu": "MAST-U 45272 at 0.2 s", "tcv": "TCV workshop at 0.4 s",
         "iter": "ITER 135013 at 75 s"}


def records():
    merged = {}
    for path in sorted(RUN.glob("*/measurement.json")):
        for key, value in json.loads(path.read_text()).items():
            if isinstance(value, dict) and "label" in value:
                merged.setdefault(key, {})
                merged[key].update(value)
            elif key not in merged:
                merged[key] = value
    return merged


def number(value, digits=6):
    if value is None:
        return "nonfinite"
    return f"{float(value):.{digits}g}"


def arm_row(label, record):
    if not record:
        return "<tr><td><code>{}</code></td><td colspan=\"8\">not run</td></tr>".format(label)
    cells = [
        "<code>" + label + "</code>",
        html.escape(str(record.get("case"))),
        html.escape(str(record.get("seed_policy") or "-")),
        number(record.get("residual")),
        str(record.get("termination_reason") + 0) if record.get("termination_reason") is not None else "-",
        "yes" if record.get("qualified") else "no",
        "yes" if record.get("converged") else "no",
        number(record.get("wall_seconds"), 5),
    ]
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def copy_figures(records_map):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        record = records_map.get(label) or {}
        if not record.get("terminal_flux"):
            continue
        svg = Path(record.get("panel", ""))
        png = (Path(record["terminal_flux"]).parent / "comparison"
               / f"{record['case']}-{label}-preview.png")
        for path in (svg, png):
            if path.is_file():
                shutil.copy2(path, FIGURES / path.name)


def figure(label, record, caption):
    if not record or not record.get("terminal_flux"):
        return ""
    name = f"{record.get('case')}-{label}.png"
    return ('<figure id="stored-state-refusals-' + label + '"><img alt="'
            + html.escape(caption) + '" width="100%" src="' + SRC + "/" + name
            + '"><figcaption>' + html.escape(caption) + "</figcaption></figure>")


def mesh_of(records_map, case):
    if case == "mastu":
        return records_map.get("mesh") or {}
    return (records_map.get("meshes") or {}).get(case) or {}


def coefficients_of(records_map, case):
    stored = records_map.get("stored_coefficients") or {}
    return stored.get(case) + [] if isinstance(stored.get(case), list) else []


def case_block(records_map, case):
    mesh = mesh_of(records_map, case)
    route = (records_map.get("default_policy") or {}).get("route", "-")
    lines = [
        "case=" + CASES[case],
        "mesh=" + json.dumps({k: mesh.get(k) for k in ("kind", "nodes", "stencil_rows")}),
        "route=" + str(route) + " with the from_defaults policy",
        "coefficients=" + json.dumps(coefficients_of(records_map, case)),
    ]
    return lines


def policy_table(records_map):
    default = records_map.get("default_policy") or {}
    control = (records_map.get("mastu-stored-continuation") or {}).get("policy") or {}
    keys = ("route", "newton_steps", "gmres_iterations", "active_set_steps",
            "kernel_tolerance", "qualification_tolerance")
    rows = "".join("<tr><td><code>" + key + "</code></td><td>" + str(default.get(key))
                   + "</td><td>" + str(control.get(key)) + "</td></tr>"
                   for key in keys if key in default)
    return ("<table><thead><tr><th>policy field</th><th>default</th>"
            "<th>MAST-U continuation</th></tr></thead><tbody>" + rows
            + "</tbody></table>")



def assemble(records_map):
    parts = ['<section id="stored-state-refusals-record">']
    parts.append("<h2>Forward-solve refusals beside stored imas-efit states</h2>")
    parts.append(
        "<p>Each arm starts the public forward solve from the stored imas-efit flux "
        "map with imas-efit's stored conductor currents and profile coefficients. "
        "The table gives every arm's terminal state.</p>")
    parts.append("<table><thead><tr><th>arm</th><th>case</th><th>seed</th>"
                 "<th>residual</th><th>reason</th><th>qualified</th>"
                 "<th>converged</th><th>wall s</th></tr></thead><tbody>"
                 + "".join(arm_row(label, records_map.get(label)) for label in LABELS)
                 + "</tbody></table>")
    return "\n".join(parts)


def main():
    records_map = records()
    copy_figures(records_map)
    parts = [assemble(records_map)]
    parts.append("<section id=\"stored-state-refusals-inputs\">"
                 "<h3>Inputs, mesh and route per solve</h3>")
    for case in ("mastu", "tcv", "iter"):
        lines = case_block(records_map, case)
        parts.append('<pre id="stored-state-refusals-' + case + '">'
                     + html.escape("\n".join(lines)) + "</pre>")
    parts.append("</section>")
    displaced = records_map.get("displacement") or {}
    if displaced:
        parts.append('<section id="stored-state-refusals-displacement">'
                     "<h3>The displaced coordinate</h3><pre>"
                     + html.escape(json.dumps(displaced, indent=1))
                     + "</pre></section>")
    parts.append('<section id="stored-state-refusals-panels">'
                 "<h3>Terminal states on the stored state</h3>")
    for label in LABELS:
        record = records_map.get(label) or {}
        if not record.get("terminal_flux"):
            continue
        caption = (label + ": terminal residual " + number(record.get("residual"))
                   + ", reason " + str(record.get("termination_reason"))
                   + ", qualified " + str(record.get("qualified")))
        parts.append(figure(label, record, caption))
    parts.append("</section>")
    parts.append("<section id=\"stored-state-refusals-policy\">"
                 "<h3>The default policy beside the continuation policy</h3>")
    parts.append(policy_table(records_map))
    parts.append("</section>")
    parts.append("</section>")
    FRAGMENT.write_text("\n".join(parts) + "\n")
    print("FRAGMENT", FRAGMENT)


if __name__ == "__main__":
    main()
