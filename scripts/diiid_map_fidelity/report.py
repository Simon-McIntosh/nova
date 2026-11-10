"""Insert factual measurement tables and figure captions into the research record."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

REASONS = {
    1: "CONVERGED",
    2: "ITERATION_BUDGET_EXHAUSTED",
    3: "NONFINITE_RESIDUAL",
    4: "KRYLOV_ACTION_REFUSED",
    5: "MANIFOLD_ADVANCE_REFUSED",
    6: "SUFFICIENT_DECREASE_REFUSED",
    7: "ACTIVE_SET_CYCLE_DETECTED",
    8: "ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED",
    9: "ACTIVE_SET_STAGNATED",
    10: "ACTIVE_SET_SETTLED",
    11: "ACTIVE_SET_STATIONARY",
    12: "ACTIVE_SET_STEP_REFUSED",
}


def table(headers, rows):
    head = "".join(f"<th>{html.escape(text)}</th>" for text in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def number(value):
    return "—" if value is None else f"{value:.8g}"


def section(document, identity, body):
    pattern = rf'(<section id="{identity}">).*?(</section>)'
    return re.sub(
        pattern, lambda match: match[1] + body + match[2], document, flags=re.S
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("*/row.json")):
        data = json.loads(path.read_text())
        audit = json.loads((path.parent / "audit.json").read_text())
        rows.append((path.parent.name, data, audit))
    assert len(rows) == 4
    ref_rows, shift_rows, solve_rows, components = [], [], [], []
    for identity, data, audit in rows:
        display = (
            "MAST 21978/35"
            if identity.startswith("mast-")
            else "DIII-D " + identity.rsplit("-", 1)[1]
        )
        ref = data["reference"][0]
        ref_rows.append(
            [
                display,
                number(ref["relative"]),
                number(ref["centred_relative"]),
                number(ref["force_n"]),
                number(ref["force_over_hoop"]),
                number(ref["reference_gradient_force_n"]),
                number(ref["hoop_scale_n"]),
                number(ref["amplitude"]),
                ref["core_cells"],
                ref["diverted"],
                audit["maximum_full_state_region"]
                + ": "
                + ", ".join(
                    number(value) for value in audit["maximum_full_state_rz_m"]
                ),
            ]
        )
        for control in data["reference"]:
            shift_rows.append(
                [
                    display,
                    number(control["dz_m"] * 100),
                    number(control["relative"]),
                    number(control["force_n"]),
                    number(control["force_over_hoop"]),
                    control["core_cells"],
                ]
            )
        solved = data["terminal"]
        solve_rows.append(
            [
                display,
                number(solved["residual"]),
                number(solved["direct_terminal_map"]["relative"]),
                solved["converged"],
                solved["qualified"],
                REASONS.get(solved["reason"], str(solved["reason"])),
                solved["topology"],
                number(100 * solved["axis_delta_m"][0]),
                number(100 * solved["axis_delta_m"][1]),
                number(100 * solved["axis_distance_m"]),
                number(
                    None
                    if audit["admitted_xpoint_distance_m"] is None
                    else 100 * audit["admitted_xpoint_distance_m"]
                ),
            ]
        )
        split = data["decomposition"]
        values = split["components"]
        prescribed = values.get(
            "prescribed_circuits_only", values.get("prescribed_passive_only")
        )
        components.append(
            [
                display,
                number(values["plasma_against_label_tare"]["centred_rms_wb"]),
                number(values["external_against_label_remainder"]["centred_rms_wb"]),
                number(ref["target_scaling_delta_sup_wb"]),
                number(ref["current_target_effect"]["relative"]),
                number(values["coils_only"]["sup_wb"]),
                number(prescribed["sup_wb"]),
                split["largest_centred_rms"],
            ]
        )
    document = args.document.read_text()
    body = "<h2>Residual and vertical force at the reference</h2>" + table(
        [
            "Row",
            "Raw residual",
            "Centred residual",
            "Fz [N]",
            "Fz / hoop scale",
            "Reference-gradient Fz [N]",
            "Hoop scale [N]",
            "Current amplitude",
            "Core cells",
            "Map reads diverted",
            "Largest defect R, Z [m]",
        ],
        ref_rows,
    )
    body += "<h3>Rigid displacement controls</h3>" + table(
        ["Row", "ΔZ [cm]", "Raw residual", "Fz [N]", "Fz / hoop scale", "Core cells"],
        shift_rows,
    )
    document = section(document, "reference", body)
    body = "<h2>Newton seeded from EFIT</h2>" + table(
        [
            "Row",
            "Solver residual",
            "Direct map residual",
            "Converged",
            "Qualified",
            "Reason",
            "Topology",
            "Axis ΔR [cm]",
            "Axis ΔZ [cm]",
            "Axis gap [cm]",
            "Verified admitted X-point gap [cm]",
        ],
        solve_rows,
    )
    body += (
        "<p>Axis offsets use the stationary point polished on the reference "
        "flux spline. X-point gaps require an admitted terminal X-point that "
        "matches the lower reference saddle continued through 41 linearly "
        "interpolated fields, with valid saddle Hessians and a maximum 10 cm "
        "local root hop. An empty gap is deliberately unreported. This field "
        "interpolation diagnoses root identity; it is distinct from the executed "
        "Newton trajectory. Raw earlier solve receipts used a half-plane "
        "heuristic; the independent audit supersedes that heuristic for every "
        "reported X-point comparison.</p>"
    )
    renders = json.loads(args.renders.read_text())
    for record in renders:
        terminal = record["terminal"]
        levels = record["flux_levels_wb"]
        body += (
            '<figure><img src="/nova/figures/diiid-reference-map-fidelity/'
            f'{record["file"]}" alt="{record["identity"]}: '
            'EFIT reference, seeded from EFIT and signed map defect">'
            f"<figcaption>{record['identity']}. Seeded from EFIT: "
            f"residual {terminal['residual']:.9g}, "
            f"converged {terminal['converged']}, {terminal['topology']}. "
            f"The two flux panels share {len(levels)} equally spaced physical "
            f"levels from {min(levels):.8g} to {max(levels):.8g} Wb. "
            "Grey dashed contours and grey nulls are EFIT; blue contours and "
            "blue nulls are the terminal state. Triangles mark axes; filled "
            "crosses mark admitted terminal X-points and hollow crosses other "
            "shown saddles. The wall is drawn on all three panels. Defect contours "
            f"are ±0.2, ±0.5 and ±0.8 times the reference grid defect maximum. "
            "The defect is evaluated at EFIT.</figcaption></figure>"
        )
    document = section(document, "newton", body)
    supplement = table(
        [
            "Row",
            "Plasma vs tare centred RMS [Wb]",
            "External vs remainder centred RMS [Wb]",
            "Target scaling sup change [Wb]",
            "Residual without target scaling",
            "Coils sup [Wb]",
            "Prescribed circuits sup [Wb]",
            "Largest centred component",
        ],
        components,
    )
    document = re.sub(
        r'(<section id="decomposition">\s*<h2>Contributor switches</h2>)'
        r"(?:\s*<table>.*?</table>)?",
        lambda match: match[1] + supplement,
        document,
        flags=re.S,
    )
    args.document.write_text(document)
    print(f"REPORT: {len(rows)} complete numerical rows inserted into {args.document}")


if __name__ == "__main__":
    main()
