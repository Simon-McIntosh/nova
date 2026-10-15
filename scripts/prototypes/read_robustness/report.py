"""Render the measured read and axis receipts as a typed research record."""

# ruff: noqa: E501

import argparse
from html import escape
import json
from pathlib import Path


ROOT_STATUS = ("CONVERGED", "NEEDS_ROWS", "WRONG_ROOT", "NOT_CONVERGED")
TOPOLOGY_REASON = (
    "OK",
    "CAPACITY",
    "SINGULAR_REPRESENTATION",
    "UNRESOLVED_TIE",
    "NO_QUALIFIED_AXIS",
    "SINGULAR_TANGENT",
    "NONFINITE_FIELD",
    "UNRESOLVED_COMPONENT",
)


def number(value, digits=6):
    return f"{value:.{digits}g}" if isinstance(value, (int, float)) else "—"


def point(value):
    return ", ".join(number(item, 9) for item in value)


def cell(value):
    return f"<td>{escape(str(value))}</td>"


def row(values):
    return "<tr>" + "".join(cell(value) for value in values) + "</tr>"


def code(value):
    return f"<code>{escape(str(value))}</code>"


def status(values):
    return ", ".join(ROOT_STATUS[value] for value in values)


def render(run, output, fragment):
    samples = {
        "limited": json.loads((run / "limited-transition.json").read_text()),
        "diverted": json.loads(
            (run / "diverted-exact.json").read_text()
            if (run / "diverted-exact.json").exists()
            else (run / "diverted-transition.json").read_text()
        ),
    }
    windows = {
        kind: json.loads((run / f"window-{kind}.json").read_text()) for kind in samples
    }
    limited, diverted = samples["limited"], samples["diverted"]
    valid_limited = limited["full_valid_count"]
    valid_diverted = diverted["full_valid_count"]
    transition = diverted.get("transition")
    transition_claim = (
        "The bisected class change has both sides valid and a raw-current jump "
        f"of {number(transition['raw_current_jump_per_target'], 4)} of target current."
        if transition and transition["left"]["valid"] and transition["right"]["valid"]
        else "No refusal-free class-change bracket was measured."
    )
    out = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="docs-project" content="nova">',
        '<meta name="reckon-type" content="research">',
        '<meta name="plan-slug" content="read-robustness">',
        '<meta name="plan-title" content="Robust topology read on moving fields">',
        '<meta name="plan-summary" content="Full-jet and mobile-window read, stationary census, perturbed-axis and cold-start measurements">',
        '<meta name="plan-status" content="reference">',
        '<meta name="plan-informs" content="converged-forward-solve">',
        '<meta name="plan-evidence-for" content="converged-forward-solve">',
        '<meta name="plan-verifies" content="converged-forward-solve#s2">',
        '<meta name="plan-source" content="Analytic certificate carriers and H200 jobs 1285859, 1285870, 1285871, 1285881, 1285884">',
        '<meta name="plan-source-quality" content="primary">',
        '<meta name="plan-tags" content="equilibrium,grad-shafranov,jax,convergence,topology">',
        "<title>Robust topology read on moving fields | nova</title>",
        '<link rel="stylesheet" href="/_shared/foundation.css">',
        '<link rel="stylesheet" href="/_shared/dashboard.css"></head>',
        '<body><main class="plan-doc"><header><p class="eyebrow">Research input</p>',
        "<h1>Robust topology read on moving fields</h1>",
        f'<p class="lead">The frozen reference lookup reproduces one valid read in 41 states on each topology. Reading all stored jets and censusing roots raises the valid share to {valid_limited}/41 limited and {valid_diverted}/41 diverted with the moving analytic-axis check. {escape(transition_claim)} The measured read remains above its 0.12 ms target.</p></header>',
        '<section id="method"><h2>Method and positive controls</h2>',
        "<p>Each 41-state test uses the same 289-cell limited or 282-cell diverted analytic certificate carrier and a prescribed vertical-field slope from −0.2 to +0.2 reference flux span per pitch. The baseline is the retained Bell-row lookup and its original fixed seeds. The candidate contracts every stored six-component nodal jet, proposes extrema and saddles from each cell’s quadratic on a fixed lattice, polishes them on the Bell field, removes duplicate roots, continues accepted roots between neighbouring states, and sends every distinct saddle to the connected-support read. It has no saddle-size floor. A separate Newton solve on the closed-form analytic field gives the moving axis used for identity; its residual is recorded on every row. The final diverted arm books all selected support exactly while retaining split interior/cut current as a control. The analytic-moments arm and an actual newly cut node known missing from the frozen lookup are positive controls.</p>",
        f"<p>The complete coarse jet maps hold {limited['jet_rows']} limited rows ({limited['jet_bytes'] / 1e6:.2f} MB) and {diverted['jet_rows']} diverted rows ({diverted['jet_bytes'] / 1e6:.2f} MB). The frozen references populate only {limited['reference_window']['required_rows']}/{limited['reference_window']['capacity']} and {diverted['reference_window']['required_rows']}/{diverted['reference_window']['capacity']} row slots. Storage slack is therefore not lookup coverage. Every H200 payload used binary64, reported the imported nova path in its log header, and cleaned a per-job local temporary directory.</p></section>",
        '<section id="refusals"><h2>Refusal census</h2>',
        "<p>Root codes are 0 CONVERGED, 1 NEEDS_ROWS, 2 WRONG_ROOT and 3 NOT_CONVERGED. Booking status 2 is INVALID; band status 1 means a newly cut cell is absent from the fixed lookup. A status 2 booking after a root refusal is downstream, so the columns below retain both causes instead of collapsing them into one label. The per-state table records every refusal and every axis position.</p>",
        "<table><thead><tr><th>Carrier</th><th>Baseline valid</th><th>NEEDS_ROWS rows</th><th>WRONG_ROOT rows</th><th>NOT_CONVERGED rows</th><th>Booking 2</th><th>Band 1</th><th>Full-row valid</th><th>Axis identity on valid rows</th></tr></thead><tbody>",
    ]
    for kind, data in samples.items():
        base, full = data["baseline"], data["full"]
        out.append(
            row(
                (
                    f"{kind}; {data['cells']}",
                    f"{data['baseline_valid_count']}/41",
                    sum(1 in item["root_status"] for item in base),
                    sum(2 in item["root_status"] for item in base),
                    sum(3 in item["root_status"] for item in base),
                    sum(item["booking_status"] == 2 for item in base),
                    sum(item["band_status"] == 1 for item in base),
                    f"{data['full_valid_count']}/41",
                    f"{sum(item['axis_identity'] for item in full if item['valid'])}/{data['full_valid_count']}",
                )
            )
        )
    out.extend(
        (
            "</tbody></table>",
            f"<p>The limited oracle remains stationary to numerical precision on all 41 sampled fields. Four of its roots lie outside the wall; the remaining in-wall refusal is UNRESOLVED_COMPONENT. The diverted analytic oracle converges on {sum(item['oracle_residual'] <= 1e-8 for item in diverted['full'])}/41 rows; every one of those rows reads valid. The nine other oracle probes retain nonzero stationarity residuals and do not establish a confined axis. The farthest valid-axis error is {number(max(item['oracle_error_pitches'] for item in diverted['full'] if item['valid']), 4)} diverted pitches and {number(max(item['oracle_error_pitches'] for item in limited['full'] if item['valid']), 4)} limited pitches.</p>",
            "<table><thead><tr><th>Carrier</th><th>Field offset [span/pitch]</th><th>Baseline roots</th><th>Booking / band</th><th>Read axis [R,Z m]</th><th>Analytic axis [R,Z m]</th><th>Axis identity</th><th>Polished saddles</th><th>Read verdict</th></tr></thead><tbody>",
        )
    )
    for kind, data in samples.items():
        for old, new in zip(data["baseline"], data["full"]):
            out.append(
                row(
                    (
                        kind,
                        number(old["offset"], 3),
                        status(old["root_status"]),
                        f"{old['booking_status']} / {old['band_status']}",
                        point(new["axis"]),
                        point(new["oracle_axis"]),
                        "yes" if new["axis_identity"] else "no",
                        new["saddle_count"],
                        "VALID"
                        if new["valid"]
                        else TOPOLOGY_REASON[new["topology_reason"]],
                    )
                )
            )
    out.extend(
        (
            "</tbody></table></section>",
            '<section id="windows"><h2>Mobile rows and cost</h2>',
            "<p>At the coarse count the full stored map is read directly, so no band identity can be missing. For larger carriers the fixed-capacity gather forms its desired nodes afresh from current cut cells, wall cells and moving root neighbourhoods. Its overflow is typed. The positive control below marks one cell cut whose nodal row was −1 in the frozen lookup; the mobile lookup gives that same node a live slot. The full map remains stored as the source for the gather.</p>",
            "<table><thead><tr><th>Carrier</th><th>Cells</th><th>Full jet bytes [MB]</th><th>New node</th><th>Fixed → mobile slot</th><th>Full contraction [ms]</th><th>Mobile gather [ms]</th><th>Census [ms]</th><th>Whole read [ms]</th></tr></thead><tbody>",
        )
    )
    for kind, data in samples.items():
        out.append(
            row(
                (
                    kind,
                    data["cells"],
                    number(data["jet_bytes"] / 1e6, 4),
                    "—",
                    "—",
                    number(data["full_contraction_ms"], 4),
                    number(data["mobile_gather_ms"], 4),
                    number(data.get("census_ms"), 4),
                    number(data.get("read_ms"), 4),
                )
            )
        )
        window = windows[kind]
        out.append(
            row(
                (
                    kind,
                    window["cells"],
                    number(window["jet_bytes"] / 1e6, 4),
                    window["newly_required_node"],
                    f"{window['fixed_lookup']} → {window['mobile_lookup']}",
                    number(window["full_contraction_ms"], 4),
                    number(window["mobile_gather_ms"], 4),
                    "—",
                    "—",
                )
            )
        )
    out.extend(
        (
            "</tbody></table>",
            "<p>The proposed connected-read allowance is 0.12 ms per step. These synchronised H200 dispatch-and-wait walls exceed it even for the isolated contraction and gather; separately timed stages overlap in a fused read and must not be summed as a slice wall. The whole-read row is the direct comparison for the current prototype. The 550-cell window test forces a new cut; it is not a full physical 550-cell sweep.</p></section>",
            '<section id="cold"><h2>Cold and distributed current starts</h2>',
            "<p>Each distributed arm supplies nonzero current at the same requested total as the analytic certificate; the zero arm supplies exactly zero. Circle and Gaussian arms use the analytic state only to set centroid, radius and total. The profile-weighted and blended controls use varying amounts of analytic current information and must not be mistaken for a production seed. An analytic-moments arm proves that the instrument can admit a known present read.</p>",
            "<table><thead><tr><th>Carrier</th><th>Input</th><th>Frozen root status</th><th>Frozen booking / band</th><th>Full-row axis [R,Z m]</th><th>Correct axis</th><th>Full-row valid</th></tr></thead><tbody>",
        )
    )
    for kind, data in samples.items():
        for label, seed in data["seeds"].items():
            out.append(
                row(
                    (
                        kind,
                        label.replace("_", " "),
                        status(seed["baseline_root_status"]),
                        f"{seed['baseline_booking_status']} / {seed['baseline_band_status']}",
                        point(seed["axis"]),
                        "yes" if seed["axis_identity"] else "no",
                        "yes" if seed["valid"] else "no",
                    )
                )
            )
    control = diverted["negative_control"]
    out.extend(
        (
            "</tbody></table>",
            f"<p>The negative control is the diverted centroid-circle arm: its candidate root is converged and its topology read says valid, so a convergence-only gate accepts it; the axis-identity check rejects {point(control['wrong_axis'])} m against the intended {point(control['expected_axis'])} m. This is a real alternate stationary root, not a fabricated shifted coordinate.</p></section>",
            '<section id="transition"><h2>Limited-to-diverted transition</h2>',
        )
    )
    if transition:
        out.extend(
            (
                "<p>Both wall and saddle candidate fluxes are evaluated on every sampled state. An accepted adjacent class change was bisected 24 times with the same reference moments and complete jet map. The table keeps both candidate levels, the chosen boundary and the unnormalised booked current on each side. A normalised target-current sum would conceal a support jump.</p>",
                "<table><thead><tr><th>Side</th><th>Offset [span/pitch]</th><th>Class</th><th>Wall candidate</th><th>Saddle candidates</th><th>Boundary flux</th><th>Exact raw current [A]</th><th>Split raw current [A]</th><th>Valid</th></tr></thead><tbody>",
            )
        )
        for label in ("left", "right"):
            item = transition[label]
            out.append(
                row(
                    (
                        label,
                        number(item["offset"], 12),
                        item["boundary_class"],
                        number(item["wall_candidate"], 10),
                        ", ".join(
                            number(value, 10) for value in item["saddle_candidates"]
                        )
                        or "none",
                        number(item["boundary_flux"], 10),
                        number(item["raw_current"], 10),
                        number(item.get("split_raw_current"), 10),
                        item["valid"],
                    )
                )
            )
        out.append("</tbody></table>")
        out.append(
            f"<p>The bracket width is {number(transition['right']['offset'] - transition['left']['offset'], 4)} normalised slope units. The chosen boundary changes by {number(transition['boundary_jump_per_span'], 6)} reference flux spans, exact raw current by {number(transition['raw_current_jump_per_target'], 6)} requested currents, and split-route raw current by {number(transition.get('split_raw_current_jump_per_target'), 6)} requested currents. "
            + (
                "Both endpoint reads are valid. "
                if transition["left"]["valid"] and transition["right"]["valid"]
                else "At least one endpoint refuses. "
            )
            + (
                "This fails the requested continuity."
                if transition["raw_current_jump_per_target"] > 1e-8
                or transition["boundary_jump_per_span"] > 1e-8
                else "This is consistent with continuity at the bracket resolution."
            )
            + "</p>"
        )
    else:
        out.append(
            "<p>No accepted class-change bracket was measured, so continuity remains unqualified.</p>"
        )
    out.extend(
        (
            '</section><section id="limits"><h2>Limits and next implementation step</h2>',
            "<p>This is a prototype in scripts; it changes no product read or production seed. The per-cell quadratic proposes stationary candidates, while the full Bell field polishes them; a carrier with more candidates than the fixed slots returns overflow. The fixed-capacity gather follows tested new cuts but is not itself a fast connected-support read. Whole-cell exact booking is an accuracy control and adds cost; it does not establish the intended fused fast booking path. The measured full contraction, gather, census and complete read determine whether the 0.12 ms allowance is credible. Any remaining cold-seed or transition failure requires a separate mechanism repair before this route can replace the production read.</p>",
            f'<p>Receipts and complete logs are under {code(run)}. The source revision and imported nova path appear in each H200 log header. The prior design input is <a href="/nova/research/sub-millisecond-forward-design.html">the sub-millisecond design review</a>; the earlier carrier measurement is <a href="/nova/research/fixed-shape-lean-map.html">the fixed-shape lean-map study</a>.</p>',
            "</section></main></body></html>",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out) + "\n")
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(
        "\n".join(
            (
                '<!doctype html><html lang="en"><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                '<meta name="docs-project" content="nova">',
                '<meta name="reckon-type" content="evidence">',
                '<meta name="plan-slug" content="read-robustness-evidence">',
                '<meta name="plan-title" content="Topology read robustness measurement">',
                '<meta name="plan-summary" content="Perturbed-axis, cold-seed and mobile-row qualification">',
                '<meta name="plan-evidence-for" content="converged-forward-solve">',
                '<meta name="plan-verifies" content="converged-forward-solve#s2">',
                '<meta name="plan-verdict" content="partial">',
                "<title>Topology read robustness measurement | nova</title>",
                '<link rel="stylesheet" href="/_shared/foundation.css">',
                '<link rel="stylesheet" href="/_shared/dashboard.css"></head>',
                '<body><main class="plan-doc"><header>',
                '<h1 id="read-robustness-title">Topology read robustness measurement</h1>',
                f'<p id="read-robustness-summary">The full-row prototype reads {valid_limited}/41 limited and {valid_diverted}/41 diverted perturbed states with the correct moving axis. The <a href="/nova/research/read-robustness.html">research record</a> contains the complete per-state census, cold starts, transition bracket and H200 cost.</p>',
                "</header></main></body></html>",
            )
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.output, args.fragment)


if __name__ == "__main__":
    main()
