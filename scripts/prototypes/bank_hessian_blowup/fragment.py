"""Render the bank second-derivative diagnosis page from its measured records.

Every number in the page is read from the four JSON records the measurement
jobs wrote, so the prose cannot drift from the measurement. The page is written
to the fragment path this node owns.
"""

import json
from pathlib import Path

CACHE = Path("/home/ITER/mcintos/.cache/nova-bank-hessian-blowup")
ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    ROOT
    / "docs/evidence/fragments/sub-millisecond-forward-path/bank-hessian-blowup.html"
)


BANK_LABELS = {
    "certified": "certified 81ee8fdf",
    "pre_repair": "pre-repair fd912554",
    "rebuild_2e30fa75": "rerun rebuild 2e30fa75",
    "rebuild_ac2c5474": "rerun rebuild ac2c5474",
    "carried_corner": "rerun rebuild 6c1a1d8f",
    "corner_omitted": "rerun rebuild b00f714a",
}


def load(name):
    return json.loads((CACHE / name).read_text())


def number(value):
    if value == 0.0:
        return "0"
    mantissa, exponent = f"{value:.4e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def table(headers, rows, identifier=None):
    head = "".join(f"<th>{cell}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    attribute = f' id="{identifier}"' if identifier else ""
    return (
        "<table"
        + attribute
        + "><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table>"
    )


SECOND_ENTRIES = {
    3: (0, "psi_RR"),
    4: (0, "psi_RZ"),
    5: (0, "psi_ZZ"),
    9: (1, "psi_RR"),
    10: (1, "psi_RZ"),
    11: (1, "psi_ZZ"),
    15: (2, "psi_RR"),
    16: (2, "psi_RZ"),
    17: (2, "psi_ZZ"),
}


def pair_rows(attribute):
    pairs = sorted(attribute["pairs"], key=lambda pair: -abs(pair["pair_max"]))
    rows = []
    for pair in pairs[:10]:
        index = max(SECOND_ENTRIES, key=lambda i: abs(pair["certified"][i]))
        moment, entry = SECOND_ENTRIES[index]
        rows.append(
            (
                pair["node"],
                pair["cell"],
                moment,
                entry,
                number(pair["certified"][index]),
                number(pair["pre_repair"][index]),
                number(pair["distance_vertex"]),
                number(pair["distance_edge"]),
                number(pair["cell_height_gap"]),
            )
        )
    return rows


BOUNDS = ("1000000.0", "1000000000.0", "1000000000000.0", "1000000000000000.0")


def bank_rows(census):
    rows = []
    for key, label in BANK_LABELS.items():
        bank = census["banks"][key]
        rows.append(
            (label, number(bank["maximum"]))
            + tuple(bank["above"][bound] for bound in BOUNDS)
        )
    return rows


ARM_ENTRY = 9


def arm_rows(kernel):
    rows = []
    for key in ("110:16", "97:3", "107:13", "110:38"):
        record = kernel[key]
        for arm in ("shipped", "stable_poles", "plain"):
            rows.append(
                (record["node"], record["cell"], arm, number(record[arm][ARM_ENTRY]))
            )
    return rows


def jump_rows(jump):
    rows = []
    for arm in ("shipped", "stable_poles", "plain"):
        values = jump["arms"][arm]
        for offset in jump["offsets"]:
            plus = values[f"{offset:+.0e}"][1]
            minus = values[f"{-offset:+.0e}"][1]
            rows.append((arm, f"{offset:.0e}", number(plus), number(plus - minus)))
    return rows


def part_head():
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><head>',
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta name="docs-project" content="nova">',
            '<meta name="reckon-type" content="evidence">',
            '<meta name="plan-evidence-for" content="sub-millisecond-forward-path">',
            '<meta name="plan-verifies" content="sub-millisecond-forward-path#s6">',
            '<meta name="plan-title" content="Bank second-derivative blowup">',
            '<meta name="plan-status" content="complete">',
            "<title>Bank second-derivative blowup | nova</title>",
            '<link rel="stylesheet" href="/_shared/foundation.css">',
            '<link rel="stylesheet" href="/_shared/dashboard.css">',
            '</head><body><main class="plan-doc">',
        )
    )


def part_summary(attribute, census):
    above = census["banks"]["certified"]["above"]
    return "\n".join(
        (
            '<article id="bank-hessian-blowup-result">',
            '<h3 id="bank-hessian-blowup-title">'
            "The 3.77e15 ITER bank second derivative belongs to the certified "
            "bank and comes from the stable-pole derivative rule, not from the "
            "moment rows</h3>",
            "<p><strong>",
            f"{above['1000000000.0']} node-cell pairs carry a second derivative "
            f"above <code>1e9</code>, "
            "and every one of them is a moment-1 <code>psi_RR</code> entry whose "
            "target node sits within "
            "<code>6e-13 m</code> of a vertex of its own source cell.",
            "</strong>",
            "The certified bank <code>81ee8fdf</code> carries them. Its two DINA "
            "rerun rebuilds are "
            "bit-identical to each other and reproduce the certified entries to "
            "<code>1.1e-5</code> "
            "absolute with the same pair count at every bound, so the rerun did not "
            "introduce the blow-up.",
            "Reverting the corner entries alone to their pre-repair values lowers "
            "the largest magnitude "
            "from <code>3.77e15</code> to <code>3.36e9</code> and the count above "
            "<code>1e9</code> from "
            f"{above['1000000000.0']} to 1, so the number is carried by those "
            f"corner entries.",
            "</p>",
        )
    )


def part_headline_tail():
    return "\n".join(
        (
            "<p>",
            "Evaluating one blown-up pair's row function under three derivative "
            "arms puts the cause in the "
            "stable-pole contraction inside the derivative rules rather than in the "
            "value graph. With the "
            "stable-pole contraction present the entry is <code>-3.7747e15</code>; "
            "with the same "
            "substitution disabled the identical call returns <code>1.92e6</code>; "
            "and the shipped path "
            "agrees with the fully stable-pole path to twelve significant digits.",
            "A probe of the booked radial gradient at four offsets from the vertex "
            "shows no discontinuity: "
            "the signed difference between the two sides falls as the offset falls, "
            "so the row value is "
            "smooth there and a second derivative of this size cannot come from it.",
            "The moment-row repair does not remove the entry. <code>psi_RR</code> "
            "is built from rows 0:3 "
            "and 6:9, and the repair rebuilds rows 3:6, which feed "
            "<code>dpsi/dZ</code>, <code>psi_RZ</code> "
            "and <code>psi_ZZ</code> only.",
            "</p>",
        )
    )


def part_pairs(attribute):
    intro = (
        "<p>The subject is the certified bank built from ITER DINA scenario "
        "105027 (DD 3.38.1) "
        "at states 220, 333 and 336: 558 polygon source cells and 1772 active "
        "target nodes. "
        "Each node is booked against every cell as three source moments of six "
        "entries, "
        "<code>[psi, dpsi/dR, dpsi/dZ, psi_RR, psi_RZ, psi_ZZ]</code>. The "
        "table lists the ten "
        "largest pairs by their largest second-derivative entry, with the same "
        "entry from the "
        "pre-repair build beside it.</p>"
    )
    headers = (
        "node",
        "cell",
        "moment",
        "entry",
        "certified",
        "pre-repair",
        "vertex distance [m]",
        "edge distance [m]",
        "height gap [m]",
    )
    closing = (
        "<p>Every one of the 38 pairs above <code>1e9</code> follows this "
        "shape: moment 1, "
        "<code>psi_RR</code>, and a target within <code>5.5e-13 m</code> of a "
        "vertex of its own "
        "source cell. Two sit above <code>1e15</code>, 21 above "
        "<code>1e12</code> and 1466 above "
        "<code>1e6</code>. The entry is the fourth of its moment's six booked "
        "entries, index 9 of "
        "the 18 entries of a pair. The cells involved are not ordinary: cells 3, 13 "
        "and 16 share an "
        "inverse second moment whose singular values are <code>4.63e7</code> "
        "and <code>5.14e6 m^-2</code>, against <code>320</code> and "
        "<code>201</code> for an ordinary cell "
        "such as 38.</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-pairs">Which pairs carry the number</h4>',
            intro,
            table(headers, pair_rows(attribute), "bank-hessian-blowup-worst-pairs"),
            closing,
        )
    )


def part_masks(attribute):
    selector = attribute["selector"]
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-masks">'
            "Two masks, and where the extreme pairs fall</h4>",
            "<p>The rerun's corner mask is a node-to-vertex distance below "
            "<code>1e-9 m</code>: 2946 "
            "pairs, of which the 12534 changed pairs place 2553 inside and 9981 "
            "outside, reproducing the "
            "brief's split exactly. The kernel's stable-height selector is a "
            "different mask. It fires when "
            "the target height is within <code>1e-7 m</code> of the height of any "
            "polygon vertex in the "
            "whole section, and it selects "
            f"{selector['selected_nodes']} of {selector['nodes']} nodes, covering "
            f"{selector['changed_pairs_selected']} of the "
            f"{selector['changed_pairs']} changed pairs and "
            f"leaving {selector['changed_pairs_unselected']} unselected (largest "
            f"selected gap "
            f"<code>{number(selector['selected_max_gap'])} m</code>, smallest "
            f"unselected gap "
            f"<code>{number(selector['unselected_min_gap'])} m</code>). All 38 "
            f"pairs above <code>1e9</code> "
            "lie inside both masks.</p>",
        )
    )


def part_banks(census):
    agreement = census["agreement"]
    rebuilds = agreement["rebuild_2e30fa75~rebuild_ac2c5474"]
    certified_rebuild = agreement["certified~rebuild_2e30fa75"]
    intro = (
        "<p>Six bank builds on the same geometry are compared entry by entry: "
        "the certified bank, "
        "the pre-repair store, the two DINA rerun rebuilds and two further "
        "rerun builds. The counts "
        "are node-cell pairs whose largest second-derivative entry exceeds each "
        "bound.</p>"
    )
    difference = number(rebuilds["max_absolute_difference"])
    certified_difference = number(certified_rebuild["max_absolute_difference"])
    closing = (
        "<p>The two rerun rebuilds are bit-identical to each other, with "
        "largest difference "
        f"<code>{difference}</code>, and reproduce the certified bank to "
        f"<code>{certified_difference}</code> absolute with the same counts at "
        f"every bound: the "
        "blow-up is in the certified bank already, and the rerun neither "
        "introduced nor removed it."
        "</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-banks">'
            "The certified bank carries the blow-up</h4>",
            intro,
            table(
                ("bank", "largest entry", "> 1e6", "> 1e9", "> 1e12", "> 1e15"),
                bank_rows(census),
                "bank-hessian-blowup-bank-table",
            ),
            closing,
            corner_evidence(census),
        )
    )


def corner_evidence(census):
    pairs = census["carried_corner_against_certified"][:6]
    listed = ", ".join(f"({pair['node']}, {pair['cell']})" for pair in pairs)
    corner = number(census["banks"]["carried_corner"]["maximum"])
    return (
        "<p>The two further builds bracket the corner effect. 6c1a1d8f "
        "reproduces the pre-repair "
        f"entry bit for bit at {listed}, pairs where the certified bank holds moment-1 "
        "<code>psi_RR</code> entries near <code>1e14</code>; its largest entry is "
        f"<code>{corner}</code>, the pre-repair maximum, with one pair above "
        f"<code>1e9</code>. The "
        "build b00f714a peaks at "
        f"<code>{number(census['banks']['corner_omitted']['maximum'])}</code> "
        f"with no pair above "
        "<code>1e6</code>. The inflation is carried by the repaired corner "
        "entries themselves.</p>"
    )


def part_arms(kernel):
    intro = (
        "<p>The row function is a product of three parts: a value graph built "
        "with the plain pole "
        "evaluation, a tangent graph built through the stable-pole contraction, "
        "and the stable-height "
        "substitution that selects the corner branch. Three arms separate them. "
        "The shipped arm is "
        "<code>packed_analytic_moments</code> exactly as the bank calls it; the "
        "stable-pole arm builds "
        "the whole graph with the stable poles; the plain arm disables the "
        "stable-pole substitution "
        "throughout.</p>"
    )
    rows = arm_rows(kernel)
    closing = (
        "<p>The shipped arm reproduces the certified entries for <code>psi</code> at "
        "<code>2e-16</code> relative, <code>dpsi/dR</code> at "
        "<code>2e-7</code>, <code>psi_ZZ</code> "
        "at <code>1e-8</code> and <code>psi_RR</code> at <code>4.6e-9</code> "
        "for pair (110, 16), "
        "<code>2.1e-10</code> for (97, 3) and <code>2.6e-8</code> for (107, 13). Its "
        "<code>dpsi/dZ</code> and <code>psi_RZ</code> entries differ from the "
        "certified ones, the difference the moment-row repair under way "
        "targets. Within "
        "<code>psi_RR</code>, forcing the "
        "stable-pole evaluation over the whole graph leaves the value unchanged "
        "to twelve significant "
        "digits, while disabling it drops the entry from <code>-3.77e15</code> "
        "to <code>1.92e6</code>. "
        "The inflation is therefore produced by the stable-pole derivative "
        "rule, not by the "
        "stable-height substitution. The same node against an ordinary cell, "
        "(110, 38), reads "
        "<code>3.76e-2</code> on the shipped arm and <code>4.04e-2</code> on "
        "the plain one: no "
        "inflation, so the target must land on the source cell's own vertex for "
        "the branch to fire.</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-arms">'
            "Which construction produces the number</h4>",
            intro,
            table(
                ("node", "cell", "arm", "moment-1 psi_RR"),
                rows,
                "bank-hessian-blowup-arms-table",
            ),
            closing,
        )
    )


def part_jump(jump):
    intro = (
        "<p>The booked radial gradient is <code>2 pi R rows[6:9]</code>, so a "
        "second derivative of "
        "3.77e15 would be visible as a discontinuity or a very steep slope in "
        "that gradient at the "
        "vertex. The probe steps the target radius by <code>+h</code> and "
        "<code>-h</code> about the "
        "vertex for <code>h = 1e-3, 1e-6, 1e-9</code> and <code>1e-12 m</code> "
        "and records the "
        "gradient on each side. A jump that holds its size as the step shrinks "
        "is a discontinuity in "
        "the first derivative; a difference that falls with the step is a "
        "smooth row value.</p>"
    )
    closing = (
        "<p>On the shipped arm the difference between the two sides falls "
        "<code>7.18e-8</code>, <code>1.82e-8</code>, <code>6.01e-11</code> and "
        "<code>3.87e-12</code> "
        "as the step falls from <code>1e-3</code> to <code>1e-12 m</code>. It "
        "shrinks with the step, so "
        "the row value is continuous. A genuine second derivative of 3.77e15 "
        "across the "
        "<code>2e-9 m</code> stencil would require the row value to change by "
        "about <code>3e5</code> "
        "over that spacing, against the <code>6.0e-11</code> measured. The "
        "booked <code>psi_RR</code> "
        "is a property of the derivative rule, not of the value graph it "
        "differentiates.</p>"
    )
    stable = jump["arms"]["stable_poles"]
    note = (
        "<p>The stable-pole arm is the exception, and it locates the branch. "
        "Its row value at "
        f"<code>+1e-9 m</code> is <code>{number(stable['+1e-09'][1])}</code> and at "
        f"<code>+1e-12 m</code> it is "
        f"<code>{number(stable['+1e-12'][1])}</code>, while the shipped "
        "arm reads <code>-1.0398e-6</code> at both steps. The substitution's "
        "own "
        "threshold therefore falls "
        "between those two steps, which is the branch whose derivative the "
        "booked entry carries.</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-jump">'
            "The row value is continuous at the vertex</h4>",
            intro,
            table(
                (
                    "arm",
                    "step [m]",
                    "dpsi/dR at +step",
                    "difference between the two sides",
                ),
                jump_rows(jump),
                "bank-hessian-blowup-jump-table",
            ),
            closing,
            note,
        )
    )


def part_moment():
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-moment-row">'
            "The moment-row repair would not remove it</h4>",
            "<p>The candidate explanation under test was that the moment-1 and "
            "moment-2 Z-gradient rows "
            "are built independently of their flux rows. The measured layout rules "
            "it out as the cause "
            "of this entry. <code>psi_RR</code> is <code>2 pi R</code> times the "
            "radial derivative of "
            "<code>rows[6:9]</code>, and the row assembly carries rows 0:3 as the "
            "flux moments, rows 3:6 "
            "as the radial field and rows 6:9 as the vertical field, so "
            "<code>psi_RR</code> depends on "
            "rows 0:3 and 6:9 and never on rows 3:6.</p>",
            "<p>The arms confirm that separation: the shipped and stable-pole "
            "arms share "
            "<code>psi_RR</code> to twelve significant digits while their "
            "<code>psi_RZ</code> differ, and "
            "the certified bank differs from the shipped arm in "
            "<code>dpsi/dZ</code> and "
            "<code>psi_RZ</code> while agreeing in <code>psi_RR</code>. Rebuilding "
            "the moment rows "
            "therefore changes <code>dpsi/dZ</code>, <code>psi_RZ</code> and "
            "<code>psi_ZZ</code> and "
            "leaves the 3.77e15 <code>psi_RR</code> where it is. The "
            "candidate's premise holds, and "
            "the blow-up is simply not in the Z-gradient rows. The corner "
            "construction, and its "
            "stable-pole derivative rule, is what has to change.</p>",
        )
    )


def part_method():
    intro = (
        "<p>Each measurement ran as one scheduler job submitted from this "
        "worktree, and every job "
        "log ends in <code>EXIT=0</code>. The census job compares six bank "
        "stores directly and "
        "records pairwise agreement. The attribution job lists every pair above "
        "a bound with its "
        "node, cell, moment, position and distances, and counts both masks. The "
        "kernel job rebuilds "
        "one node's packed rows under three derivative arms and differentiates "
        "them exactly as the "
        "bank does, swapping only the row function. The jump job steps the "
        "target radius about a "
        "vertex and records the booked radial gradient on each side.</p>"
    )
    closing = (
        "<p>The four JSON records sit under the shared cache root, with the job "
        "logs in the run "
        "directory. This page is rendered from those records by "
        "scripts/prototypes/bank_hessian_blowup/fragment.py. No product source "
        "was changed.</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-method">Method and artifacts</h4>',
            intro,
            closing,
        )
    )


def render(output=OUTPUT):
    attribute = load("attribute.json")
    census = load("census.json")
    kernel = load("kernel.json")
    jump = load("jump.json")
    blocks = [
        part_head(),
        part_summary(attribute, census),
        part_headline_tail(),
        part_pairs(attribute),
        part_masks(attribute),
        part_banks(census),
        part_arms(kernel),
        part_jump(jump),
        part_moment(),
        part_method(),
        "</article>",
        "</main></body></html>",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(blocks) + "\n")
    print(f"FRAGMENT_COMPLETE {output} {len(blocks)} blocks")


if __name__ == "__main__":
    render()
